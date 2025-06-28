# Copied from v1/main_hybrid.py
# --- USER CONFIGURABLE ---
TEXTURE_MODE = "TILE"  # Options: "DIRECT", "TILE", "CROP", "SMART_TILE"
RESOLUTION = (1024, 1024)

import torch
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import os
from transformers import SamModel, SamProcessor
import gc
from skimage.exposure import match_histograms
import shutil
import cv2

# Clear outputs folder at the start
outputs_dir = "outputs"
if os.path.exists(outputs_dir):
    for filename in os.listdir(outputs_dir):
        file_path = os.path.join(outputs_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
else:
    os.makedirs(outputs_dir)

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def smart_tile_texture(texture, wall_width_px, wall_height_px, panel_width_ft, panel_height_ft, scale_factor):
    """
    Smart tiling based on panel dimensions and wall size.
    """
    # Convert panel dimensions from feet to pixels
    panel_width_px = int(panel_width_ft * scale_factor)
    panel_height_px = int(panel_height_ft * scale_factor)
    
    print(f"Panel size in pixels: {panel_width_px} x {panel_height_px}")
    print(f"Wall size in pixels: {wall_width_px} x {wall_height_px}")
    
    # Calculate how many panels we need
    panels_horizontal = max(1, int(np.ceil(wall_width_px / panel_width_px)))
    panels_vertical = max(1, int(np.ceil(wall_height_px / panel_height_px)))
    
    print(f"Panels needed: {panels_horizontal} x {panels_vertical}")
    
    # Resize texture to panel size
    panel_texture = texture.resize((panel_width_px, panel_height_px), resample=Image.Resampling.LANCZOS)
    
    # Create the tiled wall
    tiled_wall = Image.new('RGB', (wall_width_px, wall_height_px))
    
    # Tile the panels
    for y in range(panels_vertical):
        for x in range(panels_horizontal):
            # Calculate position
            pos_x = x * panel_width_px
            pos_y = y * panel_height_px
            
            # Paste the panel
            tiled_wall.paste(panel_texture, (pos_x, pos_y))
    
    return tiled_wall

def step1_prepare_images():
    print(f"Step 1: Preparing images... (Texture mode: {TEXTURE_MODE})")
    room_image = Image.open("inputs/room.jpeg").convert("RGB")
    texture = Image.open("inputs/texture.JPG").convert("RGB")
    room_image = room_image.resize(RESOLUTION)
    # Texture processing
    if TEXTURE_MODE == "DIRECT":
        final_texture = texture.resize(RESOLUTION, resample=Image.Resampling.LANCZOS)
    elif TEXTURE_MODE == "TILE":
        tw, th = texture.size
        tiled = Image.new('RGB', RESOLUTION)
        for y in range(0, RESOLUTION[1], th):
            for x in range(0, RESOLUTION[0], tw):
                tiled.paste(texture, (x, y))
        final_texture = tiled
    elif TEXTURE_MODE == "CROP":
        tw, th = texture.size
        left = max(0, (tw - RESOLUTION[0]) // 2)
        top = max(0, (th - RESOLUTION[1]) // 2)
        right = left + RESOLUTION[0]
        bottom = top + RESOLUTION[1]
        final_texture = texture.crop((left, top, right, bottom)).resize(RESOLUTION, resample=Image.Resampling.LANCZOS)
    else:
        raise ValueError(f"Unknown TEXTURE_MODE: {TEXTURE_MODE}")
    os.makedirs("outputs", exist_ok=True)
    room_image.save("outputs/01_resized_room.png")
    final_texture.save("outputs/01_final_texture.png")
    print("Step 1 complete: Images prepared and saved")
    return room_image, final_texture

def refine_mask_with_edges(mask_image, room_image):
    print("Refining mask with edge detection and morphology...")
    mask_np = np.array(mask_image)
    room_np = np.array(room_image)
    # Edge detection on grayscale room image
    room_gray = cv2.cvtColor(room_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(room_gray, 80, 180)
    # Morphological closing to fill small holes in mask
    kernel = np.ones((7, 7), np.uint8)
    mask_closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    # Morphological opening to remove small noise
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    # Optionally, align mask edge to strong image edges (dilate mask edge, intersect with edges)
    mask_edges = cv2.Canny(mask_opened, 80, 180)
    mask_edges_dilated = cv2.dilate(mask_edges, np.ones((5, 5), np.uint8), iterations=1)
    # Ensure both arrays are binary (0 or 255)
    mask_edges_dilated_bin = (mask_edges_dilated > 0).astype(np.uint8)
    edges_bin = (edges > 0).astype(np.uint8)
    aligned_edges = np.bitwise_and(mask_edges_dilated_bin, edges_bin) * 255
    refined_mask = np.where(aligned_edges, 255, mask_opened)
    # Smooth the final mask
    refined_mask_img = Image.fromarray(refined_mask.astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=6))
    return refined_mask_img

def step2_generate_mask(room_image):
    print("Step 2: Generating wall mask with SAM...")
    device = "cpu"
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    click_point = [[RESOLUTION[0] // 2, RESOLUTION[1] // 2]]
    input_points = torch.tensor([click_point], device=device)
    inputs = sam_processor(room_image, input_points=input_points, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sam_model(**inputs)
    pred_masks = outputs.pred_masks
    if hasattr(pred_masks, 'cpu'):
        pred_masks = pred_masks.cpu()
    mask = sam_processor.image_processor.post_process_masks(
        pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )[0][0][0]
    mask_uint8 = (mask.numpy() * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_uint8, mode="L")
    # Edge feathering
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=12))
    # Mask refinement
    mask_image = refine_mask_with_edges(mask_image, room_image)
    mask_image.save("outputs/02_wall_mask.png")
    clear_memory()
    del sam_model, sam_processor
    clear_memory()
    print("Step 2 complete: Mask generated and saved")
    return mask_image

def step3_paste_texture(room_image, texture, mask_image):
    print("Step 3: Pasting texture onto wall region with direct composite...")
    # Use direct composite to preserve exact texture color and pattern
    result = Image.composite(texture, room_image, mask_image)
    result.save("outputs/03_hybrid_result_improved.png")
    print("Step 3 complete: Improved hybrid result saved as outputs/03_hybrid_result_improved.png")

def run_hybrid_pipeline(room_path, texture_path, output_dir, texture_mode=TEXTURE_MODE, resolution=RESOLUTION, 
                       panel_width_ft=None, panel_height_ft=None, scale_factor=None):
    """
    Run the hybrid texture mapping pipeline with custom input/output paths and parameters.
    """
    global TEXTURE_MODE, RESOLUTION
    old_texture_mode = TEXTURE_MODE
    old_resolution = RESOLUTION
    TEXTURE_MODE = texture_mode
    RESOLUTION = resolution

    # Step 1: Prepare images
    print(f"Step 1: Preparing images... (Texture mode: {TEXTURE_MODE})")
    room_image = Image.open(room_path).convert("RGB")
    texture = Image.open(texture_path).convert("RGB")
    room_image = room_image.resize(RESOLUTION)
    
    # Texture processing with smart tiling
    if TEXTURE_MODE == "SMART_TILE":
        if panel_width_ft is None or panel_height_ft is None or scale_factor is None:
            raise ValueError("SMART_TILE mode requires panel_width_ft, panel_height_ft, and scale_factor parameters")
        final_texture = smart_tile_texture(texture, RESOLUTION[0], RESOLUTION[1], 
                                         panel_width_ft, panel_height_ft, scale_factor)
    elif TEXTURE_MODE == "DIRECT":
        final_texture = texture.resize(RESOLUTION, resample=Image.Resampling.LANCZOS)
    elif TEXTURE_MODE == "TILE":
        tw, th = texture.size
        tiled = Image.new('RGB', RESOLUTION)
        for y in range(0, RESOLUTION[1], th):
            for x in range(0, RESOLUTION[0], tw):
                tiled.paste(texture, (x, y))
        final_texture = tiled
    elif TEXTURE_MODE == "CROP":
        tw, th = texture.size
        left = max(0, (tw - RESOLUTION[0]) // 2)
        top = max(0, (th - RESOLUTION[1]) // 2)
        right = left + RESOLUTION[0]
        bottom = top + RESOLUTION[1]
        final_texture = texture.crop((left, top, right, bottom)).resize(RESOLUTION, resample=Image.Resampling.LANCZOS)
    else:
        raise ValueError(f"Unknown TEXTURE_MODE: {TEXTURE_MODE}")
    
    os.makedirs(output_dir, exist_ok=True)
    room_out = os.path.join(output_dir, "01_resized_room.png")
    texture_out = os.path.join(output_dir, "01_final_texture.png")
    room_image.save(room_out)
    final_texture.save(texture_out)
    print("Step 1 complete: Images prepared and saved")

    # Step 2: Generate mask
    print("Step 2: Generating wall mask with SAM...")
    device = "cpu"
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    click_point = [[RESOLUTION[0] // 2, RESOLUTION[1] // 2]]
    input_points = torch.tensor([click_point], device=device)
    inputs = sam_processor(room_image, input_points=input_points, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sam_model(**inputs)
    pred_masks = outputs.pred_masks
    if hasattr(pred_masks, 'cpu'):
        pred_masks = pred_masks.cpu()
    mask = sam_processor.image_processor.post_process_masks(
        pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )[0][0][0]
    mask_uint8 = (mask.numpy() * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_uint8, mode="L")
    # Edge feathering
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=12))
    # Mask refinement
    mask_image = refine_mask_with_edges(mask_image, room_image)
    mask_out = os.path.join(output_dir, "02_wall_mask.png")
    mask_image.save(mask_out)
    clear_memory()
    del sam_model, sam_processor
    clear_memory()
    print("Step 2 complete: Mask generated and saved")

    # Step 3: Paste texture
    print("Step 3: Pasting texture onto wall region with direct composite...")
    result = Image.composite(final_texture, room_image, mask_image)
    result_out = os.path.join(output_dir, "03_hybrid_result_improved.png")
    result.save(result_out)
    print(f"Step 3 complete: Improved hybrid result saved as {result_out}")

    # Restore globals
    TEXTURE_MODE = old_texture_mode
    RESOLUTION = old_resolution
    return result_out


def main():
    run_hybrid_pipeline(
        room_path="inputs/room.jpeg",
        texture_path="inputs/texture.JPG",
        output_dir="outputs",
        texture_mode=TEXTURE_MODE,
        resolution=RESOLUTION
    )
    print("Hybrid process completed! Check outputs/03_hybrid_result_improved.png for the improved texture mapping result.")

if __name__ == "__main__":
    main() 