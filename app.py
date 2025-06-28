import streamlit as st
import os
from PIL import Image
from main_hybrid import run_hybrid_pipeline, TEXTURE_MODE, RESOLUTION
import io

st.title("Hybrid Texture Mapping Webapp (v2)")

st.markdown("""
Upload a room image and a texture image, then specify the panel dimensions to apply smart tiling to the wall region.
""")

# Uploaders
room_file = st.file_uploader("Upload Room Image", type=["jpg", "jpeg", "png"])
texture_file = st.file_uploader("Upload Texture Image", type=["jpg", "jpeg", "png"])

# Wall dimensions (in feet)
st.subheader("Wall Dimensions")
col1, col2, col3 = st.columns(3)
with col1:
    wall_width_ft = st.number_input("Wall Width (feet)", min_value=1.0, max_value=50.0, value=16.0, step=0.5)
with col2:
    wall_height_ft = st.number_input("Wall Height (feet)", min_value=1.0, max_value=50.0, value=12.0, step=0.5)
with col3:
    resolution = st.selectbox(
        "Wall Resolution",
        options=[(1024, 1024), (2048, 2048)],
        index=0
    )

# Calculate scale factor based on wall size and resolution
wall_width_px, wall_height_px = resolution
scale_factor = min(wall_width_px / wall_width_ft, wall_height_px / wall_height_ft)

# Panel dimensions (in feet)
st.subheader("Panel Dimensions")
col1, col2, col3 = st.columns(3)
with col1:
    panel_width_ft = st.number_input("Panel Width (feet)", min_value=0.1, max_value=20.0, value=4.0, step=0.1)
with col2:
    panel_height_ft = st.number_input("Panel Height (feet)", min_value=0.1, max_value=20.0, value=8.0, step=0.1)
with col3:
    # Display calculated scale factor
    st.metric("Scale (pixels/foot)", f"{scale_factor:.1f}")

# Display panel info
wall_width_px, wall_height_px = resolution
wall_width_ft_calc = wall_width_px / scale_factor
wall_height_ft_calc = wall_height_px / scale_factor

st.info(f"""
**Wall Size:** {wall_width_ft} ft × {wall_height_ft} ft → {wall_width_px} × {wall_height_px} pixels  
**Panel Size:** {panel_width_ft} ft × {panel_height_ft} ft  
**Panels needed:** {wall_width_ft/panel_width_ft:.1f} × {wall_height_ft/panel_height_ft:.1f} panels
""")

if room_file and texture_file:
    # Read files into memory ONCE
    room_bytes = room_file.read()
    texture_bytes = texture_file.read()
    # Show small previews for input images
    try:
        st.image(Image.open(io.BytesIO(room_bytes)), caption="Room Image", width=120)
    except Exception:
        st.warning("Could not preview room image.")
    try:
        st.image(Image.open(io.BytesIO(texture_bytes)), caption="Texture Image", width=120)
    except Exception:
        st.warning("Could not preview texture image.")
    # Save uploads to v2/inputs/
    os.makedirs("inputs", exist_ok=True)
    room_path = os.path.join("inputs", "uploaded_room" + os.path.splitext(room_file.name)[-1])
    texture_path = os.path.join("inputs", "uploaded_texture" + os.path.splitext(texture_file.name)[-1])
    with open(room_path, "wb") as f:
        f.write(room_bytes)
    with open(texture_path, "wb") as f:
        f.write(texture_bytes)

    if st.button("Generate"):
        with st.spinner("Processing..."):
            output_path = run_hybrid_pipeline(
                room_path=room_path,
                texture_path=texture_path,
                output_dir="outputs",
                texture_mode="SMART_TILE",  # New mode for smart tiling
                resolution=resolution,
                panel_width_ft=panel_width_ft,
                panel_height_ft=panel_height_ft,
                scale_factor=scale_factor
            )
        st.success("Done!")
        # Show result image using the new recommended parameter
        st.image(output_path, caption="Result", use_container_width=True)
else:
    st.info("Please upload both a room image and a texture image.")

# Note: use_column_width is deprecated in Streamlit. Use width for fixed size or use_container_width for full width. 