# Interior Texture Mapping Webapp (v2)

A web application that automatically applies textures to wall regions in room images using AI-powered segmentation and smart tiling based on real-world panel dimensions.

## 🎯 What This App Does

This webapp takes a room image and a texture image, then:
1. **Automatically detects wall regions** using AI (Segment Anything Model)
2. **Applies smart tiling** based on real panel dimensions (in feet)
3. **Generates a realistic result** showing how the texture would look on the wall

Perfect for interior designers, contractors, or anyone wanting to visualize how different textures would look on walls before installation.

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- At least 8GB RAM (for AI model processing)
- Internet connection (for first-time model download)

### Installation

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/drewcode/interior-texture-mapping.git
   cd interior-texture-mapping/v2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the webapp**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to the URL shown in the terminal (usually `http://localhost:8501`)

## 📖 How to Use

### Step 1: Upload Images
- **Room Image**: Upload a photo of the room with the wall you want to texture
- **Texture Image**: Upload the texture/panel image you want to apply

### Step 2: Set Wall Dimensions
- **Wall Width**: Enter the actual width of the wall in feet (e.g., 16.0)
- **Wall Height**: Enter the actual height of the wall in feet (e.g., 12.0)
- **Resolution**: Choose image resolution (1024x1024 or 2048x2048)

### Step 3: Set Panel Dimensions
- **Panel Width**: Width of each panel in feet (e.g., 4.0 for 4-foot panels)
- **Panel Height**: Height of each panel in feet (e.g., 8.0 for 8-foot panels)
- **Scale**: Automatically calculated pixels per foot

### Step 4: Generate
Click "Generate" and wait for processing. The app will:
1. Detect the wall region using AI
2. Calculate optimal panel tiling
3. Apply the texture to the wall
4. Show the final result

## 🎨 Smart Tiling Features

### Real-World Dimensions
- Works with actual panel sizes (e.g., 4' × 8' panels)
- Automatically calculates how many panels fit on the wall
- Maintains proper proportions and scale

### Mathematical Precision
- No more arbitrary "tile" or "crop" modes
- Calculates exact panel placement based on dimensions
- Ensures complete wall coverage

### Flexible Scaling
- Adjust pixels per foot for different detail levels
- Works with any wall or panel dimensions
- Real-time calculation updates

## 📊 Example Usage

### Scenario: 16' × 12' Wall with 4' × 8' Panels
- **Wall**: 16 feet wide × 12 feet tall
- **Panels**: 4 feet wide × 8 feet tall
- **Result**: 4 panels horizontally × 1.5 panels vertically
- **Scale**: 64 pixels per foot (for 1024×1024 resolution)

### Input Images
- **Room**: Photo of living room with blank wall
- **Texture**: Wood paneling, stone veneer, or any texture image

### Output
- Realistic visualization of the texture applied to the detected wall region

## 🔧 Technical Details

### AI Components
- **Segment Anything Model (SAM)**: Detects wall regions automatically
- **Edge Detection**: Refines mask boundaries for better alignment
- **Morphological Operations**: Cleans up the mask for smooth results

### Smart Tiling Algorithm
- Converts panel dimensions from feet to pixels
- Calculates optimal panel grid to cover entire wall
- Handles partial panels and edge cases
- Maintains texture quality and proportions

### File Structure
```
v2/
├── app.py              # Main Streamlit webapp
├── main_hybrid.py      # Core pipeline logic
├── requirements.txt    # Python dependencies
├── inputs/            # Uploaded images
├── outputs/           # Generated results
└── README.md          # This file
```

## 🛠️ Customization

### Adding New Textures
Simply upload different texture images to see how they look on your wall.

### Adjusting Panel Sizes
Change the panel dimensions to match your actual materials:
- Standard drywall: 4' × 8'
- Large panels: 4' × 10' or 4' × 12'
- Custom sizes: Any dimensions you specify

### Different Wall Sizes
The app works with any wall dimensions:
- Small accent walls: 8' × 8'
- Large feature walls: 20' × 16'
- Custom dimensions: Enter any size

## 🐛 Troubleshooting

### Common Issues

**"Cannot identify image file" error**
- Make sure your images are in JPG, JPEG, or PNG format
- Try uploading smaller images (under 10MB)

**Slow processing**
- The AI model loads on first use (may take 1-2 minutes)
- Use 1024×1024 resolution for faster processing
- Close other applications to free up memory

**Poor wall detection**
- Ensure the wall is clearly visible in the room image
- Try different room angles if detection fails
- The AI clicks the center of the image to detect walls

### Performance Tips
- Use 1024×1024 resolution for faster processing
- Close other applications during generation
- Ensure good lighting in room photos for better wall detection

## 📝 Requirements

### Python Packages
- `streamlit==1.40.0` - Web framework
- `torch` - PyTorch for AI models
- `transformers` - Hugging Face models
- `PIL` - Image processing
- `numpy` - Numerical operations
- `opencv-python` - Computer vision
- `scikit-image` - Image processing
- And others (see requirements.txt)

### System Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space for models
- **GPU**: Optional, uses CPU by default

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome!

## 📄 License

This project is for educational and personal use.

## 🙏 Acknowledgments

- **Segment Anything Model (SAM)** by Meta AI for wall detection
- **Streamlit** for the web framework
- **Hugging Face** for model hosting

---

**Happy texturing! 🎨** 