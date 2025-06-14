# Scrolling Wall Generator 🎬

A Gradio-based application for creating mesmerizing scrolling video walls with dynamic image transitions. Perfect for creating ambient displays, video art, or showcasing image collections.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Gradio](https://img.shields.io/badge/gradio-5.33+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features ✨

- **Smooth Scrolling**: Pixel-perfect smooth scrolling in both horizontal and vertical directions
- **Infinite Loop**: Seamless infinite scrolling without glitches
- **Dynamic Fade Effects**: Random fade transitions between images
- **Customizable Grid**: Adjustable rows, columns, and padding
- **Quality Presets**: Low, Medium, High, and Ultra quality output
- **Real-time Preview**: See a preview frame before generating the full video
- **Pure Black Background**: Professional look with true black background

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/scrolling-wall-generator.git
cd scrolling-wall-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 💻

1. Run the application:
```bash
python scrolling_wall_gradio.py
```

2. Open your browser and navigate to: `http://localhost:8085`

3. Configure your settings:
   - **Image folder**: Path to your image collection
   - **Video dimensions**: Width and height in pixels
   - **Grid layout**: Number of images per row and number of rows
   - **Scroll settings**: Direction (vertical/horizontal) and speed
   - **Fade effects**: Frequency and duration of transitions
   - **Output settings**: Duration, FPS, and quality

4. Click "Preview Frame" to see a sample
5. Click "Generate Video" to create your scrolling wall video

## Configuration Options 🎛️

### Video Settings
- **Width/Height**: Video resolution (default: 1920x1080)
- **Images per row**: 2-50 images (default: 6)
- **Rows**: 1-20 rows (default: 3)
- **Padding**: Space between images in pixels

### Animation Settings
- **Scroll Direction**: Vertical or Horizontal
- **Scroll Speed**: 5-500 px/sec (default: 25 px/sec for cinematic effect)

### Fade Effects
- **Fade Frequency**: Probability of fade per frame per cell
- **Fade Randomness**: 0=uniform timing, 1=very random
- **Fade Duration**: Time for fade in/out transitions
- **Black Duration**: Time image stays black during transition

### Output Settings
- **Duration**: Video length in seconds
- **FPS**: Frames per second (10-60)
- **Quality**: Low (5M), Medium (15M), High (30M), Ultra (50M) bitrate

## Example Use Cases 🎨

- **Art Galleries**: Display rotating artwork collections
- **Digital Signage**: Create ambient displays for events
- **Video Backgrounds**: Generate unique backgrounds for videos
- **Screen Savers**: Create custom screen savers with your photos
- **Social Media**: Eye-catching content for Instagram/TikTok

## Technical Details 🔧

The application uses:
- **Gradio**: For the web interface
- **NumPy**: For efficient array operations
- **Pillow (PIL)**: For image processing
- **MoviePy**: For video generation
- **FFmpeg**: For video encoding (via MoviePy)

## Requirements 📋

- Python 3.8+
- FFmpeg (automatically installed with imageio-ffmpeg)
- 4GB+ RAM recommended for HD videos
- Sufficient disk space for video output

## Troubleshooting 🔍

**Port already in use**: The app runs on port 8085 by default. If this port is busy, modify the last line of `scrolling_wall_gradio.py`:
```python
demo.launch(server_name="127.0.0.1", server_port=YOUR_PORT, share=False)
```

**Memory issues**: For large videos or many images, try:
- Reducing video resolution
- Using fewer images in the grid
- Lowering the FPS
- Generating shorter videos

**No images found**: Ensure your image folder contains `.jpg`, `.jpeg`, `.png`, `.gif`, or `.bmp` files.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- Built with [Gradio](https://gradio.app/)
- Video processing powered by [MoviePy](https://zulko.github.io/moviepy/)
- Inspired by video art and ambient display projects

---

Made with ❤️ by [Your Name]