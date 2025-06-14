# Immersive Room 🎬

A Gradio-based application for creating mesmerizing scrolling video walls with frequency-synced music and dynamic image transitions. Transform your images into immersive audio-visual experiences perfect for ambient displays, video art, or showcasing collections.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Gradio](https://img.shields.io/badge/gradio-5.33+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features ✨

### Visual Effects
- **Smooth Scrolling**: Pixel-perfect smooth scrolling in both horizontal and vertical directions
- **Infinite Loop**: Seamless infinite scrolling without glitches
- **Dynamic Fade Effects**: Random fade transitions between images
- **Customizable Grid**: Adjustable rows, columns, and padding
- **Quality Presets**: Low, Medium, High, and Ultra quality output
- **Real-time Preview**: See a preview frame before generating the full video
- **Pure Black Background**: Professional look with true black background

### 🎵 Music Synchronization
- **Frequency Analysis**: Advanced spectral analysis of audio files
- **Smart Audio Processing**: Separate audio processing with caching for efficiency
- **Frequency-Responsive Effects**: 
  - High frequencies → More image transitions
  - Low frequencies → Longer, dramatic fades
  - Mid frequencies → Black duration variations
- **Auto Audio Cropping**: Automatically crop/loop audio to match video duration
- **Multiple Audio Formats**: Support for MP3, WAV, FLAC, M4A, etc.

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

4. **Music Sync Workflow** (optional):
   - Upload an audio file
   - Click "Process Audio" to analyze frequencies
   - Enable sync options (fades to frequencies, scroll to energy)

5. Click "Preview Frame" to see a sample
6. Click "Generate Video" to create your immersive audio-visual experience

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

- **Art Galleries**: Display rotating artwork collections with ambient music
- **Digital Signage**: Create immersive displays for events and exhibitions
- **Music Videos**: Generate unique music-reactive visuals for artists
- **Video Backgrounds**: Create dynamic backgrounds synchronized to audio
- **Screen Savers**: Custom screen savers with your photos and favorite music
- **Social Media**: Eye-catching music-synced content for Instagram/TikTok
- **Ambient Displays**: Transform any screen into an immersive art installation

## Technical Details 🔧

The application uses:
- **Gradio**: For the web interface
- **NumPy**: For efficient array operations
- **Pillow (PIL)**: For image processing
- **MoviePy**: For video generation and audio processing
- **Librosa**: For advanced audio analysis and frequency extraction
- **SoundFile**: For audio file I/O
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

Made with ❤️ for immersive audio-visual experiences