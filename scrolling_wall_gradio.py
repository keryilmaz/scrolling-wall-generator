import os
import random
import numpy as np
from PIL import Image
import gradio as gr
from moviepy import VideoClip
import tempfile

def load_and_prepare_images(image_folder, tile_width, tile_height):
    """Load and resize all images from the folder"""
    if not os.path.exists(image_folder):
        return None
    
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
    
    if not image_files:
        return None
    
    images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
            images.append(np.array(img))
        except Exception as e:
            print(f"Couldn't load {os.path.basename(img_path)}: {str(e)}")
    
    return images

def generate_preview(image_folder, video_width, video_height, images_per_row, rows, padding):
    """Generate a preview frame"""
    # Calculate tile dimensions
    tile_width = (video_width - (images_per_row + 1) * padding) // images_per_row
    tile_height = (video_height - (rows + 1) * padding) // rows
    
    # Load images
    images = load_and_prepare_images(image_folder, tile_width, tile_height)
    if not images:
        return None
    
    # Create preview frame
    frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
    
    for row in range(rows):
        for col in range(images_per_row):
            if images:
                img = random.choice(images)
                x = padding + col * (tile_width + padding)
                y = padding + row * (tile_height + padding)
                frame[y:y+tile_height, x:x+tile_width] = img
    
    return Image.fromarray(frame)

def generate_video(image_folder, video_width, video_height, images_per_row, rows, padding,
                  scroll_speed, scroll_direction, fade_probability, fade_randomness, fade_duration, 
                  black_duration, duration, fps, quality):
    """Generate the scrolling wall video"""
    try:
        # Input validation
        if not image_folder or not os.path.exists(image_folder):
            return None, f"❌ Image folder not found: {image_folder}"
        
        if video_width <= 0 or video_height <= 0:
            return None, "❌ Invalid video dimensions"
        
        if images_per_row <= 0 or rows <= 0:
            return None, "❌ Invalid grid dimensions"
        
        # Calculate tile dimensions
        tile_width = (video_width - (images_per_row + 1) * padding) // images_per_row
        tile_height = (video_height - (rows + 1) * padding) // rows
        
        if tile_width <= 0 or tile_height <= 0:
            return None, "❌ Tiles too small! Reduce padding or grid size"
        
        # Load images
        images = load_and_prepare_images(image_folder, tile_width, tile_height)
        if not images:
            return None, f"❌ No images found in folder: {image_folder}"
        
        print(f"✓ Loaded {len(images)} images, tile size: {tile_width}x{tile_height}")
        
    except Exception as e:
        return None, f"❌ Setup error: {str(e)}"
    
    # Initialize grid state (extend in appropriate direction for seamless scrolling)
    if scroll_direction == "Horizontal":
        grid_cols = images_per_row * 3  # 3x width for horizontal scrolling
        grid_rows = rows
    else:  # Vertical
        grid_cols = images_per_row
        grid_rows = rows * 3  # 3x height for vertical scrolling
    
    grid_state = []
    for row in range(grid_rows):
        row_state = []
        for col in range(grid_cols):
            row_state.append({
                'img': random.choice(images),
                'fade': None,
                'progress': 0.0,
                'next_img': None,
                'fade_offset': random.uniform(0, fade_randomness)
            })
        grid_state.append(row_state)
    
    def make_frame(t):
        # Calculate smooth scroll offset in pixels
        scroll_offset_pixels = scroll_speed * t
        
        # Update fade states
        for row in range(grid_rows):
            for col in range(grid_cols):
                cell = grid_state[row][col]
                
                # Random fade trigger
                if cell['fade'] is None and random.random() < fade_probability + cell['fade_offset'] * fade_probability:
                    cell['fade'] = 'out'
                    cell['progress'] = 0.0
                    cell['next_img'] = random.choice(images)
                
                # Update fade progress
                if cell['fade'] == 'out':
                    cell['progress'] += 1.0 / fps / fade_duration
                    if cell['progress'] >= 1.0:
                        cell['fade'] = 'black'
                        cell['progress'] = 0.0
                elif cell['fade'] == 'black':
                    cell['progress'] += 1.0 / fps / black_duration
                    if cell['progress'] >= 1.0:
                        cell['fade'] = 'in'
                        cell['progress'] = 0.0
                elif cell['fade'] == 'in':
                    cell['progress'] += 1.0 / fps / fade_duration
                    if cell['progress'] >= 1.0:
                        cell['img'] = cell['next_img']
                        cell['fade'] = None
                        cell['progress'] = 0.0
                        cell['next_img'] = None
        
        # Render frame with smooth scrolling
        frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        
        if scroll_direction == "Horizontal":
            # Horizontal smooth scrolling
            tile_stride = tile_width + padding
            
            # Calculate how many tiles we need to render (with extra for smooth edges)
            tiles_needed = images_per_row + 2
            
            for row in range(rows):
                for tile_idx in range(tiles_needed):
                    # Calculate screen position with continuous scrolling
                    screen_x = int(padding + tile_idx * tile_stride - (scroll_offset_pixels % tile_stride))
                    screen_y = int(padding + row * (tile_height + padding))
                    
                    # Calculate which grid tile to use (infinite loop)
                    grid_tile_idx = int(scroll_offset_pixels // tile_stride) + tile_idx
                    grid_col = grid_tile_idx % grid_cols
                    
                    # Only render if tile is at least partially visible
                    if screen_x + tile_width > 0 and screen_x < video_width:
                        cell = grid_state[row][grid_col]
                        
                        # Calculate clipping for partial tiles
                        src_x_start = max(0, -screen_x)
                        src_x_end = min(tile_width, video_width - screen_x)
                        dst_x_start = max(0, screen_x)
                        dst_x_end = min(video_width, screen_x + tile_width)
                        
                        if src_x_start < src_x_end and dst_x_start < dst_x_end:
                            # Get the image for this cell
                            if cell['fade'] == 'out':
                                alpha = 1.0 - cell['progress']
                                img_data = (cell['img'] * alpha).astype(np.uint8)
                            elif cell['fade'] == 'black':
                                img_data = np.zeros_like(cell['img'])
                            elif cell['fade'] == 'in':
                                alpha = cell['progress']
                                img_data = (cell['next_img'] * alpha).astype(np.uint8)
                            else:
                                img_data = cell['img']
                            
                            # Place the (possibly clipped) tile
                            frame[screen_y:screen_y+tile_height, 
                                  dst_x_start:dst_x_end] = img_data[:, src_x_start:src_x_end]
        
        else:  # Vertical smooth scrolling
            tile_stride = tile_height + padding
            
            # Calculate how many tiles we need to render (with extra for smooth edges)
            tiles_needed = rows + 2
            
            for tile_idx in range(tiles_needed):
                # Calculate screen position with continuous scrolling
                screen_y = int(padding + tile_idx * tile_stride - (scroll_offset_pixels % tile_stride))
                
                # Calculate which grid tile to use (infinite loop)
                grid_tile_idx = int(scroll_offset_pixels // tile_stride) + tile_idx
                grid_row = grid_tile_idx % grid_rows
                
                # Only render if tile row is at least partially visible
                if screen_y + tile_height > 0 and screen_y < video_height:
                    for col in range(images_per_row):
                        tile_screen_x = int(padding + col * (tile_width + padding))
                        cell = grid_state[grid_row][col]
                        
                        # Calculate clipping for partial tiles
                        src_y_start = max(0, -screen_y)
                        src_y_end = min(tile_height, video_height - screen_y)
                        dst_y_start = max(0, screen_y)
                        dst_y_end = min(video_height, screen_y + tile_height)
                        
                        if src_y_start < src_y_end and dst_y_start < dst_y_end:
                            # Get the image for this cell
                            if cell['fade'] == 'out':
                                alpha = 1.0 - cell['progress']
                                img_data = (cell['img'] * alpha).astype(np.uint8)
                            elif cell['fade'] == 'black':
                                img_data = np.zeros_like(cell['img'])
                            elif cell['fade'] == 'in':
                                alpha = cell['progress']
                                img_data = (cell['next_img'] * alpha).astype(np.uint8)
                            else:
                                img_data = cell['img']
                            
                            # Place the (possibly clipped) tile
                            frame[dst_y_start:dst_y_end,
                                  tile_screen_x:tile_screen_x+tile_width] = img_data[src_y_start:src_y_end, :]
        
        return frame
    
    # Quality to bitrate mapping
    quality_bitrate = {
        "Low": "5M",
        "Medium": "15M", 
        "High": "30M",
        "Ultra": "50M"
    }
    
    # Create video
    try:
        print("✓ Creating video clip...")
        clip = VideoClip(make_frame, duration=duration)
        
        # Generate output filename
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        print(f"✓ Writing video to: {output_path}")
        clip.write_videofile(
            output_path, 
            fps=fps, 
            codec='libx264', 
            bitrate=quality_bitrate[quality],
            audio=False,
            logger=None
        )
        
        # Verify file was created
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            return output_path, f"✅ Video generated! ({len(images)} images, {file_size:.1f}MB)"
        else:
            return None, "❌ Video file was not created or is empty"
            
    except Exception as e:
        return None, f"❌ Video generation failed: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Scrolling Image Wall Generator") as demo:
    gr.Markdown("# 🎬 Scrolling Image Wall Video Generator")
    gr.Markdown("Create mesmerizing scrolling video walls with dynamic image transitions")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input settings
            with gr.Group():
                gr.Markdown("### 📁 Input Settings")
                image_folder = gr.Textbox(
                    value="/Users/kaaneryilmaz/Downloads/xl",
                    label="Image folder path",
                    info="Path to folder containing images"
                )
            
            # Video dimensions
            with gr.Group():
                gr.Markdown("### 📐 Video Dimensions")
                with gr.Row():
                    video_width = gr.Number(value=1920, label="Width (px)", minimum=640, maximum=10000)
                    video_height = gr.Number(value=1080, label="Height (px)", minimum=480, maximum=4000)
            
            # Grid layout
            with gr.Group():
                gr.Markdown("### 🏗️ Grid Layout")
                images_per_row = gr.Slider(2, 50, value=6, step=1, label="Images per row")
                rows = gr.Slider(1, 20, value=3, step=1, label="Number of rows")
                padding = gr.Slider(0, 100, value=10, step=1, label="Padding (px)")
            
            # Animation settings
            with gr.Group():
                gr.Markdown("### 🎯 Animation Settings")
                scroll_direction = gr.Radio(["Horizontal", "Vertical"], value="Vertical", label="Scroll direction")
                scroll_speed = gr.Slider(5, 500, value=25, step=5, label="Scroll speed (px/sec)", 
                                       info="Lower values = slower, more cinematic")
            
            # Fade effects
            with gr.Group():
                gr.Markdown("### ✨ Fade Effects")
                fade_probability = gr.Slider(0.0, 0.05, value=0.002, step=0.001, 
                                           label="Fade frequency",
                                           info="Probability per frame per cell")
                fade_randomness = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                          label="Fade randomness",
                                          info="0=uniform timing, 1=very random")
                fade_duration = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Fade duration (sec)")
                black_duration = gr.Slider(0.1, 5.0, value=0.5, step=0.1, label="Black duration (sec)")
            
            # Output settings
            with gr.Group():
                gr.Markdown("### 🎥 Output Settings")
                duration = gr.Slider(1, 300, value=10, step=1, label="Video duration (sec)")
                fps = gr.Slider(10, 60, value=30, step=1, label="FPS")
                quality = gr.Radio(["Low", "Medium", "High", "Ultra"], value="High", label="Video quality")
        
        with gr.Column(scale=2):
            # Preview and output
            preview_btn = gr.Button("👁️ Preview Frame", variant="secondary")
            preview_image = gr.Image(label="Preview", type="pil")
            
            generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
            
            with gr.Row():
                video_output = gr.Video(label="Generated Video")
                status_text = gr.Textbox(label="Status", interactive=False)
    
    # Event handlers
    preview_btn.click(
        fn=generate_preview,
        inputs=[image_folder, video_width, video_height, images_per_row, rows, padding],
        outputs=preview_image
    )
    
    generate_btn.click(
        fn=generate_video,
        inputs=[
            image_folder, video_width, video_height, images_per_row, rows, padding,
            scroll_speed, scroll_direction, fade_probability, fade_randomness, fade_duration, 
            black_duration, duration, fps, quality
        ],
        outputs=[video_output, status_text]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8085, share=False)