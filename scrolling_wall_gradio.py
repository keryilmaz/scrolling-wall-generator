"""Main Gradio interface for the Immersive Room application."""

import gradio as gr
from video_generator import VideoGenerator
from config import (
    DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT, DEFAULT_IMAGES_PER_ROW, DEFAULT_ROWS,
    DEFAULT_PADDING, DEFAULT_SCROLL_SPEED, DEFAULT_FADE_PROBABILITY,
    DEFAULT_FADE_RANDOMNESS, DEFAULT_FADE_DURATION, DEFAULT_BLACK_DURATION,
    DEFAULT_DURATION, DEFAULT_FPS, DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT
)

# Initialize application state
video_generator = VideoGenerator()

# Create Gradio interface
with gr.Blocks(title="Immersive Room") as demo:
    gr.Markdown("# 🎬 Immersive Room - Visual Generator")
    gr.Markdown("Transform your images into mesmerizing scrolling wall videos with dynamic fade transitions")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input settings
            with gr.Group():
                gr.Markdown("### 📁 Input Settings")
                image_folder = gr.Textbox(
                    value="",
                    placeholder="e.g., /Users/username/Pictures/wallpapers",
                    label="Image folder path (NOT a single file)",
                    info="Path to FOLDER containing multiple images (JPG, PNG, etc.) - NOT a single image file"
                )
            
            # Video dimensions
            with gr.Group():
                gr.Markdown("### 📐 Video Dimensions")
                with gr.Row():
                    video_width = gr.Number(value=DEFAULT_VIDEO_WIDTH, label="Width (px)", minimum=640, maximum=10000)
                    video_height = gr.Number(value=DEFAULT_VIDEO_HEIGHT, label="Height (px)", minimum=480, maximum=4000)
            
            # Grid layout
            with gr.Group():
                gr.Markdown("### 🏗️ Grid Layout")
                images_per_row = gr.Slider(2, 50, value=DEFAULT_IMAGES_PER_ROW, step=1, label="Images per row")
                rows = gr.Slider(1, 20, value=DEFAULT_ROWS, step=1, label="Number of rows")
                padding = gr.Slider(0, 100, value=DEFAULT_PADDING, step=1, label="Padding (px)")
            
            # Animation settings
            with gr.Group():
                gr.Markdown("### 🎯 Animation Settings")
                scroll_direction = gr.Radio(["Horizontal", "Vertical"], value="Vertical", label="Scroll direction")
                scroll_speed = gr.Slider(5, 500, value=DEFAULT_SCROLL_SPEED, step=5, label="Scroll speed (px/sec)", 
                                       info="Lower values = slower, more cinematic")
            
            
            # Fade effects
            with gr.Group():
                gr.Markdown("### ✨ Fade Effects")
                fade_probability = gr.Slider(0.0, 0.05, value=DEFAULT_FADE_PROBABILITY, step=0.001, 
                                           label="Fade frequency",
                                           info="Probability per frame per cell")
                fade_randomness = gr.Slider(0.0, 1.0, value=DEFAULT_FADE_RANDOMNESS, step=0.05,
                                          label="Fade randomness",
                                          info="0=uniform timing, 1=very random")
                with gr.Row():
                    fade_duration = gr.Slider(0.1, 5.0, value=DEFAULT_FADE_DURATION, step=0.1, 
                                            label="Base fade duration (sec)",
                                            info="Average duration for fade transitions")
                    fade_duration_variance = gr.Slider(0.0, 1.0, value=0.3, step=0.05,
                                                     label="Duration randomness",
                                                     info="0=all same duration, 1=very random (±100%)")
                with gr.Row():
                    black_duration = gr.Slider(0.1, 5.0, value=DEFAULT_BLACK_DURATION, step=0.1, 
                                             label="Base black duration (sec)",
                                             info="Average duration for black moments")
                    black_duration_variance = gr.Slider(0.0, 1.0, value=0.2, step=0.05,
                                                       label="Black randomness", 
                                                       info="0=all same duration, 1=very random (±100%)")
                
                gr.Markdown("💡 **Duration Randomness**: 0.3 = durations vary ±30%, 1.0 = durations vary ±100% (e.g., 1s base → 0.5s to 2s range)")
                
                gr.Markdown("**🎨 Fade Style Mix** - Control the variety of transitions:")
                with gr.Row():
                    blend_weight = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                           label="Direct Blends", 
                                           info="Smooth crossfades between images")
                    fade_black_weight = gr.Slider(0.0, 1.0, value=0.35, step=0.05,
                                                label="Fade to Black",
                                                info="Traditional fade through black")
                    flash_weight = gr.Slider(0.0, 1.0, value=0.15, step=0.05,
                                           label="Flash Effects",
                                           info="Quick bright transitions")
            
            # Cinematic effects
            with gr.Group():
                gr.Markdown("### 🎬 Cinematic Effects")
                with gr.Row():
                    fade_in_enabled = gr.Checkbox(value=True, label="Fade In from Black")
                    fade_out_enabled = gr.Checkbox(value=True, label="Fade Out to Black")
                
                with gr.Row():
                    fade_in_duration = gr.Slider(0.5, 10.0, value=2.0, step=0.1,
                                                label="Fade In Duration (sec)",
                                                info="Time to fill screen from black")
                    fade_out_duration = gr.Slider(0.5, 10.0, value=2.0, step=0.1,
                                                 label="Fade Out Duration (sec)", 
                                                 info="Time to fade to black at end")
                
                easing_curve = gr.Radio(
                    ["ease_in_out", "ease_in", "ease_out", "linear"],
                    value="ease_in_out",
                    label="Easing Curve",
                    info="Controls the acceleration of fade timing"
                )
                
                dynamic_image_count = gr.Checkbox(
                    value=True,
                    label="Dynamic Image Count",
                    info="Start with few images, gradually add more during fade-in, remove during fade-out"
                )
                
                with gr.Row():
                    dynamic_fade_in_duration = gr.Slider(0.5, 15.0, value=3.0, step=0.1,
                                                        label="Dynamic Fade In Duration (sec)",
                                                        info="Time to add all images to scene")
                    dynamic_fade_out_duration = gr.Slider(0.5, 15.0, value=3.0, step=0.1,
                                                         label="Dynamic Fade Out Duration (sec)",
                                                         info="Time to remove all images from scene")
                
                with gr.Row():
                    appearance_randomness = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                                    label="Appearance Randomness",
                                                    info="0=center outward, 1=completely random")
                    transition_smoothness = gr.Slider(0.1, 2.0, value=0.6, step=0.1,
                                                    label="Transition Smoothness (sec)",
                                                    info="How long each image takes to fade in/out")
                
                gr.Markdown("💡 **Dynamic Count**: Images appear/disappear with separate timing from opacity fade")
                gr.Markdown("💡 **Randomness**: 0.8 = mostly random with slight center bias, 1.0 = completely random")
                gr.Markdown("💡 **Easing**: ease_in_out = smooth start & end, ease_in = slow start, ease_out = slow end, linear = constant speed")
            
            # Output settings
            with gr.Group():
                gr.Markdown("### 🎥 Output Settings")
                duration = gr.Slider(1, 300, value=DEFAULT_DURATION, step=1, label="Video duration (sec)")
                fps = gr.Slider(10, 60, value=DEFAULT_FPS, step=1, label="FPS")
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
        fn=video_generator.generate_preview,
        inputs=[image_folder, video_width, video_height, images_per_row, rows, padding],
        outputs=preview_image
    )
    
    generate_btn.click(
        fn=video_generator.generate_video,
        inputs=[
            image_folder, video_width, video_height, images_per_row, rows, padding,
            scroll_speed, scroll_direction, fade_probability, fade_randomness, fade_duration, 
            fade_duration_variance, black_duration, black_duration_variance, duration, fps, quality, 
            blend_weight, fade_black_weight, flash_weight,
            fade_in_enabled, fade_out_enabled, fade_in_duration, fade_out_duration, easing_curve,
            dynamic_image_count, dynamic_fade_in_duration, dynamic_fade_out_duration, appearance_randomness, transition_smoothness
        ],
        outputs=[video_output, status_text]
    )

# Launch the app
if __name__ == "__main__":
    import os
    # Try different ports if the default is busy
    ports_to_try = [7860, 7861, 7862, 8080, 8081]
    port = int(os.environ.get("GRADIO_SERVER_PORT", DEFAULT_SERVER_PORT))
    if port not in ports_to_try:
        ports_to_try.insert(0, port)
    
    for try_port in ports_to_try:
        try:
            demo.launch(server_name=DEFAULT_SERVER_HOST, server_port=try_port, share=False)
            break
        except OSError as e:
            if "Cannot find empty port" in str(e):
                print(f"Port {try_port} is busy, trying next...")
                continue
            else:
                raise