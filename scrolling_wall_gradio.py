"""Main Gradio interface for the Immersive Room application."""

import gradio as gr
import json
import os
from datetime import datetime
from video_generator import VideoGenerator
from config import (
    DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT, DEFAULT_IMAGES_PER_ROW, DEFAULT_ROWS,
    DEFAULT_PADDING, DEFAULT_SCROLL_SPEED, DEFAULT_FADE_PROBABILITY,
    DEFAULT_FADE_RANDOMNESS, DEFAULT_FADE_DURATION, DEFAULT_BLACK_DURATION,
    DEFAULT_DURATION, DEFAULT_FPS, DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT
)

# Initialize application state
video_generator = VideoGenerator()

# Preset management functions
def save_preset(preset_name, image_folder, video_width, video_height, images_per_row, rows, padding,
                scroll_speed, scroll_direction, fade_probability, fade_randomness, fade_duration, 
                fade_duration_variance, black_duration, black_duration_variance, duration, fps, quality, 
                blend_weight, fade_black_weight, flash_weight,
                fade_in_enabled, fade_out_enabled, fade_in_duration, fade_out_duration, easing_curve,
                dynamic_image_count, dynamic_fade_in_duration, dynamic_fade_out_duration, adaptive_timing, 
                appearance_randomness, transition_smoothness):
    """Save current settings as a preset"""
    if not preset_name:
        return "❌ Please enter a preset name"
    
    # Create presets directory if it doesn't exist
    presets_dir = "presets"
    if not os.path.exists(presets_dir):
        os.makedirs(presets_dir)
    
    # Create preset data
    preset_data = {
        "name": preset_name,
        "created": datetime.now().isoformat(),
        "settings": {
            "image_folder": image_folder,
            "video_width": video_width,
            "video_height": video_height,
            "images_per_row": images_per_row,
            "rows": rows,
            "padding": padding,
            "scroll_speed": scroll_speed,
            "scroll_direction": scroll_direction,
            "fade_probability": fade_probability,
            "fade_randomness": fade_randomness,
            "fade_duration": fade_duration,
            "fade_duration_variance": fade_duration_variance,
            "black_duration": black_duration,
            "black_duration_variance": black_duration_variance,
            "duration": duration,
            "fps": fps,
            "quality": quality,
            "blend_weight": blend_weight,
            "fade_black_weight": fade_black_weight,
            "flash_weight": flash_weight,
            "fade_in_enabled": fade_in_enabled,
            "fade_out_enabled": fade_out_enabled,
            "fade_in_duration": fade_in_duration,
            "fade_out_duration": fade_out_duration,
            "easing_curve": easing_curve,
            "dynamic_image_count": dynamic_image_count,
            "dynamic_fade_in_duration": dynamic_fade_in_duration,
            "dynamic_fade_out_duration": dynamic_fade_out_duration,
            "adaptive_timing": adaptive_timing,
            "appearance_randomness": appearance_randomness,
            "transition_smoothness": transition_smoothness
        }
    }
    
    # Save preset file
    filename = f"{preset_name.replace(' ', '_').lower()}.json"
    filepath = os.path.join(presets_dir, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(preset_data, f, indent=2)
        return f"✅ Preset '{preset_name}' saved successfully!"
    except Exception as e:
        return f"❌ Error saving preset: {str(e)}"

def get_available_presets():
    """Get list of available presets"""
    presets_dir = "presets"
    if not os.path.exists(presets_dir):
        return []
    
    presets = []
    for filename in os.listdir(presets_dir):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(presets_dir, filename), 'r') as f:
                    data = json.load(f)
                    presets.append(data['name'])
            except:
                continue
    
    return sorted(presets)

def load_preset(preset_name):
    """Load settings from a preset"""
    if not preset_name:
        return [None] * 30 + ["Please select a preset"]
    
    presets_dir = "presets"
    filename = f"{preset_name.replace(' ', '_').lower()}.json"
    filepath = os.path.join(presets_dir, filename)
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            settings = data['settings']
            
        return [
            settings['image_folder'],
            settings['video_width'],
            settings['video_height'],
            settings['images_per_row'],
            settings['rows'],
            settings['padding'],
            settings['scroll_speed'],
            settings['scroll_direction'],
            settings['fade_probability'],
            settings['fade_randomness'],
            settings['fade_duration'],
            settings.get('fade_duration_variance', 0.3),
            settings['black_duration'],
            settings.get('black_duration_variance', 0.2),
            settings['duration'],
            settings['fps'],
            settings['quality'],
            settings.get('blend_weight', 0.5),
            settings.get('fade_black_weight', 0.35),
            settings.get('flash_weight', 0.15),
            settings.get('fade_in_enabled', True),
            settings.get('fade_out_enabled', True),
            settings.get('fade_in_duration', 2.0),
            settings.get('fade_out_duration', 2.0),
            settings.get('easing_curve', 'ease-in-out'),
            settings.get('dynamic_image_count', False),
            settings.get('dynamic_fade_in_duration', 3.0),
            settings.get('dynamic_fade_out_duration', 3.0),
            settings.get('adaptive_timing', False),
            settings.get('appearance_randomness', 0.8),
            settings.get('transition_smoothness', 0.6),
            f"✅ Preset '{preset_name}' loaded successfully!"
        ]
    except Exception as e:
        return [None] * 31 + [f"❌ Error loading preset: {str(e)}"]

def delete_preset(preset_name):
    """Delete a saved preset"""
    if not preset_name:
        return "❌ Please select a preset to delete", gr.update(choices=get_available_presets())
    
    presets_dir = "presets"
    filename = f"{preset_name.replace(' ', '_').lower()}.json"
    filepath = os.path.join(presets_dir, filename)
    
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return f"✅ Preset '{preset_name}' deleted successfully!", gr.update(choices=get_available_presets())
        else:
            return f"❌ Preset '{preset_name}' not found", gr.update(choices=get_available_presets())
    except Exception as e:
        return f"❌ Error deleting preset: {str(e)}", gr.update(choices=get_available_presets())

# Create Gradio interface
with gr.Blocks(title="Immersive Room") as demo:
    gr.Markdown("# 🎬 Immersive Room - Cinematic Transitions")
    gr.Markdown("Transform your images into mesmerizing scrolling wall videos with professional-grade dynamic transitions")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Preset management
            with gr.Group():
                gr.Markdown("### 💾 Presets")
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=get_available_presets(),
                        label="Select preset",
                        interactive=True
                    )
                    load_preset_btn = gr.Button("📂 Load", size="sm")
                
                with gr.Row():
                    preset_name_input = gr.Textbox(
                        label="New preset name",
                        placeholder="Enter preset name"
                    )
                    save_preset_btn = gr.Button("💾 Save", size="sm")
                
                with gr.Row():
                    delete_preset_btn = gr.Button("🗑️ Delete Selected", size="sm", variant="stop")
                
                preset_status = gr.Textbox(label="Preset status", interactive=False)
            
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
                
                adaptive_timing = gr.Checkbox(
                    value=True,
                    label="Adaptive Timing",
                    info="Automatically adjust timing based on grid size - more images = longer duration"
                )
                
                with gr.Row():
                    appearance_randomness = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                                    label="Appearance Randomness",
                                                    info="0=center outward, 1=completely random")
                    transition_smoothness = gr.Slider(0.1, 2.0, value=0.6, step=0.1,
                                                    label="Transition Smoothness (sec)",
                                                    info="How long each image takes to fade in/out")
                
                gr.Markdown("💡 **Dynamic Count**: Images appear/disappear with separate timing from opacity fade")
                gr.Markdown("💡 **Adaptive Timing**: Automatically scales duration based on grid size - 6 images = 3s, 54 images = ~7s, 390 images = ~12s")
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
            dynamic_image_count, dynamic_fade_in_duration, dynamic_fade_out_duration, adaptive_timing, appearance_randomness, transition_smoothness
        ],
        outputs=[video_output, status_text],
        api_name="generate_video",
        show_progress=True
    )
    
    # Preset event handlers
    save_preset_btn.click(
        fn=save_preset,
        inputs=[
            preset_name_input, image_folder, video_width, video_height, images_per_row, rows, padding,
            scroll_speed, scroll_direction, fade_probability, fade_randomness, fade_duration, 
            fade_duration_variance, black_duration, black_duration_variance, duration, fps, quality, 
            blend_weight, fade_black_weight, flash_weight,
            fade_in_enabled, fade_out_enabled, fade_in_duration, fade_out_duration, easing_curve,
            dynamic_image_count, dynamic_fade_in_duration, dynamic_fade_out_duration, adaptive_timing, 
            appearance_randomness, transition_smoothness
        ],
        outputs=preset_status
    ).then(
        fn=lambda: gr.update(choices=get_available_presets()),
        outputs=preset_dropdown
    )
    
    load_preset_btn.click(
        fn=load_preset,
        inputs=preset_dropdown,
        outputs=[
            image_folder, video_width, video_height, images_per_row, rows, padding,
            scroll_speed, scroll_direction, fade_probability, fade_randomness, fade_duration, 
            fade_duration_variance, black_duration, black_duration_variance, duration, fps, quality, 
            blend_weight, fade_black_weight, flash_weight,
            fade_in_enabled, fade_out_enabled, fade_in_duration, fade_out_duration, easing_curve,
            dynamic_image_count, dynamic_fade_in_duration, dynamic_fade_out_duration, adaptive_timing, 
            appearance_randomness, transition_smoothness, preset_status
        ]
    )
    
    delete_preset_btn.click(
        fn=delete_preset,
        inputs=preset_dropdown,
        outputs=[preset_status, preset_dropdown]
    )

# Launch the app
if __name__ == "__main__":
    import os
    import sys
    
    # Try different ports if the default is busy
    ports_to_try = [7777, 7860, 7861, 7862, 8080, 8081]
    port = int(os.environ.get("GRADIO_SERVER_PORT", DEFAULT_SERVER_PORT))
    if port not in ports_to_try:
        ports_to_try.insert(0, port)
    
    for try_port in ports_to_try:
        try:
            print(f"Attempting to start server on port {try_port}...")
            demo.queue(  # Enable queue for better stability
                concurrency_count=3,
                max_size=20
            ).launch(
                server_name=DEFAULT_SERVER_HOST, 
                server_port=try_port, 
                share=False,
                max_threads=10,
                show_error=True,
                quiet=False,
                prevent_thread_lock=True  # Prevent thread locking issues
            )
            break
        except OSError as e:
            if "Cannot find empty port" in str(e):
                print(f"Port {try_port} is busy, trying next...")
                continue
            else:
                print(f"Error: {e}")
                raise
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            sys.exit(0)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise