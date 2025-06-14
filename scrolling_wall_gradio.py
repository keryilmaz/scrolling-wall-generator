import os
import random
import numpy as np
from PIL import Image
import gradio as gr
from moviepy import VideoClip
import tempfile

# Audio processing for music sync
try:
    import librosa
    import soundfile as sf
    AUDIO_SUPPORT = True
except ImportError:
    AUDIO_SUPPORT = False
    print("Warning: librosa not installed. Music sync features disabled.")

# Global state for processed audio
processed_audio_data = {
    'original_file': None,
    'processed_file': None,
    'frequency_maps': None,
    'sample_rate': None,
    'original_duration': None,
    'analysis_fps': 30
}

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

def analyze_audio_spectrum(audio_file, fps=30, duration=10):
    """Analyze audio file for frequency spectrum (high/mid/low frequencies)"""
    if not AUDIO_SUPPORT or not audio_file:
        return None, None, None, None
    
    try:
        print(f"✓ Analyzing audio spectrum: {audio_file}")
        
        # Load audio file
        y, sr = librosa.load(audio_file, duration=duration)
        
        # Compute spectrogram using Short-Time Fourier Transform
        hop_length = int(sr / fps)  # Hop length to match video FPS
        stft = librosa.stft(y, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # Define frequency bands (in Hz)
        frequencies = librosa.fft_frequencies(sr=sr)
        low_freq_mask = frequencies <= 250      # Bass/sub-bass: 0-250 Hz
        mid_freq_mask = (frequencies > 250) & (frequencies <= 4000)  # Mids: 250-4000 Hz  
        high_freq_mask = frequencies > 4000     # Highs: 4000+ Hz
        
        # Calculate energy in each frequency band over time
        low_energy = np.mean(magnitude[low_freq_mask, :], axis=0)
        mid_energy = np.mean(magnitude[mid_freq_mask, :], axis=0)
        high_energy = np.mean(magnitude[high_freq_mask, :], axis=0)
        
        # Normalize to 0-100 scale
        def normalize_to_100(data):
            if np.max(data) > 0:
                normalized = (data / np.max(data)) * 100
                # Apply some smoothing to avoid too rapid changes
                window_size = max(1, fps // 10)  # Smooth over ~0.1 seconds
                if len(normalized) > window_size:
                    normalized = np.convolve(normalized, np.ones(window_size)/window_size, mode='same')
                return normalized.astype(np.float32)
            return np.zeros_like(data, dtype=np.float32)
        
        low_map = normalize_to_100(low_energy)
        mid_map = normalize_to_100(mid_energy)
        high_map = normalize_to_100(high_energy)
        
        # Ensure arrays match video duration
        target_frames = int(duration * fps)
        
        def resize_to_target(data, target_size):
            if len(data) == 0:
                return np.zeros(target_size)
            from scipy.interpolate import interp1d
            if len(data) == 1:
                return np.full(target_size, data[0])
            x_old = np.linspace(0, 1, len(data))
            x_new = np.linspace(0, 1, target_size)
            f = interp1d(x_old, data, kind='linear', fill_value='extrapolate')
            return f(x_new)
        
        # Install scipy if not available, otherwise use simple interpolation
        try:
            from scipy.interpolate import interp1d
            low_map = resize_to_target(low_map, target_frames)
            mid_map = resize_to_target(mid_map, target_frames) 
            high_map = resize_to_target(high_map, target_frames)
        except ImportError:
            # Simple interpolation fallback
            low_map = np.interp(np.linspace(0, len(low_map)-1, target_frames), 
                               np.arange(len(low_map)), low_map)
            mid_map = np.interp(np.linspace(0, len(mid_map)-1, target_frames),
                               np.arange(len(mid_map)), mid_map) 
            high_map = np.interp(np.linspace(0, len(high_map)-1, target_frames),
                                np.arange(len(high_map)), high_map)
        
        print(f"✓ Spectral analysis complete:")
        print(f"  - Low freq (0-250Hz): avg={np.mean(low_map):.1f}, max={np.max(low_map):.1f}")
        print(f"  - Mid freq (250-4kHz): avg={np.mean(mid_map):.1f}, max={np.max(mid_map):.1f}")
        print(f"  - High freq (4kHz+): avg={np.mean(high_map):.1f}, max={np.max(high_map):.1f}")
        print(f"  - Total frames: {len(low_map)}")
        
        return low_map, mid_map, high_map, sr
        
    except Exception as e:
        print(f"❌ Audio spectrum analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def process_and_cache_audio(audio_file, analysis_fps=30):
    """Process audio file and cache the results for video generation"""
    global processed_audio_data
    
    if not AUDIO_SUPPORT:
        return "❌ Audio processing not available (librosa not installed)"
    
    if not audio_file or not os.path.exists(audio_file):
        return "❌ No audio file provided or file not found"
    
    try:
        print(f"🎵 PROCESSING AUDIO: {os.path.basename(audio_file)}")
        
        # Load the full audio file to get its duration
        y_full, sr = librosa.load(audio_file)
        full_duration = len(y_full) / sr
        
        print(f"✓ Audio loaded: {full_duration:.1f} seconds, {sr} Hz")
        
        # Analyze the entire audio file for frequency content
        print("🔍 Analyzing frequency spectrum across entire audio...")
        
        # Use a longer analysis for better frequency resolution
        hop_length = int(sr / analysis_fps)  # Hop length to match desired FPS
        stft = librosa.stft(y_full, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # Define frequency bands
        frequencies = librosa.fft_frequencies(sr=sr)
        low_freq_mask = frequencies <= 250      # Bass/sub-bass: 0-250 Hz
        mid_freq_mask = (frequencies > 250) & (frequencies <= 4000)  # Mids: 250-4000 Hz  
        high_freq_mask = frequencies > 4000     # Highs: 4000+ Hz
        
        # Calculate energy in each frequency band over time
        low_energy = np.mean(magnitude[low_freq_mask, :], axis=0)
        mid_energy = np.mean(magnitude[mid_freq_mask, :], axis=0)
        high_energy = np.mean(magnitude[high_freq_mask, :], axis=0)
        
        # Normalize to 0-100 scale with smoothing
        def normalize_to_100(data):
            if np.max(data) > 0:
                normalized = (data / np.max(data)) * 100
                # Apply smoothing to avoid rapid changes
                window_size = max(1, analysis_fps // 10)  # ~0.1 seconds
                if len(normalized) > window_size:
                    normalized = np.convolve(normalized, np.ones(window_size)/window_size, mode='same')
                return normalized.astype(np.float32)
            return np.zeros_like(data, dtype=np.float32)
        
        low_map = normalize_to_100(low_energy)
        mid_map = normalize_to_100(mid_energy)
        high_map = normalize_to_100(high_energy)
        
        # Cache the processed data
        processed_audio_data.update({
            'original_file': audio_file,
            'processed_file': audio_file,  # We'll crop this later as needed
            'frequency_maps': (low_map, mid_map, high_map),
            'sample_rate': sr,
            'original_duration': full_duration,
            'analysis_fps': analysis_fps
        })
        
        # Show analysis results
        total_frames = len(low_map)
        timeline_duration = total_frames / analysis_fps
        
        print(f"✅ Audio processing complete!")
        print(f"  📊 Frequency analysis:")
        print(f"    - {total_frames} frames analyzed ({timeline_duration:.1f}s timeline)")
        print(f"    - Low freq (0-250Hz): avg={np.mean(low_map):.1f}, max={np.max(low_map):.1f}")
        print(f"    - Mid freq (250-4kHz): avg={np.mean(mid_map):.1f}, max={np.max(mid_map):.1f}")
        print(f"    - High freq (4kHz+): avg={np.mean(high_map):.1f}, max={np.max(high_map):.1f}")
        
        # Statistical insights
        high_peaks = np.sum(high_map > 70)
        bass_drops = np.sum(low_map > 80)
        mid_variance = np.std(mid_map)
        
        print(f"  🎶 Musical characteristics:")
        print(f"    - High frequency peaks: {high_peaks} moments")
        print(f"    - Heavy bass moments: {bass_drops} drops") 
        print(f"    - Mid-range dynamics: {mid_variance:.1f} variation")
        
        return (f"✅ Audio processed successfully!\n"
                f"📄 File: {os.path.basename(audio_file)}\n"
                f"⏱️ Duration: {full_duration:.1f} seconds\n"
                f"📊 Analysis: {total_frames} frequency frames\n"
                f"🎵 High peaks: {high_peaks}, Bass drops: {bass_drops}\n"
                f"💫 Ready for video generation with music sync!")
                
    except Exception as e:
        print(f"❌ Audio processing failed: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Audio processing failed: {str(e)}"

def get_cropped_audio_for_video(duration):
    """Get audio file cropped to video duration"""
    global processed_audio_data
    
    if not processed_audio_data['original_file']:
        return None
    
    try:
        print(f"🎵 Cropping audio to {duration} seconds for video...")
        
        # Load original audio
        y, sr = librosa.load(processed_audio_data['original_file'])
        
        # Crop to video duration
        target_samples = int(duration * sr)
        if len(y) > target_samples:
            y_cropped = y[:target_samples]
            print(f"✓ Audio trimmed from {len(y)/sr:.1f}s to {duration}s")
        elif len(y) < target_samples:
            # Loop the audio to match video duration
            repeats = int(np.ceil(target_samples / len(y)))
            y_repeated = np.tile(y, repeats)
            y_cropped = y_repeated[:target_samples]
            print(f"✓ Audio looped from {len(y)/sr:.1f}s to {duration}s")
        else:
            y_cropped = y
            print(f"✓ Audio duration matches video ({duration}s)")
        
        # Save cropped audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            cropped_path = tmp_file.name
        
        sf.write(cropped_path, y_cropped, sr)
        print(f"✓ Cropped audio saved: {os.path.basename(cropped_path)}")
        
        return cropped_path
        
    except Exception as e:
        print(f"❌ Audio cropping failed: {e}")
        return None

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
                  black_duration, duration, fps, quality, audio_file=None, music_sync_fade=False, music_sync_scroll=False):
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
        
        # STEP 1: USE CACHED AUDIO DATA OR ANALYZE IF NOT CACHED
        low_freq_map, mid_freq_map, high_freq_map, sample_rate = None, None, None, None
        
        # Check if we should use music sync and have cached data
        if music_sync_fade and processed_audio_data['frequency_maps'] is not None:
            print("🎵 USING CACHED AUDIO DATA for music sync...")
            low_map_full, mid_map_full, high_map_full = processed_audio_data['frequency_maps']
            
            # Crop frequency maps to video duration
            total_frames_needed = int(duration * fps)
            frames_available = len(low_map_full)
            
            if frames_available >= total_frames_needed:
                # Crop to video duration
                low_freq_map = low_map_full[:total_frames_needed]
                mid_freq_map = mid_map_full[:total_frames_needed]
                high_freq_map = high_map_full[:total_frames_needed]
                print(f"✓ Frequency data cropped to {duration}s ({total_frames_needed} frames)")
            else:
                # Loop frequency data to match video duration
                repeat_factor = int(np.ceil(total_frames_needed / frames_available))
                low_freq_map = np.tile(low_map_full, repeat_factor)[:total_frames_needed]
                mid_freq_map = np.tile(mid_map_full, repeat_factor)[:total_frames_needed]
                high_freq_map = np.tile(high_map_full, repeat_factor)[:total_frames_needed]
                print(f"✓ Frequency data looped to {duration}s ({total_frames_needed} frames)")
            
            sample_rate = processed_audio_data['sample_rate']
            
            print(f"✓ Music sync enabled with cached frequency analysis:")
            print(f"  → {len(low_freq_map)} frames of frequency data for video")
            print(f"  → High freq peaks: {np.sum(high_freq_map > 70)} strong moments")
            print(f"  → Bass drops: {np.sum(low_freq_map > 80)} heavy bass moments")
            
            if music_sync_scroll:
                # Adjust scroll speed based on overall audio energy
                avg_energy = (np.mean(low_freq_map) + np.mean(mid_freq_map) + np.mean(high_freq_map)) / 3
                energy_multiplier = 0.5 + (avg_energy / 100) * 1.5  # Scale 0.5x to 2x based on energy
                scroll_speed = max(5, int(scroll_speed * energy_multiplier))
                print(f"✓ Scroll speed adjusted to: {scroll_speed} px/sec (energy factor: {energy_multiplier:.2f})")
        
        elif music_sync_fade and audio_file and AUDIO_SUPPORT:
            print("⚠️ Music sync requested but no cached audio data found.")
            print("🎵 Please use 'Process Audio' button first for better performance!")
            print("🔄 Falling back to real-time analysis...")
            low_freq_map, mid_freq_map, high_freq_map, sample_rate = analyze_audio_spectrum(audio_file, fps, duration)
        
        elif processed_audio_data['original_file'] and not music_sync_fade:
            print("🎵 Cached audio available - will add audio track only (sync disabled)")
        
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
                
                # Determine fade trigger and parameters based on frequency analysis
                should_fade = False
                current_fade_duration = fade_duration
                
                if music_sync_fade and high_freq_map is not None:
                    current_frame = int(t * fps)
                    if current_frame < len(high_freq_map):
                        # STEP 2: USE PRE-PROCESSED FREQUENCY DATA FOR THIS EXACT MOMENT
                        # Get frequency values for this exact frame (0-100 scale)
                        high_freq = high_freq_map[current_frame]
                        mid_freq = mid_freq_map[current_frame] 
                        low_freq = low_freq_map[current_frame]
                        
                        # Show real-time frequency mapping (every 30 frames = 1 second)
                        if current_frame % 30 == 0:
                            time_sec = current_frame / fps
                            print(f"🎵 Frame {current_frame} ({time_sec:.1f}s): H={high_freq:.0f} M={mid_freq:.0f} L={low_freq:.0f}")
                        
                        # HIGH FREQUENCY controls fade probability (number of images fading)
                        # Higher frequency = more fades
                        high_factor = high_freq / 100.0  # 0.0 to 1.0
                        fade_multiplier = 0.1 + (high_factor * 2.0)  # 0.1x to 2.1x base probability
                        music_fade_prob = (fade_probability + cell['fade_offset'] * fade_probability) * fade_multiplier
                        
                        # Add some threshold for dramatic effect
                        if high_freq > 70:  # Strong highs trigger more fades
                            music_fade_prob += 0.01
                        
                        if random.random() < music_fade_prob:
                            should_fade = True
                        
                        # LOW FREQUENCY controls fade duration 
                        # Higher bass = longer fades for dramatic effect
                        low_factor = low_freq / 100.0  # 0.0 to 1.0
                        duration_multiplier = 0.5 + (low_factor * 1.5)  # 0.5x to 2.0x base duration
                        current_fade_duration = fade_duration * duration_multiplier
                        
                        # MID FREQUENCY controls black duration variations
                        mid_factor = mid_freq / 100.0
                        current_black_duration = black_duration * (0.8 + mid_factor * 0.4)  # 0.8x to 1.2x
                else:
                    # Original random fade logic when no music sync
                    if random.random() < fade_probability + cell['fade_offset'] * fade_probability:
                        should_fade = True
                    current_black_duration = black_duration
                
                # Apply fade trigger
                if cell['fade'] is None and should_fade:
                    cell['fade'] = 'out'
                    cell['progress'] = 0.0
                    cell['next_img'] = random.choice(images)
                    # Store the current fade duration for this cell
                    cell['fade_duration'] = current_fade_duration
                    cell['black_duration'] = current_black_duration if 'current_black_duration' in locals() else black_duration
                
                # Update fade progress with frequency-adjusted durations
                active_fade_duration = cell.get('fade_duration', fade_duration)
                active_black_duration = cell.get('black_duration', black_duration)
                
                if cell['fade'] == 'out':
                    cell['progress'] += 1.0 / fps / active_fade_duration
                    if cell['progress'] >= 1.0:
                        cell['fade'] = 'black'
                        cell['progress'] = 0.0
                elif cell['fade'] == 'black':
                    cell['progress'] += 1.0 / fps / active_black_duration
                    if cell['progress'] >= 1.0:
                        cell['fade'] = 'in'
                        cell['progress'] = 0.0
                elif cell['fade'] == 'in':
                    cell['progress'] += 1.0 / fps / active_fade_duration
                    if cell['progress'] >= 1.0:
                        cell['img'] = cell['next_img']
                        cell['fade'] = None
                        cell['progress'] = 0.0
                        cell['next_img'] = None
                        # Clear stored durations
                        cell.pop('fade_duration', None)
                        cell.pop('black_duration', None)
        
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
        
        # Add audio if we have cached audio or provided audio file
        audio_source = None
        
        # First priority: use cached audio (already processed)
        if processed_audio_data['original_file']:
            audio_source = get_cropped_audio_for_video(duration)
            if audio_source:
                print(f"✓ Using processed audio cropped to video duration")
        
        # Fallback: use provided audio file
        elif audio_file and os.path.exists(audio_file):
            audio_source = audio_file
            print(f"✓ Using provided audio file: {os.path.basename(audio_file)}")
        
        if audio_source and os.path.exists(audio_source):
            try:
                from moviepy import AudioFileClip
                print(f"✓ Loading audio for video: {os.path.basename(audio_source)}")
                
                # Load audio clip
                audio_clip = AudioFileClip(audio_source)
                
                # Trim audio to match video duration if needed
                if audio_clip.duration > duration:
                    audio_clip = audio_clip.subclipped(0, duration)
                    print(f"✓ Trimmed audio to {duration} seconds")
                elif audio_clip.duration < duration:
                    # Loop audio if it's shorter than video
                    audio_clip = audio_clip.with_duration(duration)
                    print(f"✓ Looped audio to {duration} seconds")
                
                # Set audio to video
                clip = clip.with_audio(audio_clip)
                print(f"✓ Audio track added: {os.path.basename(audio_source)} ({audio_clip.duration:.1f}s)")
                
                # Write with audio
                clip.write_videofile(
                    output_path, 
                    fps=fps, 
                    codec='libx264', 
                    audio_codec='aac',  # Ensure compatible audio codec
                    bitrate=quality_bitrate[quality],
                    logger=None
                )
                
            except Exception as e:
                print(f"⚠️ Could not add audio: {e}")
                print("⚠️ Generating video without audio...")
                # Fallback: generate without audio
                clip.write_videofile(
                    output_path, 
                    fps=fps, 
                    codec='libx264', 
                    bitrate=quality_bitrate[quality],
                    audio=False,
                    logger=None
                )
        else:
            # No audio file provided
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
            sync_info = ""
            if music_sync_fade and high_freq_map is not None:
                avg_high = np.mean(high_freq_map)
                avg_mid = np.mean(mid_freq_map) 
                avg_low = np.mean(low_freq_map)
                sync_info = f", 🎵 frequency-synced (H:{avg_high:.0f} M:{avg_mid:.0f} L:{avg_low:.0f})"
            elif audio_file:
                sync_info = ", 🎵 with audio"
            return output_path, f"✅ Video generated! ({len(images)} images, {file_size:.1f}MB{sync_info})"
        else:
            return None, "❌ Video file was not created or is empty"
            
    except Exception as e:
        return None, f"❌ Video generation failed: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Immersive Room") as demo:
    gr.Markdown("# 🎬 Immersive Room - Audio-Visual Generator")
    music_status = "🎵 with Music Sync" if AUDIO_SUPPORT else ""
    gr.Markdown(f"Transform your images into immersive audio-visual experiences with frequency-synced music and dynamic transitions {music_status}")
    
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
            
            # Music sync settings (only show if audio support is available)
            if AUDIO_SUPPORT:
                with gr.Group():
                    gr.Markdown("### 🎵 Music Synchronization")
                    audio_file = gr.Audio(type="filepath", label="Upload music file (MP3, WAV, etc.)")
                    
                    with gr.Row():
                        process_audio_btn = gr.Button("🔍 Process Audio", variant="secondary", size="sm")
                        audio_status = gr.Textbox(label="Audio Status", interactive=False, value="No audio processed")
                    
                    gr.Markdown("**Step 1**: Upload audio file and click 'Process Audio' to analyze frequencies  \n"
                               "**Step 2**: Enable sync options below and generate video")
                    
                    with gr.Row():
                        music_sync_fade = gr.Checkbox(value=False, label="Sync fades to frequencies", 
                                                    info="High freq = more fades, Low freq = longer fades")
                        music_sync_scroll = gr.Checkbox(value=False, label="Sync scroll to audio energy", 
                                                      info="Adjust scroll speed based on overall audio intensity")
            else:
                # Placeholder when audio support not available
                audio_file = gr.State(None)
                audio_status = gr.State("Audio support not available")
                process_audio_btn = gr.State(None)
                music_sync_fade = gr.State(False)
                music_sync_scroll = gr.State(False)
            
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
    
    # Audio processing handler (only if audio support is available)
    if AUDIO_SUPPORT:
        process_audio_btn.click(
            fn=process_and_cache_audio,
            inputs=[audio_file],
            outputs=audio_status
        )
    
    generate_btn.click(
        fn=generate_video,
        inputs=[
            image_folder, video_width, video_height, images_per_row, rows, padding,
            scroll_speed, scroll_direction, fade_probability, fade_randomness, fade_duration, 
            black_duration, duration, fps, quality, audio_file, music_sync_fade, music_sync_scroll
        ],
        outputs=[video_output, status_text]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8085, share=False)