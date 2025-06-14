"""Video generation module for creating scrolling wall videos."""

import os
import random
import tempfile
import numpy as np
from PIL import Image
from moviepy import VideoClip
from typing import Optional, List, Tuple, Dict, Any, Callable
from contextlib import contextmanager
import atexit

from config import (
    SUPPORTED_IMAGE_FORMATS, QUALITY_BITRATES, GRID_MULTIPLIER, TILES_BUFFER
)
from logger import get_logger, log_error_with_context
from temp_file_manager import temp_manager


class ImageLoader:
    """Handles loading and preparing images for video generation."""
    
    @staticmethod
    def load_and_prepare_images(image_folder: str, tile_width: int, tile_height: int) -> Optional[List[np.ndarray]]:
        """
        Load and resize all images from the folder.
        
        Args:
            image_folder: Path to folder containing images
            tile_width: Target width for each tile
            tile_height: Target height for each tile
            
        Returns:
            List of processed images as numpy arrays or None if no images found
        """
        logger = get_logger(__name__)
        if not os.path.exists(image_folder):
            logger.error(f"Image folder not found: {image_folder}")
            return None
        
        image_files = [
            os.path.join(image_folder, f) 
            for f in os.listdir(image_folder) 
            if f.lower().endswith(SUPPORTED_IMAGE_FORMATS)
        ]
        
        if not image_files:
            logger.error(f"No supported images found in: {image_folder}")
            return None
        
        images = []
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
                images.append(np.array(img))
            except Exception as e:
                logger.warning(f"Couldn't load {os.path.basename(img_path)}: {str(e)}")
        
        if not images:
            logger.error("No images could be loaded successfully")
            return None
            
        logger.info(f"Loaded {len(images)} images")
        return images


class GridCell:
    """Represents a single cell in the scrolling grid."""
    
    def __init__(self, img: np.ndarray, fade_offset: float = 0.0):
        self.img = img
        self.fade: Optional[str] = None  # None, 'blend', 'fade_black', 'flash'
        self.progress: float = 0.0
        self.next_img: Optional[np.ndarray] = None
        self.fade_offset = fade_offset
        self.fade_duration: Optional[float] = None
        self.black_duration: Optional[float] = None
        self.fade_style: Optional[str] = None  # 'blend', 'fade_black', 'flash'
        
        # Dynamic appearance tracking
        self.appearance_time: Optional[float] = None  # When this cell started appearing
        self.disappearance_time: Optional[float] = None  # When this cell started disappearing
        self.appearance_alpha: float = 1.0  # Alpha for dynamic appearance (0.0 = invisible, 1.0 = visible)
    
    def reset_fade(self) -> None:
        """Reset fade state after completion."""
        if self.next_img is not None:
            self.img = self.next_img
        self.fade = None
        self.progress = 0.0
        self.next_img = None
        self.fade_duration = None
        self.black_duration = None
        self.fade_style = None



class ImageSelector:
    """Manages smart image selection to prevent duplicates on screen."""
    
    def __init__(self, images: List[np.ndarray]):
        self.images = images
        self.used_indices = set()
        self.available_indices = list(range(len(images)))
        random.shuffle(self.available_indices)
        self.current_pos = 0
        
    def get_next_image(self) -> np.ndarray:
        """Get the next unique image from the pool."""
        if self.current_pos >= len(self.available_indices):
            # Reshuffle when we've used all images
            random.shuffle(self.available_indices)
            self.current_pos = 0
        
        image_idx = self.available_indices[self.current_pos]
        self.current_pos += 1
        return self.images[image_idx]
    
    def get_non_duplicate_image(self, current_grid_images: set) -> np.ndarray:
        """Get an image that's not currently visible on the grid."""
        # Try to find an image not currently in use
        attempts = 0
        max_attempts = min(50, len(self.images))  # Reasonable limit
        
        while attempts < max_attempts:
            candidate_idx = random.randint(0, len(self.images) - 1)
            candidate_img = self.images[candidate_idx]
            
            # Check if this image is already visible (using image identity)
            img_id = id(candidate_img)
            if img_id not in current_grid_images:
                return candidate_img
            
            attempts += 1
        
        # If we can't find a non-duplicate after reasonable attempts,
        # fall back to sequential selection which guarantees no immediate repeats
        return self.get_next_image()
    
    def get_stats(self) -> str:
        """Get statistics about image usage."""
        return f"ImageSelector: {len(self.images)} total images, position {self.current_pos}/{len(self.available_indices)}"


class CinematicController:
    """Handles cinematic fade in/out effects with easing curves."""
    
    @staticmethod
    def ease_in_out(t: float) -> float:
        """Smooth start and end (cubic ease-in-out)."""
        return t * t * (3.0 - 2.0 * t)
    
    @staticmethod
    def ease_in(t: float) -> float:
        """Slow start, fast end (quadratic ease-in)."""
        return t * t
    
    @staticmethod
    def ease_out(t: float) -> float:
        """Fast start, slow end (quadratic ease-out)."""
        return 1.0 - (1.0 - t) * (1.0 - t)
    
    @staticmethod
    def linear(t: float) -> float:
        """Constant speed."""
        return t
    
    @staticmethod
    def get_easing_function(curve_type: str):
        """Get easing function by name."""
        easing_functions = {
            'ease_in_out': CinematicController.ease_in_out,
            'ease_in': CinematicController.ease_in,
            'ease_out': CinematicController.ease_out,
            'linear': CinematicController.linear
        }
        return easing_functions.get(curve_type, CinematicController.ease_in_out)
    
    @staticmethod
    def get_cinematic_progress(
        current_time: float,
        video_duration: float,
        fade_in_enabled: bool,
        fade_out_enabled: bool,
        fade_in_duration: float,
        fade_out_duration: float,
        easing_curve: str
    ) -> float:
        """
        Calculate the cinematic fade progress (0.0 = fully black, 1.0 = fully visible).
        
        Returns:
            Float between 0.0 and 1.0 representing visibility level
        """
        easing_func = CinematicController.get_easing_function(easing_curve)
        
        # Fade in phase
        if fade_in_enabled and current_time < fade_in_duration:
            progress = current_time / fade_in_duration
            return easing_func(progress)
        
        # Fade out phase
        if fade_out_enabled and current_time > (video_duration - fade_out_duration):
            time_into_fadeout = current_time - (video_duration - fade_out_duration)
            progress = time_into_fadeout / fade_out_duration
            return 1.0 - easing_func(progress)
        
        # Full visibility phase (middle of video)
        return 1.0
    
    @staticmethod
    def get_visible_image_count(
        current_time: float,
        video_duration: float,
        fade_in_enabled: bool,
        fade_out_enabled: bool,
        fade_in_duration: float,
        fade_out_duration: float,
        easing_curve: str,
        total_images: int,
        min_images: int = 1
    ) -> int:
        """
        Calculate how many images should be visible based on cinematic progress.
        
        Returns:
            Number of images that should be visible (min_images to total_images)
        """
        cinematic_progress = CinematicController.get_cinematic_progress(
            current_time, video_duration, fade_in_enabled, fade_out_enabled,
            fade_in_duration, fade_out_duration, easing_curve
        )
        
        # Calculate visible count based on progress
        visible_range = total_images - min_images
        visible_count = min_images + int(cinematic_progress * visible_range)
        
        return max(min_images, min(total_images, visible_count))
    
    @staticmethod
    def get_dynamic_visible_image_count(
        current_time: float,
        video_duration: float,
        dynamic_image_count: bool,
        dynamic_fade_in_duration: float,
        dynamic_fade_out_duration: float,
        easing_curve: str,
        total_images: int,
        min_images: int = 1
    ) -> int:
        """
        Calculate how many images should be visible based on separate dynamic timing.
        Independent of cinematic fade timing.
        """
        if not dynamic_image_count:
            return total_images
            
        easing_func = CinematicController.get_easing_function(easing_curve)
        
        # Dynamic fade in phase
        if current_time < dynamic_fade_in_duration:
            progress = current_time / dynamic_fade_in_duration
            eased_progress = easing_func(progress)
            visible_range = total_images - min_images
            visible_count = min_images + int(eased_progress * visible_range)
            return max(min_images, min(total_images, visible_count))
        
        # Dynamic fade out phase
        if current_time > (video_duration - dynamic_fade_out_duration):
            time_into_fadeout = current_time - (video_duration - dynamic_fade_out_duration)
            progress = time_into_fadeout / dynamic_fade_out_duration
            eased_progress = easing_func(progress)
            visible_range = total_images - min_images
            visible_count = total_images - int(eased_progress * visible_range)
            return max(min_images, min(total_images, visible_count))
        
        # Full visibility phase (middle of video)
        return total_images
    
    @staticmethod
    def get_visible_cells(
        grid_rows: int,
        grid_cols: int,
        visible_count: int,
        total_count: int,
        randomness: float = 0.8
    ) -> set:
        """
        Determine which grid cells should be visible based on count.
        Returns set of (row, col) tuples that should be visible.
        Uses randomized center-outward pattern for organic appearance.
        
        Args:
            randomness: 0.0 = perfect center-outward, 1.0 = completely random
        """
        if visible_count >= total_count:
            # All cells visible
            return {(r, c) for r in range(grid_rows) for c in range(grid_cols)}
        
        if visible_count <= 0:
            return set()
        
        # Create list of all cell positions
        center_row, center_col = grid_rows / 2.0, grid_cols / 2.0
        all_cells = []
        
        for r in range(grid_rows):
            for c in range(grid_cols):
                if randomness >= 1.0:
                    # Completely random order
                    sort_key = hash((r, c, visible_count)) % 10000
                else:
                    # Mix of center-outward and randomness
                    center_dist = ((r - center_row) ** 2 + (c - center_col) ** 2) ** 0.5
                    random_factor = hash((r, c, visible_count)) % 1000 / 1000.0
                    sort_key = center_dist * (1.0 - randomness) + random_factor * randomness * 10
                
                all_cells.append((sort_key, r, c))
        
        # Sort by the computed key
        all_cells.sort(key=lambda x: x[0])
        
        # Take the first visible_count cells
        visible_cells = set()
        for i in range(min(visible_count, len(all_cells))):
            _, r, c = all_cells[i]
            visible_cells.add((r, c))
        
        return visible_cells


class VideoGenerator:
    """Handles video generation with scrolling effects and fade transitions."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.image_selector = None
        self.cinematic = CinematicController()
    
    def _update_dynamic_appearance_states(
        self,
        grid_state: List[List[GridCell]], 
        current_time: float,
        visible_cells: set,
        fps: int,
        transition_duration: float = 0.8
    ) -> None:
        """
        Update the dynamic appearance state of all grid cells with smooth transitions.
        
        Args:
            transition_duration: How long it takes for a cell to fade in/out (seconds)
        """
        if visible_cells is None:
            # If no dynamic visibility, ensure all cells are fully visible
            for row in grid_state:
                for cell in row:
                    cell.appearance_alpha = 1.0
                    cell.appearance_time = None
                    cell.disappearance_time = None
            return
        
        grid_rows = len(grid_state)
        grid_cols = len(grid_state[0]) if grid_state else 0
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                cell = grid_state[row][col]
                cell_position = (row, col)
                should_be_visible = cell_position in visible_cells
                
                # Check if visibility state changed
                if should_be_visible:
                    # Cell should be visible
                    if cell.appearance_time is None and cell.disappearance_time is None:
                        # Start appearing
                        cell.appearance_time = current_time
                        cell.disappearance_time = None
                    elif cell.disappearance_time is not None:
                        # Was disappearing, now should appear again
                        # Calculate how far along the disappearance was and reverse it
                        disappear_progress = min(1.0, (current_time - cell.disappearance_time) / transition_duration)
                        # Start appearing from current alpha level
                        cell.appearance_time = current_time - (1.0 - disappear_progress) * transition_duration
                        cell.disappearance_time = None
                        
                else:
                    # Cell should not be visible
                    if cell.appearance_time is not None and cell.disappearance_time is None:
                        # Start disappearing
                        cell.disappearance_time = current_time
                
                # Calculate current appearance alpha
                if cell.disappearance_time is not None:
                    # Currently disappearing
                    disappear_progress = min(1.0, (current_time - cell.disappearance_time) / transition_duration)
                    cell.appearance_alpha = 1.0 - disappear_progress
                elif cell.appearance_time is not None:
                    # Currently appearing or fully appeared
                    appear_progress = min(1.0, (current_time - cell.appearance_time) / transition_duration)
                    cell.appearance_alpha = appear_progress
                else:
                    # Default state (shouldn't happen, but safe fallback)
                    cell.appearance_alpha = 0.0 if not should_be_visible else 1.0
    
    def _validate_parameters(
        self, 
        image_folder: str, 
        video_width: int, 
        video_height: int,
        images_per_row: int, 
        rows: int, 
        padding: int
    ) -> Optional[str]:
        """Validate input parameters."""
        if not image_folder:
            return "❌ Image folder path is required"
            
        if not os.path.exists(image_folder):
            return f"❌ Path not found: {image_folder}"
            
        if not os.path.isdir(image_folder):
            return f"❌ Path is not a folder: {image_folder}. Please provide a FOLDER containing images, not a single image file."
        
        if video_width <= 0 or video_height <= 0:
            return "❌ Invalid video dimensions"
        
        if images_per_row <= 0 or rows <= 0:
            return "❌ Invalid grid dimensions"
        
        # Calculate tile dimensions
        tile_width = (video_width - (images_per_row + 1) * padding) // images_per_row
        tile_height = (video_height - (rows + 1) * padding) // rows
        
        if tile_width <= 0 or tile_height <= 0:
            return "❌ Tiles too small! Reduce padding or grid size"
        
        return None
    
    def _initialize_grid_state(
        self, 
        images: List[np.ndarray], 
        grid_cols: int, 
        grid_rows: int,
        fade_randomness: float
    ) -> List[List[GridCell]]:
        """Initialize the grid state for infinite scrolling with no duplicate images."""
        # Initialize image selector
        self.image_selector = ImageSelector(images)
        
        grid_state = []
        for row in range(grid_rows):
            row_state = []
            for col in range(grid_cols):
                # Use image selector to get unique images
                img = self.image_selector.get_next_image()
                fade_offset = random.uniform(0, fade_randomness)
                row_state.append(GridCell(img, fade_offset))
            grid_state.append(row_state)
        
        self.logger.info(f"Initialized grid with {grid_rows}x{grid_cols} unique images from {len(images)} available")
        return grid_state
    
    def _get_currently_visible_images(self, grid_state: List[List[GridCell]]) -> set:
        """Get set of image IDs currently visible on the grid."""
        visible_images = set()
        for row in grid_state:
            for cell in row:
                if cell.img is not None:
                    visible_images.add(id(cell.img))
                if cell.next_img is not None:
                    visible_images.add(id(cell.next_img))
        return visible_images
    
    def _get_cell_cinematic_alpha(
        self, 
        row: int, 
        col: int, 
        grid_rows: int, 
        grid_cols: int, 
        cinematic_alpha: float,
        fade_in_enabled: bool,
        fade_out_enabled: bool
    ) -> float:
        """Get the cinematic alpha for a specific cell with staggered appearance."""
        if cinematic_alpha >= 1.0:
            return 1.0
        
        if not fade_in_enabled and not fade_out_enabled:
            return 1.0
        
        # Create staggered appearance pattern
        # Images appear from center outward during fade-in
        # Images disappear from edges inward during fade-out
        center_row = grid_rows / 2.0
        center_col = grid_cols / 2.0
        
        # Calculate distance from center (normalized)
        row_dist = abs(row - center_row) / (grid_rows / 2.0)
        col_dist = abs(col - center_col) / (grid_cols / 2.0)
        distance_from_center = (row_dist + col_dist) / 2.0
        
        # Stagger the appearance timing based on distance from center
        stagger_factor = 0.3  # How much to stagger (0.3 = 30% of fade duration)
        cell_delay = distance_from_center * stagger_factor
        
        # Adjust cinematic alpha based on cell position
        adjusted_alpha = (cinematic_alpha - cell_delay) / (1.0 - stagger_factor)
        return max(0.0, min(1.0, adjusted_alpha))
    
    def _update_grid_fade_states(
        self,
        grid_state: List[List[GridCell]],
        t: float,
        fps: int,
        images: List[np.ndarray],
        fade_probability: float,
        fade_duration: float,
        fade_duration_variance: float,
        black_duration: float,
        black_duration_variance: float,
        blend_weight: float = 0.5,
        fade_black_weight: float = 0.35,
        flash_weight: float = 0.15
    ) -> None:
        """Update fade states for all grid cells."""
        grid_rows = len(grid_state)
        grid_cols = len(grid_state[0]) if grid_state else 0
        
        # Get currently visible images to avoid duplicates
        visible_images = self._get_currently_visible_images(grid_state)
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                cell = grid_state[row][col]
                
                # Determine fade parameters
                should_fade = random.random() < fade_probability + cell.fade_offset * fade_probability
                
                # Add randomness to fade durations
                fade_variance_factor = 1.0 + (random.random() - 0.5) * 2 * fade_duration_variance
                fade_variance_factor = max(0.1, fade_variance_factor)  # Ensure minimum duration
                current_fade_duration = fade_duration * fade_variance_factor
                
                black_variance_factor = 1.0 + (random.random() - 0.5) * 2 * black_duration_variance
                black_variance_factor = max(0.1, black_variance_factor)  # Ensure minimum duration
                current_black_duration = black_duration * black_variance_factor
                
                # Log duration variety occasionally for debugging
                if should_fade and random.random() < 0.01:  # 1% chance to log
                    self.logger.debug(f"Fade duration variety: {current_fade_duration:.2f}s (base: {fade_duration:.2f}s, factor: {fade_variance_factor:.2f})")
                
                # Trigger new fade
                if cell.fade is None and should_fade:
                    # Use smart image selection to avoid duplicates
                    cell.next_img = self.image_selector.get_non_duplicate_image(visible_images)
                    cell.fade_duration = current_fade_duration
                    cell.black_duration = current_black_duration
                    
                    # Randomly choose fade style using user-defined weights
                    fade_styles = ['blend', 'fade_black', 'flash']
                    # Normalize weights to ensure they sum to 1
                    total_weight = blend_weight + fade_black_weight + flash_weight
                    if total_weight > 0:
                        fade_weights = [blend_weight/total_weight, fade_black_weight/total_weight, flash_weight/total_weight]
                    else:
                        fade_weights = [0.5, 0.35, 0.15]  # Fallback to defaults
                    cell.fade_style = random.choices(fade_styles, weights=fade_weights)[0]
                    
                    # Set initial fade state based on style
                    if cell.fade_style == 'blend':
                        cell.fade = 'blend'
                    elif cell.fade_style == 'fade_black':
                        cell.fade = 'out'
                    else:  # flash
                        cell.fade = 'flash'
                    
                    cell.progress = 0.0
                
                # Update fade progress
                active_fade_duration = cell.fade_duration or fade_duration
                active_black_duration = cell.black_duration or black_duration
                
                # Update fade progress based on style
                if cell.fade == 'blend':
                    # Direct blend from current to next image
                    cell.progress += 1.0 / fps / active_fade_duration
                    if cell.progress >= 1.0:
                        cell.reset_fade()
                        
                elif cell.fade == 'flash':
                    # Quick flash effect (faster transition)
                    flash_speed = 3.0  # 3x faster than normal fade
                    cell.progress += 1.0 / fps / (active_fade_duration / flash_speed)
                    if cell.progress >= 1.0:
                        cell.reset_fade()
                        
                elif cell.fade == 'out':
                    # Traditional fade to black (part 1: fade out)
                    cell.progress += 1.0 / fps / active_fade_duration
                    if cell.progress >= 1.0:
                        cell.fade = 'black'
                        cell.progress = 0.0
                        
                elif cell.fade == 'black':
                    # Traditional fade to black (part 2: stay black)
                    cell.progress += 1.0 / fps / active_black_duration
                    if cell.progress >= 1.0:
                        cell.fade = 'in'
                        cell.progress = 0.0
                        
                elif cell.fade == 'in':
                    # Traditional fade to black (part 3: fade in)
                    cell.progress += 1.0 / fps / active_fade_duration
                    if cell.progress >= 1.0:
                        cell.reset_fade()
    
    def _render_horizontal_scroll(
        self,
        frame: np.ndarray,
        grid_state: List[List[GridCell]],
        scroll_offset_pixels: float,
        video_width: int,
        video_height: int,
        images_per_row: int,
        rows: int,
        tile_width: int,
        tile_height: int,
        padding: int,
        grid_cols: int,
        cinematic_alpha: float = 1.0,
        fade_in_enabled: bool = False,
        fade_out_enabled: bool = False
    ) -> None:
        """Render frame with horizontal scrolling."""
        tile_stride = tile_width + padding
        tiles_needed = images_per_row + TILES_BUFFER
        
        for row in range(rows):
            for tile_idx in range(tiles_needed):
                screen_x = int(padding + tile_idx * tile_stride - (scroll_offset_pixels % tile_stride))
                screen_y = int(padding + row * (tile_height + padding))
                
                grid_tile_idx = int(scroll_offset_pixels // tile_stride) + tile_idx
                grid_col = grid_tile_idx % grid_cols
                
                if screen_x + tile_width > 0 and screen_x < video_width:
                    cell = grid_state[row][grid_col]
                    
                    # Calculate clipping
                    src_x_start = max(0, -screen_x)
                    src_x_end = min(tile_width, video_width - screen_x)
                    dst_x_start = max(0, screen_x)
                    dst_x_end = min(video_width, screen_x + tile_width)
                    
                    if src_x_start < src_x_end and dst_x_start < dst_x_end:
                        img_data = self._get_cell_image_data(cell)
                        
                        # Apply cinematic staggered fade
                        cell_alpha = self._get_cell_cinematic_alpha(
                            row, grid_col, len(grid_state), grid_cols,
                            cinematic_alpha, fade_in_enabled, fade_out_enabled
                        )
                        
                        # Apply dynamic appearance alpha
                        final_alpha = cell_alpha * cell.appearance_alpha
                        
                        # Only render if there's some visibility
                        if final_alpha > 0.01:
                            if final_alpha < 1.0:
                                img_data = (img_data.astype(np.float32) * final_alpha).astype(np.uint8)
                            
                            frame[screen_y:screen_y+tile_height, dst_x_start:dst_x_end] = img_data[:, src_x_start:src_x_end]
    
    def _render_vertical_scroll(
        self,
        frame: np.ndarray,
        grid_state: List[List[GridCell]],
        scroll_offset_pixels: float,
        video_width: int,
        video_height: int,
        images_per_row: int,
        rows: int,
        tile_width: int,
        tile_height: int,
        padding: int,
        grid_rows: int,
        cinematic_alpha: float = 1.0,
        fade_in_enabled: bool = False,
        fade_out_enabled: bool = False
    ) -> None:
        """Render frame with vertical scrolling."""
        tile_stride = tile_height + padding
        tiles_needed = rows + TILES_BUFFER
        
        for tile_idx in range(tiles_needed):
            screen_y = int(padding + tile_idx * tile_stride - (scroll_offset_pixels % tile_stride))
            grid_tile_idx = int(scroll_offset_pixels // tile_stride) + tile_idx
            grid_row = grid_tile_idx % grid_rows
            
            if screen_y + tile_height > 0 and screen_y < video_height:
                for col in range(images_per_row):
                    tile_screen_x = int(padding + col * (tile_width + padding))
                    cell = grid_state[grid_row][col]
                    
                    # Calculate clipping
                    src_y_start = max(0, -screen_y)
                    src_y_end = min(tile_height, video_height - screen_y)
                    dst_y_start = max(0, screen_y)
                    dst_y_end = min(video_height, screen_y + tile_height)
                    
                    if src_y_start < src_y_end and dst_y_start < dst_y_end:
                        img_data = self._get_cell_image_data(cell)
                        
                        # Apply cinematic staggered fade
                        cell_alpha = self._get_cell_cinematic_alpha(
                            grid_row, col, grid_rows, images_per_row,
                            cinematic_alpha, fade_in_enabled, fade_out_enabled
                        )
                        
                        # Apply dynamic appearance alpha
                        final_alpha = cell_alpha * cell.appearance_alpha
                        
                        # Only render if there's some visibility
                        if final_alpha > 0.01:
                            if final_alpha < 1.0:
                                img_data = (img_data.astype(np.float32) * final_alpha).astype(np.uint8)
                            
                            frame[dst_y_start:dst_y_end, tile_screen_x:tile_screen_x+tile_width] = img_data[src_y_start:src_y_end, :]
    
    def _get_cell_image_data(self, cell: GridCell) -> np.ndarray:
        """Get the current image data for a cell based on its fade state."""
        if cell.fade == 'blend':
            # Direct blend between current and next image
            alpha = cell.progress
            current_contribution = (cell.img * (1.0 - alpha)).astype(np.float32)
            next_contribution = (cell.next_img * alpha).astype(np.float32)
            return (current_contribution + next_contribution).astype(np.uint8)
            
        elif cell.fade == 'flash':
            # Quick flash transition with slight brightness boost
            alpha = cell.progress
            if alpha < 0.3:
                # First 30%: fade out current with slight brightness
                fade_alpha = 1.0 - (alpha / 0.3)
                brightness_boost = 1.0 + (alpha / 0.3) * 0.2  # Up to 20% brighter
                return (cell.img * fade_alpha * brightness_boost).astype(np.uint8)
            elif alpha < 0.7:
                # Middle 40%: blend images with brightness
                blend_alpha = (alpha - 0.3) / 0.4
                brightness = 1.2  # 20% brighter during transition
                current_contribution = (cell.img * (1.0 - blend_alpha) * brightness).astype(np.float32)
                next_contribution = (cell.next_img * blend_alpha * brightness).astype(np.float32)
                return np.clip(current_contribution + next_contribution, 0, 255).astype(np.uint8)
            else:
                # Last 30%: fade in next image
                fade_alpha = (alpha - 0.7) / 0.3
                brightness_boost = 1.2 - (fade_alpha * 0.2)  # Fade brightness back to normal
                return (cell.next_img * fade_alpha * brightness_boost).astype(np.uint8)
            
        elif cell.fade == 'out':
            # Traditional fade to black (fade out current)
            alpha = 1.0 - cell.progress
            return (cell.img * alpha).astype(np.uint8)
            
        elif cell.fade == 'black':
            # Traditional fade to black (stay black)
            return np.zeros_like(cell.img)
            
        elif cell.fade == 'in':
            # Traditional fade to black (fade in next)
            alpha = cell.progress
            return (cell.next_img * alpha).astype(np.uint8)
            
        else:
            # No fade - return current image
            return cell.img
    
    def generate_preview(
        self, 
        image_folder: str, 
        video_width: int, 
        video_height: int,
        images_per_row: int, 
        rows: int, 
        padding: int
    ) -> Optional[Image.Image]:
        """Generate a preview frame."""
        # Validate parameters
        error = self._validate_parameters(image_folder, video_width, video_height, images_per_row, rows, padding)
        if error:
            self.logger.error(error)
            return None
        
        # Calculate tile dimensions
        tile_width = (video_width - (images_per_row + 1) * padding) // images_per_row
        tile_height = (video_height - (rows + 1) * padding) // rows
        
        # Load images
        images = ImageLoader.load_and_prepare_images(image_folder, tile_width, tile_height)
        if not images:
            return None
        
        # Create preview frame
        frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        
        # Use smart image selection for preview too
        temp_selector = ImageSelector(images)
        
        for row in range(rows):
            for col in range(images_per_row):
                if images:
                    img = temp_selector.get_next_image()
                    x = padding + col * (tile_width + padding)
                    y = padding + row * (tile_height + padding)
                    frame[y:y+tile_height, x:x+tile_width] = img
        
        return Image.fromarray(frame)
    
    def generate_video(
        self,
        image_folder: str,
        video_width: int,
        video_height: int,
        images_per_row: int,
        rows: int,
        padding: int,
        scroll_speed: int,
        scroll_direction: str,
        fade_probability: float,
        fade_randomness: float,
        fade_duration: float,
        fade_duration_variance: float,
        black_duration: float,
        black_duration_variance: float,
        duration: int,
        fps: int,
        quality: str,
        blend_weight: float = 0.5,
        fade_black_weight: float = 0.35,
        flash_weight: float = 0.15,
        fade_in_enabled: bool = True,
        fade_out_enabled: bool = True,
        fade_in_duration: float = 2.0,
        fade_out_duration: float = 2.0,
        easing_curve: str = "ease_in_out",
        dynamic_image_count: bool = True,
        dynamic_fade_in_duration: float = 3.0,
        dynamic_fade_out_duration: float = 3.0,
        appearance_randomness: float = 0.8,
        transition_smoothness: float = 0.6
    ) -> Tuple[Optional[str], str]:
        """Generate the scrolling wall video."""
        try:
            # Validate parameters
            error = self._validate_parameters(image_folder, video_width, video_height, images_per_row, rows, padding)
            if error:
                self.logger.error(f"Parameter validation failed: {error}")
                return None, f"❌ {error}"
            
            
            # Calculate tile dimensions
            tile_width = (video_width - (images_per_row + 1) * padding) // images_per_row
            tile_height = (video_height - (rows + 1) * padding) // rows
            
            # Load images
            self.logger.info(f"Loading images from: {image_folder}")
            images = ImageLoader.load_and_prepare_images(image_folder, tile_width, tile_height)
            if not images:
                self.logger.error(f"No images could be loaded from: {image_folder}")
                return None, f"❌ No images found in folder: {image_folder}"
            
            self.logger.info(f"Loaded {len(images)} images, tile size: {tile_width}x{tile_height}")
            
            
            # Initialize grid dimensions
            if scroll_direction == "Horizontal":
                grid_cols = images_per_row * GRID_MULTIPLIER
                grid_rows = rows
            else:  # Vertical
                grid_cols = images_per_row
                grid_rows = rows * GRID_MULTIPLIER
            
            # Initialize grid state
            grid_state = self._initialize_grid_state(images, grid_cols, grid_rows, fade_randomness)
            
            def make_frame(t: float) -> np.ndarray:
                """Generate a single frame at time t."""
                scroll_offset_pixels = scroll_speed * t
                
                # Calculate cinematic fade progress
                cinematic_alpha = self.cinematic.get_cinematic_progress(
                    t, duration, fade_in_enabled, fade_out_enabled,
                    fade_in_duration, fade_out_duration, easing_curve
                )
                
                # Calculate dynamic visible image count (separate from cinematic fade)
                total_grid_cells = grid_rows * grid_cols
                visible_image_count = self.cinematic.get_dynamic_visible_image_count(
                    t, duration, dynamic_image_count, 
                    dynamic_fade_in_duration, dynamic_fade_out_duration, easing_curve,
                    total_grid_cells, min_images=1
                )
                
                # Get which specific cells should be visible (if dynamic count is enabled)
                visible_cells = None
                if dynamic_image_count:
                    visible_cells = self.cinematic.get_visible_cells(
                        grid_rows, grid_cols, visible_image_count, total_grid_cells, appearance_randomness
                    )
                
                # Update dynamic appearance states for smooth transitions
                self._update_dynamic_appearance_states(
                    grid_state, t, visible_cells, fps, transition_duration=transition_smoothness
                )
                
                # Log cinematic and image stats occasionally
                if int(t * fps) % 300 == 0:  # Every 10 seconds at 30fps
                    visible_count = len(self._get_currently_visible_images(grid_state))
                    self.logger.debug(f"Frame {int(t * fps)}: {visible_count} unique images visible, cinematic: {visible_image_count}/{total_grid_cells}, alpha: {cinematic_alpha:.2f}")
                
                # Update fade states (but modulate based on cinematic progress)
                self._update_grid_fade_states(
                    grid_state, t, fps, images, fade_probability * cinematic_alpha, 
                    fade_duration, fade_duration_variance, black_duration, black_duration_variance,
                    blend_weight, fade_black_weight, flash_weight
                )
                
                # Render frame
                frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
                
                if scroll_direction == "Horizontal":
                    self._render_horizontal_scroll(
                        frame, grid_state, scroll_offset_pixels, video_width, video_height,
                        images_per_row, rows, tile_width, tile_height, padding, grid_cols,
                        cinematic_alpha, fade_in_enabled, fade_out_enabled
                    )
                else:
                    self._render_vertical_scroll(
                        frame, grid_state, scroll_offset_pixels, video_width, video_height,
                        images_per_row, rows, tile_width, tile_height, padding, grid_rows,
                        cinematic_alpha, fade_in_enabled, fade_out_enabled
                    )
                
                # Apply cinematic fade to entire frame
                if cinematic_alpha < 1.0:
                    frame = (frame.astype(np.float32) * cinematic_alpha).astype(np.uint8)
                
                return frame
            
            # Create video clip
            self.logger.info("Creating video clip...")
            clip = VideoClip(make_frame, duration=duration)
            
            
            # Generate output
            with temp_manager.persistent_temp_file('.mp4', 'video_output_') as output_path:
                self.logger.info(f"Writing video to: {os.path.basename(output_path)}")
                
                # Write video file
                clip.write_videofile(
                    output_path,
                    fps=fps,
                    codec='libx264',
                    bitrate=QUALITY_BITRATES[quality],
                    logger=None
                )
                
                # Close clip to free memory
                clip.close()
                
                # Verify file creation
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                    return output_path, f"✅ Video generated! ({len(images)} unique images, {file_size:.1f}MB)"
                else:
                    return None, "❌ Video file was not created or is empty"
                    
        except Exception as e:
            log_error_with_context(self.logger, e, "Video generation failed")
            return None, f"❌ Video generation failed: {str(e)}"
