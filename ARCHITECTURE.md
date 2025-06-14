# Architecture Documentation

## Overview

The Immersive Room application has been refactored into a modular architecture with proper separation of concerns, type safety, and memory management.

## Module Structure

### 1. `config.py`
Contains all configuration constants and default values.

**Key Constants:**
- Audio frequency thresholds (for future audio features)
- Transition timing parameters (fade multipliers, duration multipliers)
- Default UI values (video dimensions, animation settings)
- Quality bitrate mappings

### 2. `audio_processor.py`
Handles all audio processing and frequency analysis (prepared for future audio integration).

**Classes:**
- `AudioProcessor`: Core audio processing functionality
  - Frequency spectrum analysis
  - Audio cropping and looping
  - Temporary file management with automatic cleanup
- `AudioState`: Manages audio processing state and caching
  - Caches processed frequency data
  - Provides video-duration-specific frequency data

**Key Features:**
- Memory-safe temporary file handling
- Type-annotated methods
- Automatic cleanup on exit
- Fallback for missing scipy dependency

### 3. `video_generator.py`
Handles video generation with scrolling effects and music synchronization.

**Classes:**
- `ImageLoader`: Static methods for loading and preparing images
- `GridCell`: Represents individual cells in the scrolling grid
- `MusicSyncController`: Manages music synchronization logic
- `VideoGenerator`: Main video generation orchestrator

**Key Features:**
- Proper memory management for video clips
- Music-synced fade effects and scroll speed
- Smooth infinite scrolling in both directions
- Comprehensive input validation

### 4. `scrolling_wall_gradio.py`
Main Gradio interface using the modular components.

**Features:**
- Clean separation between UI and business logic
- Uses configuration constants for default values
- Proper state management with class instances

## Key Improvements

### 1. Memory Management
- **Before**: Global state, memory leaks, no cleanup
- **After**: Class-based state, automatic temp file cleanup, proper resource management

### 2. Code Organization
- **Before**: Single 800+ line file with mixed concerns
- **After**: Modular architecture with single responsibility principle

### 3. Type Safety
- **Before**: No type hints
- **After**: Comprehensive type annotations throughout

### 4. Configuration Management
- **Before**: Magic numbers scattered throughout code
- **After**: Centralized configuration with meaningful constants

### 5. Error Handling
- **Before**: Basic try-catch blocks
- **After**: Comprehensive validation and graceful error handling

## Usage

### Running the Application
```bash
# Activate virtual environment (recommended)
source venv/bin/activate

# Run the application
python scrolling_wall_gradio.py
```

### Extending the Application
1. **Adding new audio effects**: Extend `MusicSyncController`
2. **New video effects**: Add methods to `VideoGenerator`
3. **Configuration changes**: Update `config.py`
4. **UI modifications**: Edit `scrolling_wall_gradio.py`

## Dependencies
- Core: gradio, numpy, Pillow, moviepy
- Audio: librosa, soundfile (optional, graceful fallback)
- Video: imageio, imageio-ffmpeg

## Technical Notes

### Virtual Environment
The application now uses a virtual environment to avoid system package conflicts. All dependencies are properly isolated.

### Backward Compatibility
The refactored application maintains full backward compatibility with the original functionality while providing improved performance and maintainability.

### Future Enhancements
The modular architecture makes it easy to add:
- New synchronization modes
- Additional video effects
- Different export formats
- Real-time preview capabilities