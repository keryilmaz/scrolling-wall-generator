"""Configuration constants for the Immersive Room application."""

# Audio frequency band thresholds (Hz)
LOW_FREQ_THRESHOLD = 250     # Bass/sub-bass: 0-250 Hz
HIGH_FREQ_THRESHOLD = 4000   # Highs: 4000+ Hz
# Mid frequencies are between LOW_FREQ_THRESHOLD and HIGH_FREQ_THRESHOLD

# Audio sync thresholds (for future audio integration)
HIGH_FREQ_STRONG_THRESHOLD = 70  # Strong high frequency moments
BASS_DROP_THRESHOLD = 80         # Heavy bass moments

# Fade effect multipliers
MIN_FADE_MULTIPLIER = 0.1
MAX_FADE_MULTIPLIER = 2.1
BASE_FADE_PROBABILITY_BOOST = 0.01

# Duration multipliers for audio sync
MIN_DURATION_MULTIPLIER = 0.5
MAX_DURATION_MULTIPLIER = 2.0
MIN_BLACK_DURATION_MULTIPLIER = 0.8
MAX_BLACK_DURATION_MULTIPLIER = 1.2

# Fade style settings for dramatic transitions
QUICK_BLINK_FADE_DURATION = 0.15    # Very fast fade for dramatic flashes
QUICK_BLINK_BLACK_DURATION = 0.1    # Brief black moment
LONG_FADE_DURATION_MULTIPLIER = 1.8  # Longer, cinematic fades
LONG_FADE_BLACK_MULTIPLIER = 1.5     # Extended black duration

# Thresholds for fade style selection
HIGH_FREQ_BLINK_THRESHOLD = 75      # High freq above this triggers quick blinks
BASS_DROP_LONG_FADE_THRESHOLD = 85  # Heavy bass triggers long fades
MID_FREQ_VARIATION_THRESHOLD = 15   # Mid freq changes trigger style mixing

# Scroll speed multipliers for audio sync
MIN_SCROLL_MULTIPLIER = 0.5
MAX_SCROLL_MULTIPLIER = 1.5
MIN_SCROLL_SPEED = 5  # Minimum scroll speed in px/sec

# Smoothing parameters
SMOOTHING_WINDOW_DIVISOR = 10  # Window size = fps // 10 (~0.1 seconds)

# Video quality settings
QUALITY_BITRATES = {
    "Low": "5M",
    "Medium": "15M", 
    "High": "30M",
    "Ultra": "50M"
}

# Default values
DEFAULT_VIDEO_WIDTH = 1920
DEFAULT_VIDEO_HEIGHT = 1080
DEFAULT_IMAGES_PER_ROW = 6
DEFAULT_ROWS = 3
DEFAULT_PADDING = 10
DEFAULT_SCROLL_SPEED = 25
DEFAULT_FADE_PROBABILITY = 0.002
DEFAULT_FADE_RANDOMNESS = 0.5
DEFAULT_FADE_DURATION = 1.0
DEFAULT_BLACK_DURATION = 0.5
DEFAULT_DURATION = 10
DEFAULT_FPS = 30
DEFAULT_ANALYSIS_FPS = 30

# Grid multipliers for infinite scrolling
GRID_MULTIPLIER = 3  # 3x width/height for seamless scrolling
TILES_BUFFER = 2     # Extra tiles to render for smooth edges

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

# Server settings
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8085

# Audio processing settings
SKIP_AUDIO_PROCESSING = False  # Set to True to skip audio entirely for faster processing