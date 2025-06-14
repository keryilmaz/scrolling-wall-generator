"""Audio processing module for frequency analysis and music synchronization."""

import os
import tempfile
import numpy as np
from typing import Optional, Tuple, Dict, Any
from contextlib import contextmanager
import atexit

from config import (
    LOW_FREQ_THRESHOLD, HIGH_FREQ_THRESHOLD, SMOOTHING_WINDOW_DIVISOR,
    DEFAULT_ANALYSIS_FPS
)
from logger import get_logger, log_error_with_context
from temp_file_manager import temp_manager

# Audio processing dependencies
try:
    import librosa
    import soundfile as sf
    AUDIO_SUPPORT = True
except ImportError:
    AUDIO_SUPPORT = False
    logger = get_logger(__name__)
    logger.warning("librosa not installed. Music sync features disabled.")


class AudioProcessor:
    """Handles audio file processing and frequency analysis."""
    
    def __init__(self):
        self.audio_support = AUDIO_SUPPORT
        self.logger = get_logger(__name__)
    
    def _normalize_frequency_data(self, data: np.ndarray, fps: int) -> np.ndarray:
        """Normalize frequency data to 0-100 scale with smoothing."""
        if np.max(data) > 0:
            normalized = (data / np.max(data)) * 100
            # Apply smoothing to avoid rapid changes
            window_size = max(1, fps // SMOOTHING_WINDOW_DIVISOR)
            if len(normalized) > window_size:
                normalized = np.convolve(
                    normalized, np.ones(window_size) / window_size, mode='same'
                )
            return normalized.astype(np.float32)
        return np.zeros_like(data, dtype=np.float32)
    
    def _resize_frequency_data(self, data: np.ndarray, target_frames: int) -> np.ndarray:
        """Resize frequency data to match target number of frames."""
        if len(data) == 0:
            return np.zeros(target_frames)
        
        try:
            from scipy.interpolate import interp1d
            if len(data) == 1:
                return np.full(target_frames, data[0])
            
            x_old = np.linspace(0, 1, len(data))
            x_new = np.linspace(0, 1, target_frames)
            f = interp1d(x_old, data, kind='linear', fill_value='extrapolate')
            return f(x_new)
        except ImportError:
            # Fallback to simple interpolation
            return np.interp(
                np.linspace(0, len(data) - 1, target_frames),
                np.arange(len(data)),
                data
            )
    
    def analyze_frequency_spectrum(
        self, 
        audio_file: str, 
        fps: int = DEFAULT_ANALYSIS_FPS, 
        duration: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
        """
        Analyze audio file for frequency spectrum.
        
        Args:
            audio_file: Path to audio file
            fps: Analysis frame rate
            duration: Duration to analyze (None for full file)
            
        Returns:
            Tuple of (low_freq_map, mid_freq_map, high_freq_map, sample_rate)
        """
        if not self.audio_support or not audio_file:
            return None, None, None, None
            
        if not os.path.exists(audio_file):
            self.logger.error(f"Audio file not found: {audio_file}")
            return None, None, None, None
        
        try:
            self.logger.info(f"Analyzing audio spectrum: {os.path.basename(audio_file)}")
            
            # Load audio file
            y, sr = librosa.load(audio_file, duration=duration)
            
            # Compute spectrogram using Short-Time Fourier Transform
            hop_length = int(sr / fps)
            stft = librosa.stft(y, hop_length=hop_length)
            magnitude = np.abs(stft)
            
            # Define frequency bands
            frequencies = librosa.fft_frequencies(sr=sr)
            low_freq_mask = frequencies <= LOW_FREQ_THRESHOLD
            mid_freq_mask = (frequencies > LOW_FREQ_THRESHOLD) & (frequencies <= HIGH_FREQ_THRESHOLD)
            high_freq_mask = frequencies > HIGH_FREQ_THRESHOLD
            
            # Calculate energy in each frequency band over time
            low_energy = np.mean(magnitude[low_freq_mask, :], axis=0)
            mid_energy = np.mean(magnitude[mid_freq_mask, :], axis=0)
            high_energy = np.mean(magnitude[high_freq_mask, :], axis=0)
            
            # Normalize to 0-100 scale
            low_map = self._normalize_frequency_data(low_energy, fps)
            mid_map = self._normalize_frequency_data(mid_energy, fps)
            high_map = self._normalize_frequency_data(high_energy, fps)
            
            # Resize to target frames if duration specified
            if duration:
                target_frames = int(duration * fps)
                low_map = self._resize_frequency_data(low_map, target_frames)
                mid_map = self._resize_frequency_data(mid_map, target_frames)
                high_map = self._resize_frequency_data(high_map, target_frames)
            
            self.logger.info(f"Spectral analysis complete:")
            self.logger.info(f"  - Low freq (0-{LOW_FREQ_THRESHOLD}Hz): avg={np.mean(low_map):.1f}, max={np.max(low_map):.1f}")
            self.logger.info(f"  - Mid freq ({LOW_FREQ_THRESHOLD}-{HIGH_FREQ_THRESHOLD}Hz): avg={np.mean(mid_map):.1f}, max={np.max(mid_map):.1f}")
            self.logger.info(f"  - High freq ({HIGH_FREQ_THRESHOLD}Hz+): avg={np.mean(high_map):.1f}, max={np.max(high_map):.1f}")
            self.logger.info(f"  - Total frames: {len(low_map)}")
            
            return low_map, mid_map, high_map, sr
            
        except Exception as e:
            log_error_with_context(self.logger, e, f"Audio spectrum analysis failed for {os.path.basename(audio_file)}")
            return None, None, None, None
    
    def crop_audio_for_video(self, audio_file: str, duration: float) -> Optional[str]:
        """
        Crop audio file to match video duration.
        
        Args:
            audio_file: Path to original audio file
            duration: Target duration in seconds
            
        Returns:
            Path to cropped audio file or None if failed
        """
        if not self.audio_support or not audio_file or not os.path.exists(audio_file):
            return None
        
        try:
            self.logger.info(f"Cropping audio to {duration} seconds for video...")
            
            # Load original audio
            y, sr = librosa.load(audio_file)
            
            # Crop to video duration
            target_samples = int(duration * sr)
            if len(y) > target_samples:
                y_cropped = y[:target_samples]
                self.logger.info(f"Audio trimmed from {len(y)/sr:.1f}s to {duration}s")
            elif len(y) < target_samples:
                # Loop the audio to match video duration
                repeats = int(np.ceil(target_samples / len(y)))
                y_repeated = np.tile(y, repeats)
                y_cropped = y_repeated[:target_samples]
                self.logger.info(f"Audio looped from {len(y)/sr:.1f}s to {duration}s")
            else:
                y_cropped = y
                self.logger.info(f"Audio duration matches video ({duration}s)")
            
            # Save cropped audio to temporary file
            with temp_manager.persistent_temp_file('.wav', 'audio_crop_') as cropped_path:
                sf.write(cropped_path, y_cropped, sr)
                self.logger.info(f"Cropped audio saved: {os.path.basename(cropped_path)}")
                return cropped_path
                
        except Exception as e:
            log_error_with_context(self.logger, e, f"Audio cropping failed for {os.path.basename(audio_file)}")
            return None


class AudioState:
    """Manages audio processing state and caching."""
    
    def __init__(self):
        self.original_file: Optional[str] = None
        self.frequency_maps: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.sample_rate: Optional[int] = None
        self.original_duration: Optional[float] = None
        self.analysis_fps: int = DEFAULT_ANALYSIS_FPS
        self.processor = AudioProcessor()
        self.logger = get_logger(__name__)
    
    def process_audio_file(self, audio_file: str, analysis_fps: int = DEFAULT_ANALYSIS_FPS) -> str:
        """
        Process audio file and cache the results.
        
        Args:
            audio_file: Path to audio file
            analysis_fps: Analysis frame rate
            
        Returns:
            Status message
        """
        if not self.processor.audio_support:
            return "❌ Audio processing not available (librosa not installed)"
        
        if not audio_file or not os.path.exists(audio_file):
            return "❌ No audio file provided or file not found"
        
        try:
            self.logger.info(f"Processing audio: {os.path.basename(audio_file)}")
            
            # Load the full audio file to get its duration
            y_full, sr = librosa.load(audio_file)
            full_duration = len(y_full) / sr
            
            self.logger.info(f"Audio loaded: {full_duration:.1f} seconds, {sr} Hz")
            
            # Analyze the entire audio file
            low_map, mid_map, high_map, _ = self.processor.analyze_frequency_spectrum(
                audio_file, analysis_fps
            )
            
            if low_map is None:
                return "❌ Audio analysis failed"
            
            # Cache the processed data
            self.original_file = audio_file
            self.frequency_maps = (low_map, mid_map, high_map)
            self.sample_rate = sr
            self.original_duration = full_duration
            self.analysis_fps = analysis_fps
            
            # Show analysis results
            total_frames = len(low_map)
            timeline_duration = total_frames / analysis_fps
            
            self.logger.info(f"Audio processing complete!")
            self.logger.info(f"  Frequency analysis:")
            self.logger.info(f"    - {total_frames} frames analyzed ({timeline_duration:.1f}s timeline)")
            self.logger.info(f"    - Low freq: avg={np.mean(low_map):.1f}, max={np.max(low_map):.1f}")
            self.logger.info(f"    - Mid freq: avg={np.mean(mid_map):.1f}, max={np.max(mid_map):.1f}")
            self.logger.info(f"    - High freq: avg={np.mean(high_map):.1f}, max={np.max(high_map):.1f}")
            
            # Statistical insights
            from config import HIGH_FREQ_STRONG_THRESHOLD, BASS_DROP_THRESHOLD
            high_peaks = np.sum(high_map > HIGH_FREQ_STRONG_THRESHOLD)
            bass_drops = np.sum(low_map > BASS_DROP_THRESHOLD)
            mid_variance = np.std(mid_map)
            
            self.logger.info(f"  Musical characteristics:")
            self.logger.info(f"    - High frequency peaks: {high_peaks} moments")
            self.logger.info(f"    - Heavy bass moments: {bass_drops} drops")
            self.logger.info(f"    - Mid-range dynamics: {mid_variance:.1f} variation")
            
            return (f"✅ Audio processed successfully!\n"
                    f"📄 File: {os.path.basename(audio_file)}\n"
                    f"⏱️ Duration: {full_duration:.1f} seconds\n"
                    f"📊 Analysis: {total_frames} frequency frames\n"
                    f"🎵 High peaks: {high_peaks}, Bass drops: {bass_drops}\n"
                    f"💫 Ready for video generation with music sync!")
                    
        except Exception as e:
            log_error_with_context(self.logger, e, f"Audio processing failed for {os.path.basename(audio_file)}")
            return f"Audio processing failed: {str(e)}"
    
    def get_frequency_data_for_video(self, duration: float, fps: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get frequency data cropped/looped to match video duration.
        
        Args:
            duration: Video duration in seconds
            fps: Video frame rate
            
        Returns:
            Tuple of (low_freq_map, mid_freq_map, high_freq_map)
        """
        if not self.frequency_maps:
            return None, None, None
        
        low_map_full, mid_map_full, high_map_full = self.frequency_maps
        total_frames_needed = int(duration * fps)
        frames_available = len(low_map_full)
        
        if frames_available >= total_frames_needed:
            # Crop to video duration
            low_freq_map = low_map_full[:total_frames_needed]
            mid_freq_map = mid_map_full[:total_frames_needed]
            high_freq_map = high_map_full[:total_frames_needed]
            self.logger.info(f"Frequency data cropped to {duration}s ({total_frames_needed} frames)")
        else:
            # Loop frequency data to match video duration
            repeat_factor = int(np.ceil(total_frames_needed / frames_available))
            low_freq_map = np.tile(low_map_full, repeat_factor)[:total_frames_needed]
            mid_freq_map = np.tile(mid_map_full, repeat_factor)[:total_frames_needed]
            high_freq_map = np.tile(high_map_full, repeat_factor)[:total_frames_needed]
            self.logger.info(f"Frequency data looped to {duration}s ({total_frames_needed} frames)")
        
        return low_freq_map, mid_freq_map, high_freq_map
    
    def get_cropped_audio_for_video(self, duration: float) -> Optional[str]:
        """Get audio file cropped to video duration."""
        if not self.original_file:
            return None
        
        return self.processor.crop_audio_for_video(self.original_file, duration)