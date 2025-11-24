"""Improved temporary file management for the scrolling wall generator."""

import os
import tempfile
import atexit
from typing import List, Optional
from contextlib import contextmanager
from logger import get_logger

logger = get_logger(__name__)


class TempFileManager:
    """Manages temporary files with automatic cleanup."""
    
    def __init__(self):
        self._temp_files: List[str] = []
        atexit.register(self.cleanup_all)
    
    @contextmanager
    def temp_file(self, suffix: str = '', prefix: str = 'scrolling_wall_'):
        """
        Context manager for temporary files with immediate cleanup.
        
        Args:
            suffix: File extension (e.g., '.wav', '.mp4')
            prefix: File prefix for identification
            
        Yields:
            Path to temporary file
        """
        with tempfile.NamedTemporaryFile(
            suffix=suffix, 
            prefix=prefix, 
            delete=False
        ) as tmp_file:
            temp_path = tmp_file.name
        
        self._temp_files.append(temp_path)
        try:
            yield temp_path
        finally:
            # Immediate cleanup on context exit
            self._remove_temp_file(temp_path)
    
    @contextmanager 
    def persistent_temp_file(self, suffix: str = '', prefix: str = 'scrolling_wall_'):
        """
        Context manager for temporary files that persist until explicit cleanup.
        Use this when the file needs to exist beyond the immediate context.
        
        Args:
            suffix: File extension (e.g., '.wav', '.mp4')
            prefix: File prefix for identification
            
        Yields:
            Path to temporary file
        """
        with tempfile.NamedTemporaryFile(
            suffix=suffix, 
            prefix=prefix, 
            delete=False
        ) as tmp_file:
            temp_path = tmp_file.name
        
        self._temp_files.append(temp_path)
        try:
            yield temp_path
        finally:
            # Don't clean up immediately - will be cleaned up by cleanup_all()
            pass
    
    def _remove_temp_file(self, temp_path: str) -> None:
        """Remove a single temporary file."""
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.debug(f"Cleaned up temp file: {os.path.basename(temp_path)}")
            if temp_path in self._temp_files:
                self._temp_files.remove(temp_path)
        except Exception as e:
            logger.warning(f"Could not clean up temp file {temp_path}: {e}")
    
    def cleanup_all(self) -> None:
        """Clean up all remaining temporary files."""
        if not self._temp_files:
            return
            
        logger.debug(f"Cleaning up {len(self._temp_files)} temporary files")
        for temp_file in self._temp_files[:]:  # Copy list to avoid modification during iteration
            self._remove_temp_file(temp_file)
    
    def get_temp_count(self) -> int:
        """Get count of tracked temporary files."""
        return len(self._temp_files)


# Global instance for convenience
temp_manager = TempFileManager()