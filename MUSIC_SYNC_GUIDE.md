# 🎵 Music Synchronization Guide

## Overview

The music sync feature allows you to create videos where image transitions are perfectly synchronized to the beat of your music!

## How It Works

1. **Beat Detection**: The app analyzes your audio file using librosa to detect beats and tempo
2. **Fade Sync**: Image transitions trigger on musical beats instead of random timing
3. **Scroll Sync**: Optionally adjust scroll speed to match the music's BPM
4. **Audio Embedding**: Your music is included in the final video

## Using Music Sync

### Step 1: Upload Audio
- Click "Upload music file" in the 🎵 Music Synchronization section
- Supported formats: MP3, WAV, FLAC, M4A, etc.
- The app will analyze the audio for beats and tempo

### Step 2: Enable Sync Features
- **"Sync fades to beats"**: Image transitions happen on musical beats
- **"Sync scroll to tempo"**: Scroll speed adjusts to match BPM (experimental)

### Step 3: Generate Video
- The app will show detected tempo (e.g., "120.5 BPM")
- Beat count will be displayed in the success message
- Final video includes both visuals and audio

## Tips for Best Results

### Music Selection
- **Clear beats**: Electronic, hip-hop, rock work great
- **Consistent tempo**: Avoid songs with tempo changes
- **Medium tempo**: 80-140 BPM works best for visuals
- **Short clips**: Start with 10-30 second clips

### Settings
- **Lower fade frequency**: Let music drive the transitions
- **Slower scroll**: Allows beats to be more noticeable  
- **More images**: Creates variety in beat-synced transitions

### Recommended Workflow
1. Start with "Sync fades to beats" only
2. Test with a short clip (5-10 seconds)
3. Try "Sync scroll to tempo" if you like the result
4. Generate longer videos once you're happy

## Troubleshooting

**No beats detected**: 
- Try a different audio file with clearer rhythm
- Ensure audio file isn't corrupted

**Weird scroll speed**:
- Disable "Sync scroll to tempo" - this is experimental
- Very fast/slow BPM can cause extreme scroll speeds

**Audio doesn't match video length**:
- Audio will be trimmed to match video duration
- For best sync, make audio and video the same length

## Examples

### Electronic Music (128 BPM)
- Sync fades: ✅ ON
- Sync scroll: ✅ ON  
- Fade frequency: 0.001 (low)
- Scroll speed: 25 px/sec

### Hip-Hop (90 BPM)
- Sync fades: ✅ ON
- Sync scroll: ❌ OFF
- Fade frequency: 0.002
- Scroll speed: 15 px/sec

### Rock (140 BPM)
- Sync fades: ✅ ON
- Sync scroll: ✅ ON
- Fade frequency: 0.001
- Scroll speed: 30 px/sec

## Advanced Tips

- **Offset timing**: The sync isn't perfect - music production often has slight delays
- **Multiple attempts**: Try generating the same video multiple times for best sync
- **Tempo doubling**: If beats feel too fast, the BPM might be detected as double - this is normal
- **Visual feedback**: Watch for fade transitions happening on obvious beats like kick drums

Enjoy creating music videos with your artwork! 🎨🎵