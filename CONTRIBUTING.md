# Contributing to Scrolling Wall Generator

## Development Workflow

This project uses Git branches to manage features and versions. The `main` branch contains the stable version.

### Working on New Features

1. **Create a new branch for your feature:**
```bash
git checkout -b feature/your-feature-name
```

Common branch naming conventions:
- `feature/add-music-support`
- `feature/custom-transitions`
- `feature/text-overlay`
- `fix/memory-leak`
- `docs/update-readme`

2. **Make your changes and commit:**
```bash
git add .
git commit -m "Add: your feature description"
```

3. **Push your branch:**
```bash
git push origin feature/your-feature-name
```

4. **Create a Pull Request on GitHub**

### Example: Adding a New Feature

```bash
# Switch to main and pull latest changes
git checkout main
git pull

# Create new feature branch
git checkout -b feature/music-sync

# Work on your feature...
# Then commit and push
git add .
git commit -m "Add: Music synchronization support"
git push origin feature/music-sync
```

### Switching Between Versions

```bash
# See all branches
git branch -a

# Switch to a branch
git checkout branch-name

# Switch back to main
git checkout main
```

### Keeping Your Branch Updated

```bash
# While on your feature branch
git fetch origin
git merge origin/main
```

## Feature Ideas for Future Versions

- **Audio Support**: Add background music or sound effects
- **Text Overlays**: Add customizable text on images
- **Transition Effects**: More transition types (slide, zoom, rotate)
- **Image Filters**: Apply filters to images (blur, color shift, etc.)
- **Pattern Mode**: Create geometric patterns with images
- **3D Effects**: Add perspective/3D transformations
- **Multi-layer**: Support multiple scrolling layers
- **Interactive Mode**: Mouse/keyboard controlled scrolling
- **Live Preview**: Real-time preview while adjusting settings
- **Batch Processing**: Generate multiple videos with different settings

## Code Style

- Use clear variable names
- Add docstrings to functions
- Keep functions focused and small
- Test your changes thoroughly
- Update README if adding new features