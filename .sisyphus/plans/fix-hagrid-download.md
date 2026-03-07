# Fix: Download HAGRID Images from Correct Source

## Problem Identified

The current notebook Cell 3 downloads from Kaggle (`kapitanov/hagrid`), but that dataset only contains **annotation JSON files** (metadata), not the actual images.

**Current state:**
- ✅ `data/ann_train_val/` - Has annotations (like.json, etc.)
- ❌ `data/hagrid_raw/` - **EMPTY** (no images)
- ❌ `data/processed/` - **EMPTY**

## Root Cause

The HAGRID dataset has two components:
1. **Annotations** - On Kaggle as JSON files
2. **Images** - On GitHub Releases as `.zip` files

The notebook only downloads #1, missing #2 entirely.

## Solution

Replace Cell 3 with code that downloads images from GitHub:

```python
# Cell 3: Download HAGRID Images from GitHub
import zipfile
from pathlib import Path
import urllib.request

print("📥 Task 3: Downloading HAGRID images from GitHub...")
print("Note: Kaggle only has annotations. Images are on GitHub.")

# Create directories
hagrid_dir = HAGRID_RAW
hagrid_dir.mkdir(parents=True, exist_ok=True)

# Download specific gesture classes
# Available: call, dislike, fist, four, like, mute, ok, one, palm, peace, etc.
gesture_classes = ['like']  # Thumbs up class

base_url = "https://github.com/hukenovs/hagrid/releases/download/v1.0/hagrid_30k_"

for gesture in gesture_classes:
    zip_path = DATA_DIR / f'hagrid_{gesture}.zip'
    gesture_dir = hagrid_dir / gesture
    
    # Skip if already extracted
    if gesture_dir.exists() and any(gesture_dir.iterdir()):
        print(f"✅ {gesture}/ already exists, skipping")
        continue
    
    # Download if not exists
    if not zip_path.exists():
        url = f"{base_url}{gesture}.zip"
        print(f"\n⬇️  Downloading {gesture}.zip...")
        print(f"   URL: {url}")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"✅ Downloaded {gesture}.zip")
        except Exception as e:
            print(f"❌ Failed to download {gesture}: {e}")
            continue
    else:
        print(f"✅ {gesture}.zip already exists")
    
    # Extract
    print(f"\n📦 Extracting {gesture}.zip...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(hagrid_dir)
        print(f"✅ Extracted {gesture}/")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        continue
    
    # Clean up zip
    zip_path.unlink()
    print(f"✅ Cleaned up {gesture}.zip")

# Check what we have
print("\n📊 Download Summary:")
if hagrid_dir.exists():
    for item in hagrid_dir.iterdir():
        if item.is_dir():
            image_count = len(list(item.glob('*.jpg')))
            print(f"  {item.name}/: {image_count} images")
else:
    print("  ❌ No data downloaded")

print("\n🎉 Task 3 Complete!")
```

## Expected Result

After running the fixed Cell 3:
```
📥 Task 3: Downloading HAGRID images from GitHub...

⬇️  Downloading like.zip...
   URL: https://github.com/hukenovs/hagrid/releases/download/v1.0/hagrid_30k_like.zip
✅ Downloaded like.zip

📦 Extracting like.zip...
✅ Extracted like/
✅ Cleaned up like.zip

📊 Download Summary:
  like/: ~16,000 images

🎉 Task 3 Complete!
```

## Alternative: Manual Download

If the download fails, manually download from:
- https://github.com/hukenovs/hagrid/releases/tag/v1.0
- Download `hagrid_30k_like.zip`
- Extract to `data/hagrid_raw/like/`

## References

- GitHub Repo: https://github.com/hukenovs/hagrid
- Releases: https://github.com/hukenovs/hagrid/releases
- Original Paper: https://arxiv.org/abs/2206.08219
