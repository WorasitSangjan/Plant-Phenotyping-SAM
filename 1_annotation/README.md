# Step 1: Pot Center Annotation

This folder contains tools for annotating pot centers, which serve as biological identity anchors for plant segmentation.

---

## 📋 Overview

**Purpose:** Mark the center of each pot to establish plant identities

**Key Insight:** Pot positions are fixed in controlled imaging setups. Annotate once, use for all images from that setup.

**Time Required:** 
- Manual annotation: 5 minutes (one-time)
- Copy to all images: 10 seconds
- Auto-detection: 30 seconds + verification

---

## 🎯 Why This Step is Critical

Pot centers serve as:
- **Biological identity anchors** - Each pot = one plant
- **SAM prompt locations** - Positive prompts for segmentation
- **Reference points** - For spatial measurements and spillover calculations

Without accurate pot centers, the entire pipeline fails.

---

## 📁 Files in This Folder

### Core Scripts

**`annotate_pots.py`** - Interactive annotation tool
- Mark pot centers with mouse clicks
- Navigate between images
- Save/load annotations

**`copy_pot_centers_to_all.py`** - Copy pot centers across images
- Use when imaging setup is fixed
- Saves 40+ hours for large datasets

**`auto_detect_pots.py`** - Automatic pot detection
- Circle detection (Hough transform)
- Grid template generation
- Requires manual verification

---

## 🚀 Usage

### Method 1: Manual Annotation (Recommended for First Time)

```bash
# Annotate pot centers in one reference image
python annotate_pots.py
```

**Interactive Controls:**
- **Left click**: Add pot center
- **Right click**: Remove nearest pot center
- **`n`**: Next image
- **`p`**: Previous image
- **`s`**: Save annotations
- **`q`**: Save and quit

**Tips:**
- Be consistent (always mark center of soil surface)
- Start with a clear image (mid-growth stage)
- For very small seedlings, place slightly toward visible stem
- Mark pots in systematic order (left to right, top to bottom)

---

### Method 2: Copy to All Images (For Scalability)

```bash
# After annotating one image, copy to all others
python copy_pot_centers_to_all.py --reference rgb_061325.jpg
```

**When to use:**
- ✅ Camera and tray positions are fixed
- ✅ All images from same experimental setup
- ✅ Processing 50+ images

**Arguments:**
- `--reference` or `-r`: Name of reference image (required)
- `--no-verify`: Skip confirmation prompt

**Example:**
```bash
# Copy pot centers from first image to all others
python copy_pot_centers_to_all.py --reference rgb_061325.jpg

# Without confirmation prompt (for scripts)
python copy_pot_centers_to_all.py --reference rgb_061325.jpg --no-verify
```

---

### Method 3: Auto-Detection (Optional)

```bash
# Auto-detect pots using circle detection
python auto_detect_pots.py --image rgb_061325.jpg --method hough

# Or use grid template
python auto_detect_pots.py --image rgb_061325.jpg --method grid --rows 4 --cols 8
```

**Methods:**
- `hough`: Detects circular pots using Hough Circle Transform
- `grid`: Generates regular grid pattern

**Arguments:**
- `--image` or `-i`: Reference image name (required)
- `--method` or `-m`: Detection method (default: hough)
- `--rows`: Expected number of rows (for grid method)
- `--cols`: Expected number of columns (for grid method)
- `--no-visualize`: Skip visualization output

**Important:** Always verify auto-detection results with `annotate_pots.py`!

---

## 📤 Output

### pot_centers.json
```json
{
  "rgb_061325.jpg": [
    [245, 180],
    [345, 180],
    [445, 180],
    ...
  ],
  "rgb_061925.jpg": [
    [245, 180],
    [345, 180],
    ...
  ]
}
```

**Format:**
- Dictionary mapping image names to pot centers
- Each pot center is `[x, y]` in pixel coordinates
- Origin is top-left corner

**Location:** `/mnt/user-data/outputs/pot_centers.json`

---

## 🔧 Configuration

Edit `../config.py` to customize:

```python
# Expected grid dimensions (for validation)
EXPECTED_ROWS = 4
EXPECTED_COLS = 8

# Image patterns
RGB_PATTERN = "rgb_*.jpg"
NIR_PATTERN = "nir_*.jpg"

# Data directories
DATA_DIR = Path("/mnt/user-data/uploads")
OUTPUT_DIR = Path("/mnt/user-data/outputs")
```

---

## ✅ Verification

### Visual Check
```bash
# Open annotation tool and navigate through images
python annotate_pots.py

# Check random samples:
# - Early dates
# - Middle dates
# - Late dates

# Verify:
# ✓ All pots are marked
# ✓ Centers are accurate (±5 pixels)
# ✓ No missing or extra pots
# ✓ Numbering is consistent
```

### Quantitative Check
```python
import json

# Load pot centers
with open('/mnt/user-data/outputs/pot_centers.json', 'r') as f:
    pot_centers = json.load(f)

# Check consistency
first_image = list(pot_centers.keys())[0]
reference_count = len(pot_centers[first_image])

for image_name, centers in pot_centers.items():
    if len(centers) != reference_count:
        print(f"⚠️ {image_name}: {len(centers)} pots (expected {reference_count})")

print(f"✓ Verified {len(pot_centers)} images")
```

---

## 🐛 Troubleshooting

### Issue: "No images found"
**Cause:** Images not in expected directory

**Solution:**
```bash
# Check image location
ls /mnt/user-data/uploads/*.jpg

# Update config.py if needed
DATA_DIR = Path("/your/custom/path")
```

---

### Issue: "Pot centers are off by 5-10 pixels in some images"
**Cause:** Slight camera movement between sessions

**Solution:**
- **Option 1:** Accept it (±10 pixels is usually fine for SAM)
- **Option 2:** Annotate multiple reference images (one per session)
- **Option 3:** Use image registration to align images first

---

### Issue: "Auto-detection found 0 pots"
**Cause:** Pot edges not clear, or parameters wrong

**Solution:**
```bash
# Try grid method
python auto_detect_pots.py --image rgb_061325.jpg --method grid --rows 4 --cols 8

# Or adjust Hough parameters in auto_detect_pots.py:
min_radius = 40   # Try smaller
max_radius = 150  # Try larger
min_dist = 100    # Adjust spacing
```

---

### Issue: "Too many false positives in auto-detection"
**Cause:** Other circular objects detected

**Solution:**
```bash
# Increase minimum distance between pots
# Edit auto_detect_pots.py:
min_dist = 150  # Pots must be 150+ pixels apart

# Then manually remove false positives:
python annotate_pots.py  # Right-click to remove
```

---

## 📊 Workflow Examples

### Example 1: Small Dataset (12 images)
```bash
# Annotate first image
python annotate_pots.py

# Copy to all others
python copy_pot_centers_to_all.py --reference rgb_061325.jpg

# Verify random samples
python annotate_pots.py  # Check 3-4 different dates

# Proceed to segmentation
cd ../2_segment
python segment_plants.py
```

---

### Example 2: Large Dataset (500 images)
```bash
# Try auto-detection first
python auto_detect_pots.py --image rgb_001.jpg

# Verify and correct
python annotate_pots.py

# Copy to all 499 others
python copy_pot_centers_to_all.py --reference rgb_001.jpg

# Spot-check 5 random images
python annotate_pots.py

# Proceed to segmentation
cd ../2_segment
python segment_plants.py
```

---

### Example 3: Multiple Experimental Setups
```bash
# Setup A (images 1-200)
python annotate_pots.py  # Annotate image 1
# ... manually annotate only images from setup A ...
python copy_pot_centers_to_all.py --reference rgb_001.jpg

# Setup B (images 201-400)
python annotate_pots.py  # Annotate image 201
# ... manually annotate only images from setup B ...
python copy_pot_centers_to_all.py --reference rgb_201.jpg
```

---

## 💡 Best Practices

### Annotation Quality
- ✅ Use a clear, mid-growth stage image as reference
- ✅ Zoom in if needed for precision
- ✅ Mark pots in systematic order
- ✅ Double-check before moving to next image
- ✅ Save frequently (`s` key)

### Verification
- ✅ Always verify random samples after copying
- ✅ Check early, middle, and late dates
- ✅ Compare pot counts across images
- ✅ Visual inspection of pot_detection_*.jpg files

### Scalability
- ✅ For 50+ images, use copy_pot_centers_to_all.py
- ✅ For 100+ images, try auto-detection first
- ✅ For multiple setups, annotate one per setup
- ✅ Save pot_centers.json to version control

---

## 📈 Time Estimates

| Dataset Size | Manual (All) | Copy Method | Auto + Verify |
|-------------|-------------|-------------|---------------|
| 12 images   | 60 min      | 5 min       | 10 min        |
| 50 images   | 4.2 hours   | 5 min       | 12 min        |
| 500 images  | 42 hours    | 5 min       | 20 min        |
| 1000 images | 83 hours    | 5 min       | 30 min        |

---

## ⏭️ Next Step

Once pot centers are annotated and verified:

```bash
cd ../2_segment
python segment_plants.py
```

See [../2_segment/README.md](../2_segment/README.md) for segmentation instructions.

---

## 🔗 Related Documentation

- [Main README](../README.md) - Pipeline overview
- [Quick Start Guide](../docs/QUICKSTART.md) - Get started quickly
- [Scalable Workflow](../docs/SCALABLE_WORKFLOW.md) - Processing large datasets
