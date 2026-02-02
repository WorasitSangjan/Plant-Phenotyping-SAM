# Step 2: Plant Segmentation

This folder contains the SAM-based plant segmentation pipeline that uses pot-anchored prompts to correctly segment plants even when leaves overlap across pots.

---

## Overview

**Purpose:** Segment each plant using SAM with pot centers as biological identity anchors

**Key Advantage:** Correctly handles severe leaf overlap by assigning pixels to their biological origin, not geometric proximity

---

## How It Works

### The Segmentation Strategy

```
For each pot center:
  1. Use pot center as positive prompt (foreground)
  2. Use neighboring pot centers as negative prompts (background)
  3. SAM generates plant mask
  4. Mask includes ALL leaves, even if they extend to other pots
  5. Result: Biologically correct segmentation
```

### Why This Solves the Overlap Problem

**Traditional methods:**
- Assign pixels based on geometric proximity
- Result: Overlapping leaves go to wrong plant 

**Our method:**
- Assign pixels based on biological identity (pot anchor)
- Result: Overlapping leaves go to correct plant

---

## Files in This Folder

**`segment_plants.py`** - Main segmentation script
- Loads SAM model
- Processes all images with pot centers
- Handles both RGB and NIR images
- Saves masks and visualizations

---

## Usage

### Basic Usage

```bash
# Segment all plants in all images
python segment_plants.py
```

**Requirements:**
- Pot centers must be annotated (Step 1 complete)
- pot_centers.json must exist in outputs folder

**What it does:**
1. Downloads SAM checkpoint 
2. Loads all images with pot centers
3. Segments each plant using SAM
4. Processes both RGB and NIR images
5. Saves masks as pickle files
6. Creates color visualizations

---

## 📤 Output

### Directory Structure
```
/mnt/user-data/outputs/segmentation_results/
├── masks_2025-06-13/
│   ├── masks.pkl                 # Pickle file with all masks
│   ├── rgb_segmentation.jpg      # Visualization (RGB)
│   └── nir_segmentation.jpg      # Visualization (NIR)
├── masks_2025-06-19/
│   └── ...
└── ... (one folder per date)
```

### masks.pkl Format
```python
{
    'rgb_masks': [mask1, mask2, ...],  # List of binary masks (H, W)
    'nir_masks': [mask1, mask2, ...],  # List of binary masks (H, W)
    'pot_centers': [(x1, y1), ...],    # Pot center coordinates
    'rgb_image': 'path/to/rgb.jpg',    # Original image path
    'nir_image': 'path/to/nir.jpg',    # Original image path
    'date': '2025-06-13'               # Imaging date
}
```

### Visualization Files
- **rgb_segmentation.jpg**: RGB image with colored masks overlay
- **nir_segmentation.jpg**: NIR image with colored masks overlay
- Each plant has a distinct color
- White circles mark pot centers

---

## Configuration

### SAM Model Selection

**Available models:**
- `vit_h` (huge): Best accuracy, slowest (~5 sec/plant)
- `vit_l` (large): Good accuracy, faster (~3 sec/plant)
- `vit_b` (base): Fast, lower accuracy (~1 sec/plant)

**Recommendation:** Use `vit_h` for research (best accuracy)

---

### Memory Requirements

| Model | GPU Memory | CPU Memory | Speed (per plant) |
|-------|-----------|-----------|-------------------|
| vit_h | 8 GB      | 16 GB     | ~5 sec (GPU) / ~10 sec (CPU) |
| vit_l | 6 GB      | 12 GB     | ~3 sec (GPU) / ~7 sec (CPU) |
| vit_b | 4 GB      | 8 GB      | ~1 sec (GPU) / ~3 sec (CPU) |

---

### Negative Prompts

**What they do:**
- Tell SAM "these neighboring pots are background"
- Prevents masks from merging across plants
- Improves segmentation quality in dense plantings

**When to use:**
- High-density plantings (pots close together)
- Large plants with extensive leaf overlap
- Always recommended unless causing issues

**When to disable:**
- If masks are too small (overly conservative)
- If you want to test without them
- Very sparse plantings (pots far apart)

---

## Understanding Visualizations

### RGB Segmentation Visualization
![Example](docs/rgb_segmentation_example.jpg)

**Color coding:**
- Each plant has a unique color
- Overlapping leaves maintain their origin plant's color
- White circles = pot centers
- Different colors = different plants

**What to look for:**
- Each pot has one colored mask
- Overlapping leaves are correctly colored
- No plants are merged (different colors touching)
- Small plants are captured

**Red flags:**
- Two plants share the same color (masks merged)
- A plant has no color (segmentation failed)
- Colors bleed into neighboring pots incorrectly

---

## 🐛 Troubleshooting

### Issue: "SAM checkpoint download failed"
**Cause:** Network issue or insufficient disk space

**Solution:**
```bash
# Download manually
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Move to expected location
mv sam_vit_h_4b8939.pth /home/claude/

# Re-run segmentation
python segment_plants.py
```

---

### Issue: "Out of memory error"
**Cause:** Insufficient GPU/CPU memory for SAM

**Solutions:**
```python
# Option 1: Use smaller model
# Edit segment_plants.py:
SAM_MODEL_TYPE = "vit_b"  # Instead of vit_h

# Option 2: Process images in smaller batches
# Edit segment_plants.py to process fewer images at once

# Option 3: Use CPU instead of GPU (slower but more memory)
# Will auto-detect, or force CPU:
device = "cpu"  # Instead of "cuda"
```

---

### Issue: "Masks are merging neighboring plants"
**Cause:** Negative prompts not working or disabled

**Solutions:**
```python
# Solution 1: Enable negative prompts
use_negative_prompts = True

# Solution 2: Increase neighbor distance
neighbor_distance = 250  # From 200

# Solution 3: Check pot centers are accurate
# Go back to Step 1 and verify pot annotations
cd ../1_annotate
python annotate_pots.py
```

---

### Issue: "Masks are too small / don't capture full plant"
**Cause:** Negative prompts too aggressive or pot centers misplaced

**Solutions:**
```python
# Solution 1: Disable negative prompts temporarily
use_negative_prompts = False

# Solution 2: Adjust pot center placement
# Place closer to stem base, not exact pot center

# Solution 3: Try different SAM settings
multimask_output = True  # Get multiple proposals, pick best
```

---

### Issue: "Very small seedlings not detected"
**Cause:** SAM struggles with very small objects

**Solutions:**
- Adjust pot center placement (move toward visible stem)
- Consider skipping first imaging date if plants too small
- Manually annotate first date if needed

---

### Issue: "Segmentation is too slow"
**Cause:** CPU processing is slow

**Solutions:**
```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# If True but not being used, check CUDA installation
# If False, consider:
# 1. Using GPU-enabled machine
# 2. Using smaller model (vit_b)
# 3. Running overnight
# 4. Processing in parallel on multiple machines
```

---

## Performance Benchmarks

### Hardware Recommendations

**Minimum:**
- CPU: 4 cores
- RAM: 16 GB
- Storage: 10 GB free
- Time: Be patient (days for 500 images)

**Recommended:**
- GPU: NVIDIA with 8GB+ VRAM (RTX 3060+)
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 50 GB free (SSD)
- Time: Overnight for 500 images

**Optimal:**
- GPU: NVIDIA A100 or V100
- CPU: 16+ cores
- RAM: 64 GB
- Storage: 100 GB free (NVMe SSD)
- Time: Hours for 500 images

---

## Expected Results

### Good Segmentation
- Each plant has one complete mask
- Overlapping leaves are correctly assigned
- Small seedlings are captured
- No masks are merged
- Pot centers are inside their masks

### Poor Segmentation (needs adjustment)
- Plants are merged (use negative prompts)
- Masks too small (check pot centers)
- Missing plants (verify pot annotation)
- Masks include neighboring plants (increase negative prompt distance)

---

## Next Step

Once segmentation is complete and verified:

```bash
cd ../3_extract
python extract_traits.py
```

See [../3_extract/README.md](../3_extract/README.md) for trait extraction instructions.