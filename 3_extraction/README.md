# Step 3: Trait Extraction

This folder contains tools for extracting comprehensive phenotypic traits from segmented plant masks.

---

## Overview

**Purpose:** Extract measurable traits from each plant mask

**Key Features:**
- Structural traits (area, shape, position)
- Color statistics (RGB, HSV)
- Vegetation indices (ExG, VARI, GLI, NGRDI)
- Spatial features (centroid, spillover)

---

## What Gets Extracted

### Structural Traits
- **Leaf area** (pixels and cm² if calibrated)
- **Bounding box** dimensions (width, height)
- **Convex hull area** (smallest convex polygon containing plant)
- **Solidity** (ratio of actual area to convex hull area)
- **Spillover ratio** (percentage of leaves outside pot)

### Color Traits (RGB-based)
- **Mean RGB** values (R, G, B channels)
- **Standard deviation** (color variation)
- **Median RGB** values (robust to outliers)
- **HSV statistics** (Hue, Saturation, Value)

### Vegetation Indices
- **ExG** (Excess Green): 2G - R - B
- **VARI** (Visible Atmospherically Resistant Index)
- **GLI** (Green Leaf Index)
- **NGRDI** (Normalized Green-Red Difference Index)

### Spatial Features
- **Centroid position** (x, y coordinates)
- **Distance from pot center**
- **Pot center location**
- **Plant inside/outside pot areas**

---

## Files in This Folder

**`extract_traits.py`** - Main trait extraction script
- Loads segmentation masks
- Calculates all traits
- Handles both RGB and NIR
- Outputs CSV file

---

## Usage

### Basic Usage

```bash
# Extract traits from all segmented plants
python extract_traits.py
```

**Requirements:**
- Segmentation must be complete (Step 2)
- segmentation_results/ folder must exist with masks

**What it does:**
1. Loads all masks.pkl files
2. For each plant in each image:
   - Extracts structural traits from mask
   - Extracts color traits from RGB pixels
   - Calculates vegetation indices
   - Measures spatial features
3. Combines all data into single CSV
4. Saves to plant_traits.csv

---

### Calibration (Important for Accurate Measurements)

Edit `extract_traits.py` to calibrate measurements:

```python
# Line ~XXX: Pot radius (measure from your images)
pot_radius = 80  # pixels

# Line ~XXX: Scale conversion (from calibration target)
pixels_to_cm2 = 0.01  # 1 pixel² = 0.01 cm²
```

#### How to Calibrate Pot Radius:
```python
# 1. Open any image in image viewer
# 2. Measure pot diameter in pixels (use ruler tool)
# 3. Divide by 2
# Example: 160 pixels diameter → 80 pixels radius

pot_radius = 80  # Your measured value
```

#### How to Calibrate Scale:
```python
# 1. Find color calibration target in images
# 2. Measure its width in pixels
# 3. Look up actual width (usually printed on target)
# 4. Calculate conversion

from config import calculate_pixel_to_cm2_scale

pixels_to_cm2 = calculate_pixel_to_cm2_scale(
    calibration_target_size_pixels=100,  # Measured in image
    calibration_target_size_cm=10         # Actual size
)
# Result: 0.01 (example)
```

---

## Output

### plant_traits.csv

**Format:** One row per plant per date

**Columns:**
```csv
date,image_date,plant_id,pot_center_x,pot_center_y,
rgb_leaf_area_pixels,nir_leaf_area_pixels,
rgb_leaf_area_cm2,nir_leaf_area_cm2,
mean_R,mean_G,mean_B,std_R,std_G,std_B,
median_R,median_G,median_B,
mean_H,mean_S,mean_V,
ExG,VARI,GLI,NGRDI,
centroid_x,centroid_y,centroid_dist_from_pot,
bounding_box_width,bounding_box_height,
convex_hull_area,solidity,
plant_inside_pot_area,plant_outside_pot_area,spillover_ratio
```

**Example rows:**
```csv
2025-06-13,061325,1,245,180,1523,1489,15.23,14.89,78.2,145.3,92.1,...
2025-06-13,061325,2,345,180,1789,1756,17.89,17.56,82.1,149.2,95.3,...
2025-06-19,061925,1,245,180,3456,3398,34.56,33.98,85.4,152.1,98.2,...
```

**Location:** `/mnt/user-data/outputs/plant_traits.csv`

---

## Trait Definitions

### Leaf Area
- **Units:** pixels² (or cm² if calibrated)
- **Meaning:** Total leaf surface area
- **Use:** Primary growth measurement
- **Expected range:** 500-10,000 pixels (varies with growth stage)

### Spillover Ratio
- **Units:** Ratio (0.0 to 1.0, or 0% to 100%)
- **Meaning:** Proportion of leaf area extending beyond pot
- **Use:** Measure of plant size relative to pot
- **Expected range:** 0.1-0.6 (10%-60%)
- **Interpretation:**
  - 0.0-0.2: Small plant, mostly within pot
  - 0.2-0.5: Medium plant, significant overhang
  - 0.5+: Large plant, extensive overhang

### ExG (Excess Green)
- **Formula:** 2G - R - B
- **Range:** -0.5 to 0.5 (higher = greener)
- **Use:** Simple greenness index
- **Sensitive to:** Chlorophyll content, plant health

### VARI (Visible Atmospherically Resistant Index)
- **Formula:** (G - R) / (G + R - B)
- **Range:** -1 to 1
- **Use:** Vegetation detection, robust to lighting
- **Sensitive to:** Green vegetation vs. soil

### GLI (Green Leaf Index)
- **Formula:** (2G - R - B) / (2G + R + B)
- **Range:** -1 to 1
- **Use:** Normalized greenness
- **Sensitive to:** Overall greenness

### NGRDI (Normalized Green-Red Difference Index)
- **Formula:** (G - R) / (G + R)
- **Range:** -1 to 1
- **Use:** Simple normalized vegetation index
- **Sensitive to:** Green vs. red ratio

### Solidity
- **Formula:** Plant area / Convex hull area
- **Range:** 0 to 1
- **Use:** Measure of leaf compactness
- **Interpretation:**
  - ~1.0: Compact, dense leaves
  - ~0.6-0.8: Normal maize morphology
  - <0.5: Elongated or sparse leaves

---

## Customization

### Add Custom Traits

Edit `extract_traits.py` to add your own trait extraction:

```python
def extract_custom_trait(rgb_image, mask, pot_center):
    """
    Extract your custom trait
    
    Args:
        rgb_image: RGB image array (H, W, 3)
        mask: Binary mask (H, W)
        pot_center: (x, y) tuple
    
    Returns:
        Dictionary with trait values
    """
    # Example: Count number of leaves (complex, needs additional segmentation)
    # Example: Measure stem diameter
    # Example: Calculate leaf angle distribution
    
    custom_traits = {
        'custom_trait_1': value1,
        'custom_trait_2': value2,
    }
    
    return custom_traits

# Add to extract_plant_traits() function:
custom = extract_custom_trait(rgb_image, rgb_mask, pot_center)
traits.update(custom)
```

### Change Vegetation Indices

Edit `../config.py`:

```python
# Add or remove indices
VEGETATION_INDICES = ['ExG', 'VARI', 'GLI', 'NGRDI', 'YourCustomVI']
```

Then implement in `extract_traits.py`:

```python
def calculate_vegetation_indices(rgb_pixels):
    # ... existing code ...
    
    # Add your custom VI
    indices['YourCustomVI'] = float(np.mean(your_formula))
    
    return indices
```

---

## Quality Control

### Quick Checks

```python
import pandas as pd

# Load trait data
df = pd.read_csv('/mnt/user-data/outputs/plant_traits.csv')

# Check 1: Expected number of plants
n_dates = df['date'].nunique()
n_plants_per_date = df.groupby('date').size()
print(f"Plants per date:\n{n_plants_per_date}")

# Check 2: Leaf area increases over time
avg_area = df.groupby('date')['rgb_leaf_area_pixels'].mean()
print(f"\nAverage leaf area over time:\n{avg_area}")
# Should generally increase

# Check 3: No NaN values in critical traits
critical_traits = ['rgb_leaf_area_pixels', 'ExG', 'spillover_ratio']
for trait in critical_traits:
    n_missing = df[trait].isna().sum()
    if n_missing > 0:
        print(f"Warning: {n_missing} missing values in {trait}")

# Check 4: Values are in reasonable ranges
print(f"\nLeaf area range: {df['rgb_leaf_area_pixels'].min():.0f} - {df['rgb_leaf_area_pixels'].max():.0f}")
print(f"ExG range: {df['ExG'].min():.3f} - {df['ExG'].max():.3f}")
print(f"Spillover range: {df['spillover_ratio'].min():.1%} - {df['spillover_ratio'].max():.1%}")
```

### Statistical Summary

```python
# Overall summary
print(df.describe())

# Per-date summary
print(df.groupby('date').agg({
    'rgb_leaf_area_pixels': ['mean', 'std', 'min', 'max'],
    'spillover_ratio': 'mean',
    'ExG': 'mean'
}))

# Growth rate calculation
df_sorted = df.sort_values(['plant_id', 'date'])
df_sorted['area_change'] = df_sorted.groupby('plant_id')['rgb_leaf_area_pixels'].diff()
print(f"\nAverage daily growth: {df_sorted['area_change'].mean():.1f} pixels²/day")
```

---

## Troubleshooting

### Issue: "No segmentation results found"
**Cause:** Step 2 not completed

**Solution:**
```bash
# Check if segmentation results exist
ls -lh /mnt/user-data/outputs/segmentation_results/

# If empty, run segmentation first
cd ../2_segment
python segment_plants.py
```

---

### Issue: "Many NaN values in traits"
**Cause:** Empty masks or failed segmentation

**Solution:**
```python
# Identify which plants have issues
df = pd.read_csv('/mnt/user-data/outputs/plant_traits.csv')
problem_plants = df[df['rgb_leaf_area_pixels'].isna()]
print(problem_plants[['date', 'plant_id']])

# Check those specific masks
import pickle
with open(f'/mnt/user-data/outputs/segmentation_results/masks_{date}/masks.pkl', 'rb') as f:
    data = pickle.load(f)
    
# If masks are empty, re-run segmentation for those dates
```

---

### Issue: "Spillover ratios don't make sense"
**Cause:** Pot radius parameter is incorrect

**Solution:**
```python
# Measure actual pot radius from images
# Update pot_radius in extract_traits.py

# Re-run trait extraction (fast!)
python extract_traits.py
```

---

### Issue: "Leaf areas seem too small/large"
**Cause:** Scale conversion is wrong or missing

**Solution:**
```python
# Verify calibration
# If pixels_to_cm2 is None, areas are in pixels (correct)
# If pixels_to_cm2 is set, verify the conversion factor

# Measure calibration target to verify
# Update pixels_to_cm2 in extract_traits.py
```

---

### Issue: "Color values are all similar across plants"
**Cause:** Normal for uniform lighting/growth stage

**Solution:**
- This might be expected if plants are at same stage
- Check color variation increases over time
- Verify RGB images have good color information (not grayscale)

---

## Best Practices

### Before Running
- Complete segmentation (Step 2)
- Calibrate pot_radius and pixels_to_cm2
- Verify segmentation quality visually

### During Processing
- Monitor console output for errors
- Check progress (shows "Processing date X...")

### After Processing
- Run quality control checks
- Verify CSV has expected number of rows
- Check value ranges are reasonable
- Plot growth curves to visualize trends

---

## Next Step

Once traits are extracted and verified:

```bash
cd ../4_visualize
python visualize_results.py
```

See [../4_visualize/README.md](../4_visualize/README.md) for visualization instructions.