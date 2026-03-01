# Step 4: Results Visualization

This folder contains tools for creating visualizations and summary statistics from extracted trait data.

---

## Overview

**Purpose:** Generate plots, figures, and summary reports from trait data

**Key Outputs:**
- Growth curves over time
- Vegetation index trends
- Spatial distribution maps
- Color evolution plots
- Spillover analysis
- Summary statistics

**Time Required:** ~2 minutes for 500 images

---

## Generated Visualizations

### 1. Growth Over Time
- **Individual plant trajectories** (first 20 plants)
- **Average growth with standard deviation**
- Shows leaf area progression across dates

### 2. Vegetation Indices
- **ExG, VARI, GLI, NGRDI** over time
- Average values with error bars
- Tracks vegetation health and greenness

### 3. Spatial Distribution
- **Leaf area by pot position** (heatmap)
- **Spillover ratio by position** (heatmap)
- Reveals spatial patterns (edge effects, gradients)

### 4. Color Distribution
- **RGB channel evolution** over time
- Mean values with standard deviation
- Shows phenological changes

### 5. Spillover Analysis
- **Spillover ratio over time** (how plants grow beyond pots)
- **Relationship between area and spillover** (correlation)
- Identifies growth patterns

### 6. Summary Report (Text)
- Overall statistics
- Per-date summaries
- Growth metrics
- Key findings

---

## Files in This Folder

**`visualize_results.py`** - Main visualization script
- Loads plant_traits.csv
- Generates all plots
- Creates summary report
- Saves to visualizations/ folder

---

## Usage

### Basic Usage

```bash
# Generate all visualizations
python visualize_results.py
```

**Requirements:**
- Trait extraction must be complete (Step 3)
- plant_traits.csv must exist

**What it does:**
1. Loads trait data from CSV
2. Generates 5 plot types
3. Creates text summary report
4. Saves everything to visualizations/ folder

---

### Customization

Edit `visualize_results.py` to customize plots:

```python
# Line ~XX: Figure size
fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # Adjust width, height

# Line ~XX: DPI (resolution)
plt.savefig(output_path, dpi=300)  # Higher = better quality

# Line ~XX: Color scheme
cmap='viridis'  # Options: viridis, plasma, magma, RdYlGn, etc.

# Line ~XX: Number of plants to plot
for plant_id in df['plant_id'].unique()[:20]:  # Change 20 to any number
```

---

## Output

### Directory Structure
```
/mnt/user-data/outputs/visualizations/
├── growth_over_time.png           # Growth curves
├── vegetation_indices.png         # VI trends
├── spatial_distribution.png       # Spatial maps
├── color_distribution.png         # RGB evolution
├── spillover_analysis.png         # Spillover patterns
└── summary_report.txt             # Text statistics
```

---

## Visualization Details

### Growth Over Time (growth_over_time.png)

**Left panel: Individual plants**
- Shows first 20 plants (customizable)
- Each line = one plant
- Different colors for different plants
- Reveals individual variability

**Right panel: Average growth**
- Mean leaf area across all plants
- Error bars show standard deviation
- Shaded area = ±1 std dev
- Smooth growth trend

**What to look for:**
- Generally increasing trend
- Some individual variation is normal
- Flat or decreasing trends = possible issue
- Very large error bars = high variability

---

### Vegetation Indices (vegetation_indices.png)

**Four panels: ExG, VARI, GLI, NGRDI**
- Each index tracked over time
- Error bars show uncertainty
- Shaded area = ±1 std dev

**Interpretation:**
- **ExG**: Higher = greener, healthier
- **VARI**: Stable values indicate consistent vegetation
- **GLI**: Similar to ExG, normalized
- **NGRDI**: Green vs. red ratio

**What to look for:**
- Indices increase as plants grow
- Stabilize at mature stage
- Sudden drops = stress or senescence
- High variability = non-uniform conditions

---

### Spatial Distribution (spatial_distribution.png)

**Left panel: Leaf area by position**
- Each dot = one pot
- Color = leaf area (viridis colormap)
- Larger/yellower = more leaf area

**Right panel: Spillover by position**
- Color = spillover ratio (RdYlGn colormap)
- Red = high spillover (large plants)
- Green = low spillover (small plants)

**What to look for:**
- Edge effects (corners different from center)
- Gradients (one side larger than other)
- Clusters (spatial correlation)
- Outliers (unusually large/small plants)

---

### Color Distribution (color_distribution.png)

**Three panels: Red, Green, Blue channels**
- Mean value over time
- Error bars show variation

**What to look for:**
- **Green increases**: Plant growth, more chlorophyll
- **Red/Blue stable**: Background (soil, pots) consistent
- **All channels increase**: Possibly more leaf coverage

---

### Spillover Analysis (spillover_analysis.png)

**Left panel: Spillover over time**
- Average spillover ratio (%)
- Shows how much plants extend beyond pots
- Should increase as plants grow

**Right panel: Area vs. Spillover**
- Scatter plot by date
- Shows correlation between size and spillover
- Different colors = different dates

**What to look for:**
- Positive correlation (larger plants = more spillover)
- Spillover increases over time
- Eventually plateaus (pots "full")

---

## Customizing Plots

### Change Figure Style

```python
import seaborn as sns

# Set style
sns.set_style("whitegrid")  # Options: white, dark, whitegrid, darkgrid, ticks

# Set context (affects font sizes)
sns.set_context("paper")  # Options: paper, notebook, talk, poster

# Set color palette
sns.set_palette("husl")  # Options: deep, muted, pastel, bright, dark, colorblind
```

### Modify Individual Plots

```python
# Growth curves - change line style
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, alpha=0.7)

# Spatial distribution - change colormap
scatter = ax.scatter(x, y, c=values, cmap='plasma')

# Error bars - change cap size
plt.errorbar(x, y, yerr=error, capsize=5, capthick=2)
```

### Add Statistical Annotations

```python
from scipy import stats

# Add trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
plt.plot(x, slope*x + intercept, 'r--', label=f'R²={r_value**2:.3f}')

# Add significance markers
if p_value < 0.001:
    plt.text(x, y, '***', ha='center')
elif p_value < 0.01:
    plt.text(x, y, '**', ha='center')
elif p_value < 0.05:
    plt.text(x, y, '*', ha='center')
```

---

## Summary Report Format

### Example summary_report.txt

```
============================================================
PHENOTYPING SUMMARY REPORT
============================================================

Overall Statistics:
  Total plants measured: 3600
  Number of dates: 6
  Date range: 2025-06-13 to 2025-09-04

Growth Statistics (RGB Leaf Area):
  Mean: 4523.2 pixels
  Std: 2345.6 pixels
  Min: 523.1 pixels
  Max: 11234.5 pixels

Spillover Statistics:
  Mean spillover: 38.5%
  Plants with >50% spillover: 456

Vegetation Indices (Mean):
  ExG: 0.285
  VARI: 0.512
  GLI: 0.234
  NGRDI: 0.398

Per-Date Summary:

  2025-06-13:
    Plants: 600
    Mean area: 1234.5 pixels
    Mean spillover: 15.2%

  2025-06-19:
    Plants: 600
    Mean area: 2456.7 pixels
    Mean spillover: 25.8%

  ...

============================================================
```

---

## Advanced Customization

### Create Custom Plots

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('/mnt/user-data/outputs/plant_traits.csv')

# Custom plot: Growth rate distribution
df['days'] = (pd.to_datetime(df['date']) - pd.to_datetime(df['date'].min())).dt.days
growth_rates = df.groupby('plant_id').apply(
    lambda x: (x['rgb_leaf_area_pixels'].iloc[-1] - x['rgb_leaf_area_pixels'].iloc[0]) / 
              (x['days'].iloc[-1] - x['days'].iloc[0])
)

plt.hist(growth_rates, bins=30, edgecolor='black')
plt.xlabel('Growth Rate (pixels²/day)')
plt.ylabel('Number of Plants')
plt.title('Distribution of Plant Growth Rates')
plt.savefig('custom_growth_rate_distribution.png', dpi=300)
```

### Export Data for Other Tools

```python
# Export summary statistics
summary = df.groupby('date').agg({
    'rgb_leaf_area_pixels': ['mean', 'std', 'min', 'max'],
    'ExG': 'mean',
    'spillover_ratio': 'mean'
})
summary.to_csv('summary_statistics.csv')

# Export for Prism/GraphPad
df_prism = df.pivot(index='plant_id', columns='date', values='rgb_leaf_area_pixels')
df_prism.to_csv('data_for_prism.csv')

# Export for R
df.to_csv('data_for_R.csv', index=False)
```

---

## Quality Control

### Visual Inspection Checklist

**Growth curves:**
- [ ] Generally increasing over time
- [ ] Individual variation present but reasonable
- [ ] No sudden jumps or drops
- [ ] Error bars decrease over time (plants become more similar)

**Vegetation indices:**
- [ ] Values in expected ranges
- [ ] Trends match leaf area trends
- [ ] Not all identical (some variation expected)

**Spatial distribution:**
- [ ] No strong gradients (unless expected)
- [ ] Edge effects minimal
- [ ] Outliers are few

**Spillover:**
- [ ] Increases over time
- [ ] Positive correlation with leaf area
- [ ] Values between 0-100%

---

## Troubleshooting

### Issue: "No trait data found"
**Cause:** Step 3 not completed

**Solution:**
```bash
# Check if plant_traits.csv exists
ls -lh /mnt/user-data/outputs/plant_traits.csv

# If not, run trait extraction
cd ../3_extract
python extract_traits.py
```

---

### Issue: "Plots look wrong / unexpected"
**Cause:** Data quality issues

**Solution:**
```python
# Check data quality
import pandas as pd
df = pd.read_csv('/mnt/user-data/outputs/plant_traits.csv')

# Look for issues
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nValue ranges:\n{df.describe()}")

# Identify outliers
Q1 = df['rgb_leaf_area_pixels'].quantile(0.25)
Q3 = df['rgb_leaf_area_pixels'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['rgb_leaf_area_pixels'] < Q1 - 1.5*IQR) | 
              (df['rgb_leaf_area_pixels'] > Q3 + 1.5*IQR)]
print(f"\nOutliers:\n{outliers[['date', 'plant_id', 'rgb_leaf_area_pixels']]}")
```

---

### Issue: "Plots are low resolution"
**Cause:** Default DPI too low

**Solution:**
```python
# Edit visualize_results.py
# Change all savefig calls:
plt.savefig(output_path, dpi=600)  # Instead of dpi=300

# For very high quality (large file size):
plt.savefig(output_path, dpi=1200, bbox_inches='tight')
```

---

### Issue: "Want different plot types"
**Cause:** Default plots don't match your needs

**Solution:** Create custom visualization script:

```python
# custom_plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/mnt/user-data/outputs/plant_traits.csv')

# Example: Heatmap of traits over time
pivot = df.pivot_table(
    values=['ExG', 'VARI', 'GLI', 'NGRDI'],
    index='date',
    aggfunc='mean'
)

sns.heatmap(pivot.T, annot=True, fmt='.3f', cmap='RdYlGn')
plt.title('Vegetation Indices Heatmap')
plt.savefig('custom_vi_heatmap.png', dpi=300)
```

---

## Publication-Ready Figures

### Tips for Publication

**Resolution:**
```python
# For journal submission
dpi = 600  # Or 300 minimum

# For posters
dpi = 300  # Sufficient for large prints
```

**File formats:**
```python
# Vector format (scales infinitely)
plt.savefig('figure.pdf')  # Preferred for journals
plt.savefig('figure.svg')  # For editing in Illustrator

# Raster format (fixed resolution)
plt.savefig('figure.png', dpi=600)  # High quality
plt.savefig('figure.tiff', dpi=600)  # Some journals require TIFF
```

**Color schemes:**
```python
# Colorblind-friendly
cmap = 'viridis'  # or 'plasma', 'cividis'

# Sequential (one variable)
cmap = 'Blues'  # or 'Greens', 'Reds'

# Diverging (two extremes)
cmap = 'RdBu'  # or 'RdYlGn', 'PiYG'
```

**Font sizes:**
```python
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
```

---

## Best Practices

### Before Running
- Complete trait extraction (Step 3)
- Review trait data quality
- Decide on plot customizations

### After Running
- Review all plots visually
- Check summary report for anomalies
- Create custom plots if needed
- Export high-resolution versions for publication

---

## Example Interpretations

### Scenario 1: Normal Growth Pattern
```
✓ Leaf area increases steadily
✓ Spillover ratio increases from 10% to 50%
✓ ExG increases from 0.2 to 0.35
✓ No strong spatial patterns

Interpretation: Healthy, uniform growth
```

### Scenario 2: Edge Effect
```
Edge plants 20% larger than center 
Higher spillover at edges
Growth rate similar across locations

Interpretation: Edge effect (more light/space)
```

### Scenario 3: Stress Response
```
Growth slows after date 3
ExG decreases from 0.3 to 0.2
Spillover continues increasing

Interpretation: Possible stress (water/nutrients)
```

---

## Next Steps

After generating visualizations:

1. **Analyze results** - Interpret patterns and trends
2. **Statistical analysis** - Run statistical tests
3. **Create manuscript figures** - Combine plots for publication
4. **Share results** - Present to collaborators

### For Statistical Analysis:
```R
# In R
data <- read.csv('plant_traits.csv')

# ANOVA for treatment effects
model <- aov(rgb_leaf_area_pixels ~ treatment * date, data=data)
summary(model)

# Mixed effects model for repeated measures
library(lme4)
model <- lmer(rgb_leaf_area_pixels ~ date + (1|plant_id), data=data)
summary(model)
```