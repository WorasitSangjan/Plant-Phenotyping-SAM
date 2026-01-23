# SAM-Based Plant Phenotyping Pipeline

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-red.svg)
![Agriculture](https://img.shields.io/badge/Agriculture-Precision%20Agriculture-green.svg)
![Research](https://img.shields.io/badge/Research-USDA--ARS-navy.svg)

> **Automated extraction of per-plant phenotypic traits from top-down images using SAM (Segment Anything Model) with pot-anchored prompts. Designed for high-throughput plant phenotyping in controlled environments where leaf overlap is a challenge.**

---

## Repository Structure

```
Plant-Phenotyping-SAM/
│
├── 1_annotatation/              # Step 1: Pot center annotation
│   ├── README.md
│   ├── annotate_pots.py
│   ├── copy_pot_centers_to_all.py
│   └── auto_detect_pots.py
│
├── 2_segmentation/              # Step 2: SAM-based segmentation
│   ├── README.md
│   └── segment_plants.py
│
├── 3_extractation/              # Step 3: Trait extraction
│   ├── README.md
│   └── extract_traits.py
│
├── 4_visualization/            # Step 4: Visualization
│   ├── README.md
│   └── visualize_results.py
│
├── config.py                   # Global configuration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE
└── .gitignore
```
---

## Key Features

- **Handles Severe Leaf Overlap** - Correctly assigns overlapping leaves to their biological origin
- **Zero-Shot Segmentation** - Works with small datasets (no training required)
- **Biologically Correct** - Pot-anchored prompts ensure plant identity consistency
- **Comprehensive Traits** - Extracts 10+ phenotypic traits per plant
- **Scalable** - Annotate once, process 100+ images
- **Dual Imaging Support** - Processes both RGB and NIR images

---

## What This Pipeline Does

### The Problem
In high-density plant imaging, leaves frequently overlap across neighboring pots. Traditional segmentation methods fail because they cannot distinguish biological plant identity from geometric proximity.

### The Solution
This pipeline uses pot centers as biological identity anchors combined with SAM for accurate, zero-shot plant segmentation. Each plant is correctly segmented including leaves extending beyond its pot boundaries.

### Extracted Traits
- **Structural**: Leaf area, bounding box, convex hull, solidity, spillover ratio
- **Color**: RGB/HSV statistics (mean, std, median)
- **Vegetation Indices**: ExG, VARI, GLI, NGRDI
- **Spatial**: Centroid position, distance from pot center

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/WorasitSangjan/Plant-Phenotyping-SAM.git
cd plant-phenotyping-sam

# Install dependencies
pip install -r requirements.txt

# Run pipeline (4 steps)
cd 1_annotation && python annotate_pots.py             # Step 1: Annotate pot centers (5 min)
python copy_pot_centers_to_all.py --reference ...      # Copy to all images (10 sec)

cd ../2_segmentation && python segment_plants.py       # Step 2: Segment plants

cd ../3_extraction && python extract_traits.py         # Step 3: Extract traits (5 min)

cd ../4_visualization && python visualize_results.py   # Step 4: Visualize results (2 min)
```

---

## Installation

### Requirements
- Python 3.8+
- 8GB+ GPU (recommended) or 16GB+ RAM (CPU only)
- ~3GB disk space (for SAM checkpoint)

### Dependencies
```bash
pip install -r requirements.txt
```

**Core packages:**
- segment-anything
- opencv-python
- torch
- numpy
- pandas
- matplotlib
- seaborn

---

## Pipeline Overview

### Step 1: Annotate Pot Centers
[Go to 1_annotatation/](1_annotatation/)

Interactive tool to mark pot centers that serve as biological identity anchors.

**Time:** 5 minutes (one-time per imaging setup)

**Output:** `pot_centers.json`

---

### Step 2: Segment Plants
[Go to 2_segmentation/](2_segmentation/)

Uses SAM with pot-anchored prompts to segment each plant, correctly handling leaf overlap.

**Time:** ~20 hours for 500 images (automated)

**Output:** Binary masks + visualizations

---

### Step 3: Extract Traits
[Go to 3_extractation/](3_extractation/)

Extracts 30+ phenotypic traits from segmented plants.

**Time:** ~5 minutes for 500 images (automated)

**Output:** `plant_traits.csv`

---

### Step 4: Visualize Results
[Go to 4_visualization/](4_visualization/)

Generates publication-quality plots and summary statistics.

**Time:** ~2 minutes (automated)

**Output:** Growth curves, spatial maps, summary reports

---

## Use Cases

### Ideal For:
- Small to medium datasets (10-100 images)
- Plants in pots with severe leaf overlap
- Controlled imaging environments
- Longitudinal growth studies
- Multi-spectral imaging (RGB + NIR)

### Not Ideal For:
- Field imaging (uncontrolled conditions)
- Real-time processing requirements (use YOLO instead)
- Single-plant images (no overlap problem)

---

## Scientific Validation

This pipeline has been designed for research applications requiring:
- Accurate per-plant measurements
- Biological correctness (not geometric approximations)
- Repeatability and reproducibility
- Publication-quality outputs

**Recommended validation:**
1. Manual measurement of 10 random plants
2. Compare to automated measurements
3. Calculate R² and RMSE

---

## Documentation

- [**Quick Start Guide**](docs/QUICKSTART.md) - Get started in 5 minutes
- [**Scalable Workflow**](docs/SCALABLE_WORKFLOW.md) - Processing 100-1000+ images
- [**Visual Workflow**](docs/VISUAL_WORKFLOW.md) - Diagrams and examples

---

## Citation

If you use this pipeline in your research, please cite:

**SAM (Segment Anything Model):**
```bibtex
@article{kirillov2023segment,
  title={Segment anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}
```

**This Pipeline:**
```bibtex
@software{Plant-Phenotyping_SAM,
  title={SAM-Based Plant Phenotyping Pipeline},
  author={Worasit Sangjan},
  year={2026},
  url={https://github.com/WorasitSangjan/Plant-Phenotyping-SAM}
}
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Areas for contribution:
- Additional trait extraction functions
- New visualization types
- Performance optimizations
- Support for additional imaging modalities
- Integration with other phenotyping tools

---

## Contact

- **Issues:** [GitHub Issues](https://github.com/WorasitSangjan/Plant-Phenotyping-SAM/issues)
- **Email:** worasitsangjan.ws@gmail.com

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
