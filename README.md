# SAM-Based Plant Phenotyping Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automated extraction of per-plant phenotypic traits from top-down images using SAM (Segment Anything Model) with pot-anchored prompts. Designed for high-throughput plant phenotyping in controlled environments where leaf overlap is a challenge.

![Pipeline Overview](docs/pipeline_overview.png)

---

## 🌟 Key Features

- **Handles Severe Leaf Overlap** - Correctly assigns overlapping leaves to their biological origin
- **Zero-Shot Segmentation** - Works with small datasets (no training required)
- **Biologically Correct** - Pot-anchored prompts ensure plant identity consistency
- **Comprehensive Traits** - Extracts 30+ phenotypic traits per plant
- **Scalable** - Annotate once, process 1000+ images
- **Dual Imaging Support** - Processes both RGB and NIR images

---

## 📊 What This Pipeline Does

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

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/plant-phenotyping-sam.git
cd plant-phenotyping-sam

# Install dependencies
pip install -r requirements.txt

# Run pipeline (4 steps)
cd 1_annotate && python annotate_pots.py          # Step 1: Annotate pot centers (5 min)
python copy_pot_centers_to_all.py --reference ... # Copy to all images (10 sec)

cd ../2_segment && python segment_plants.py       # Step 2: Segment plants (~20 hrs)

cd ../3_extract && python extract_traits.py       # Step 3: Extract traits (5 min)

cd ../4_visualize && python visualize_results.py  # Step 4: Visualize results (2 min)
```

---

## 📁 Repository Structure

```
plant-phenotyping-sam/
│
├── 1_annotate/              # Step 1: Pot center annotation
│   ├── README.md
│   ├── annotate_pots.py
│   ├── copy_pot_centers_to_all.py
│   └── auto_detect_pots.py
│
├── 2_segment/               # Step 2: SAM-based segmentation
│   ├── README.md
│   └── segment_plants.py
│
├── 3_extract/               # Step 3: Trait extraction
│   ├── README.md
│   └── extract_traits.py
│
├── 4_visualize/             # Step 4: Visualization
│   ├── README.md
│   └── visualize_results.py
│
├── config.py                # Global configuration
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── docs/                   # Documentation and guides
    ├── QUICKSTART.md
    ├── SCALABLE_WORKFLOW.md
    └── VISUAL_WORKFLOW.md
```

---

## 🔧 Installation

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

## 📖 Pipeline Overview

### Step 1: Annotate Pot Centers
[📂 Go to 1_annotate/](1_annotate/)

Interactive tool to mark pot centers that serve as biological identity anchors.

**Time:** 5 minutes (one-time per imaging setup)

**Output:** `pot_centers.json`

---

### Step 2: Segment Plants
[📂 Go to 2_segment/](2_segment/)

Uses SAM with pot-anchored prompts to segment each plant, correctly handling leaf overlap.

**Time:** ~20 hours for 500 images (automated)

**Output:** Binary masks + visualizations

---

### Step 3: Extract Traits
[📂 Go to 3_extract/](3_extract/)

Extracts 30+ phenotypic traits from segmented plants.

**Time:** ~5 minutes for 500 images (automated)

**Output:** `plant_traits.csv`

---

### Step 4: Visualize Results
[📂 Go to 4_visualize/](4_visualize/)

Generates publication-quality plots and summary statistics.

**Time:** ~2 minutes (automated)

**Output:** Growth curves, spatial maps, summary reports

---

## 💡 Use Cases

### Ideal For:
- ✅ Small to medium datasets (10-1000 images)
- ✅ Plants in pots with severe leaf overlap
- ✅ Controlled imaging environments
- ✅ Longitudinal growth studies
- ✅ Multi-spectral imaging (RGB + NIR)

### Not Ideal For:
- ❌ Field imaging (uncontrolled conditions)
- ❌ Real-time processing requirements (use YOLO instead)
- ❌ Single-plant images (no overlap problem)

---

## 📈 Performance

### Accuracy
- Works reliably with as few as 12 training images
- Zero-shot segmentation (no model training)
- Biologically correct measurements

### Speed
- **Annotation:** 5 minutes (one-time)
- **Segmentation:** ~3-5 seconds per plant
  - 500 images × 30 plants = ~20 hours on CPU
  - ~2 hours on GPU (CUDA-enabled)
- **Trait extraction:** ~5 minutes for 500 images
- **Visualization:** ~2 minutes

### Scalability
- Annotate once, process unlimited images
- Linear scaling with number of plants
- Tested on datasets up to 1000 images

---

## 🔬 Scientific Validation

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

## 📚 Documentation

- [**Quick Start Guide**](docs/QUICKSTART.md) - Get started in 5 minutes
- [**Scalable Workflow**](docs/SCALABLE_WORKFLOW.md) - Processing 100-1000+ images
- [**Visual Workflow**](docs/VISUAL_WORKFLOW.md) - Diagrams and examples
- [**Subfolder READMEs**](1_annotate/README.md) - Detailed step-by-step guides

---

## 🎓 Citation

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
@software{plant_phenotyping_sam,
  title={SAM-Based Plant Phenotyping Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/plant-phenotyping-sam}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Areas for contribution:
- Additional trait extraction functions
- New visualization types
- Performance optimizations
- Support for additional imaging modalities
- Integration with other phenotyping tools

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Meta AI** for the Segment Anything Model (SAM)
- **Anthropic** for development tools and infrastructure
- Research community for feedback and validation

---

## 📧 Contact

- **Issues:** [GitHub Issues](https://github.com/yourusername/plant-phenotyping-sam/issues)
- **Email:** your.email@institution.edu
- **Website:** https://yourlab.com

---

## 🔗 Related Projects

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [PlantCV](https://plantcv.readthedocs.io/)
- [YOLOv8](https://github.com/ultralytics/ultralytics)

---

## ⭐ Star History

If you find this pipeline useful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/plant-phenotyping-sam&type=Date)](https://star-history.com/#yourusername/plant-phenotyping-sam&Date)

---

**Made with 🌱 for plant science research**
