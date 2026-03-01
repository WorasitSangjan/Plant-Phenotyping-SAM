"""
Configuration and utilities for maize seedling phenotyping pipeline
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class Config:
    """Project configuration"""
    
    # Paths
    DATA_DIR = Path("/mnt/user-data/uploads")
    OUTPUT_DIR = Path("/mnt/user-data/outputs")
    
    # Image settings
    RGB_PATTERN = "rgb_*.jpg"
    NIR_PATTERN = "nir_*.jpg"
    
    # Pot grid settings (will be refined during annotation)
    EXPECTED_ROWS = 4  # Approximate number of rows
    EXPECTED_COLS = 8  # Approximate number of columns
    
    # Scale calibration (pixels to cm²)
    # TODO: Measure from color calibration target in images
    PIXELS_PER_CM2 = None  # Will be calculated from calibration target
    
    # SAM settings
    SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"  # Will download if needed
    SAM_MODEL_TYPE = "vit_h"
    
    # Trait extraction settings
    VEGETATION_INDICES = ['ExG', 'VARI', 'GLI', 'NGRDI']
    
    @staticmethod
    def ensure_output_dir():
        """Create output directory if it doesn't exist"""
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_pot_centers(pot_centers: Dict[str, List[Tuple[int, int]]], 
                      filepath: Path):
    """Save pot center coordinates to JSON"""
    # Convert tuples to lists for JSON serialization
    data = {
        image_name: [list(coord) for coord in coords]
        for image_name, coords in pot_centers.items()
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved pot centers to {filepath}")


def load_pot_centers(filepath: Path) -> Dict[str, List[Tuple[int, int]]]:
    """Load pot center coordinates from JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    # Convert lists back to tuples
    pot_centers = {
        image_name: [tuple(coord) for coord in coords]
        for image_name, coords in data.items()
    }
    print(f"Loaded pot centers from {filepath}")
    return pot_centers


def calculate_pixel_to_cm2_scale(calibration_target_size_pixels: float,
                                  calibration_target_size_cm: float) -> float:
    """
    Calculate conversion from pixels to cm²
    
    Args:
        calibration_target_size_pixels: Width or height of calibration target in pixels
        calibration_target_size_cm: Width or height of calibration target in cm
    
    Returns:
        Conversion factor: 1 pixel² = X cm²
    """
    pixels_per_cm = calibration_target_size_pixels / calibration_target_size_cm
    pixels_per_cm2 = pixels_per_cm ** 2
    return 1.0 / pixels_per_cm2


def parse_date_from_filename(filename: str) -> str:
    """
    Extract date from filename like 'rgb_061325.jpg' -> '2025-06-13'
    """
    # Extract MMDDYY portion
    parts = filename.split('_')
    if len(parts) < 2:
        return "unknown"
    
    date_str = parts[1].replace('.jpg', '')
    if len(date_str) != 6:
        return "unknown"
    
    mm, dd, yy = date_str[:2], date_str[2:4], date_str[4:]
    # Assume 20XX for year
    return f"20{yy}-{mm}-{dd}"


def get_image_pairs(data_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    Get paired RGB and NIR images with their dates
    
    Returns:
        List of (rgb_path, nir_path, date) tuples
    """
    rgb_files = sorted(data_dir.glob(Config.RGB_PATTERN))
    nir_files = sorted(data_dir.glob(Config.NIR_PATTERN))
    
    pairs = []
    for rgb_path in rgb_files:
        # Find matching NIR file
        date_code = rgb_path.stem.split('_')[1]
        nir_path = data_dir / f"nir_{date_code}.jpg"
        
        if nir_path.exists():
            date = parse_date_from_filename(rgb_path.name)
            pairs.append((rgb_path, nir_path, date))
        else:
            print(f"Warning: No matching NIR file for {rgb_path.name}")
    
    return pairs
