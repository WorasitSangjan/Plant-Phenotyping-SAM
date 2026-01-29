"""
Automatic pot detection using circle detection

This script attempts to automatically detect pot centers using Hough Circle Transform.
Results may need manual correction, but it's much faster than full manual annotation.

Author: Worasit Sangjan
Date Created: 29 January 2026
Version: 1.1   
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).parent))
from config import Config, save_pot_centers


def detect_pots_hough(image_path: Path, 
                      min_radius: int = 60, 
                      max_radius: int = 120,
                      min_dist: int = 100) -> list:
    """Detect pot centers using Hough Circle Transform"""
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is None:
        print("Warning: No circles detected")
        return []
    
    # Convert to list of (x, y) tuples
    circles = np.round(circles[0, :]).astype(int)
    pot_centers = [(int(x), int(y)) for x, y, r in circles]
    
    # Sort by position (top to bottom, left to right)
    pot_centers.sort(key=lambda p: (p[1], p[0]))
    
    return pot_centers


def detect_pots_grid(image_path: Path,
                     expected_rows: int = 4,
                     expected_cols: int = 8) -> list:
    """Detect pots assuming a regular grid layout"""
    # Load image to get dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    h, w = img.shape[:2]
    
    # Estimate grid spacing (with margins)
    margin_x = w * 0.1  # 10% margin on sides
    margin_y = h * 0.1  # 10% margin on top/bottom
    
    usable_width = w - 2 * margin_x
    usable_height = h - 2 * margin_y
    
    spacing_x = usable_width / (expected_cols - 1) if expected_cols > 1 else 0
    spacing_y = usable_height / (expected_rows - 1) if expected_rows > 1 else 0
    
    # Generate grid points
    pot_centers = []
    for row in range(expected_rows):
        for col in range(expected_cols):
            x = int(margin_x + col * spacing_x)
            y = int(margin_y + row * spacing_y)
            pot_centers.append((x, y))
    
    return pot_centers


def visualize_detections(image_path: Path, pot_centers: list, output_path: Path = None):
    """Create visualization of detected pot centers """
    img = cv2.imread(str(image_path))
    vis = img.copy()
    
    # Draw detected centers
    for i, (x, y) in enumerate(pot_centers):
        cv2.circle(vis, (x, y), 8, (0, 255, 0), 2)
        cv2.line(vis, (x-15, y), (x+15, y), (0, 255, 0), 1)
        cv2.line(vis, (x, y-15), (x, y+15), (0, 255, 0), 1)
        cv2.putText(vis, str(i+1), (x+12, y-12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Add info text
    info = f"Detected {len(pot_centers)} pots"
    cv2.putText(vis, info, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if output_path:
        cv2.imwrite(str(output_path), vis)
        print(f"Saved visualization to {output_path}")
    
    return vis


def auto_detect_and_save(image_path: Path, 
                         method: str = 'hough',
                         expected_rows: int = 4,
                         expected_cols: int = 8,
                         visualize: bool = True):
    """Automatically detect pots and save to pot_centers.json"""
    Config.ensure_output_dir()
    
    print(f"Detecting pots in {image_path.name}...")
    print(f"Method: {method}")
    
    # Detect pots
    if method == 'hough':
        pot_centers = detect_pots_hough(image_path)
    elif method == 'grid':
        pot_centers = detect_pots_grid(image_path, expected_rows, expected_cols)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Detected {len(pot_centers)} pots")
    
    if len(pot_centers) == 0:
        print("\nNo pots detected. Try:")
        print("  1. Using method='grid' instead")
        print("  2. Adjusting min_radius/max_radius parameters")
        print("  3. Manual annotation with 01_annotate_pots.py")
        return
    
    # Create visualization
    if visualize:
        vis_path = Config.OUTPUT_DIR / f"pot_detection_{image_path.stem}.jpg"
        visualize_detections(image_path, pot_centers, vis_path)
    
    # Save pot centers
    pot_centers_dict = {image_path.name: pot_centers}
    pot_centers_file = Config.OUTPUT_DIR / "pot_centers.json"
    
    # Load existing pot centers if they exist
    if pot_centers_file.exists():
        from config import load_pot_centers
        existing = load_pot_centers(pot_centers_file)
        pot_centers_dict.update(existing)
    
    save_pot_centers(pot_centers_dict, pot_centers_file)
    
    print(f"\nSaved pot centers to {pot_centers_file}")
    print(f"\nNext steps:")
    print(f"  1. Review detection visualization: {vis_path if visualize else 'N/A'}")
    print(f"  2. IMPORTANT: Verify and correct with:")
    print(f"     python 01_annotate_pots.py")
    print(f"  3. Copy to other images:")
    print(f"     python copy_pot_centers_to_all.py --reference {image_path.name}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Automatically detect pot centers'
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Reference image name (e.g., rgb_061325.jpg)'
    )
    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['hough', 'grid'],
        default='hough',
        help='Detection method: hough (circle detection) or grid (regular pattern)'
    )
    parser.add_argument(
        '--rows',
        type=int,
        default=4,
        help='Expected number of rows (for grid method)'
    )
    parser.add_argument(
        '--cols',
        type=int,
        default=8,
        help='Expected number of columns (for grid method)'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip visualization'
    )
    
    args = parser.parse_args()
    
    # Find image
    image_path = Config.DATA_DIR / args.image
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    auto_detect_and_save(
        image_path,
        method=args.method,
        expected_rows=args.rows,
        expected_cols=args.cols,
        visualize=not args.no_visualize
    )


if __name__ == "__main__":
    main()
