"""
Trait extraction from plant masks

Extracts phenotypic traits including:
- Leaf area
- Color metrics
- Vegetation indices
- Spatial features

Author: Worasit Sangjan
Date Created: 3 February 2026
Version: 1.0
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import sys

sys.path.append(str(Path(__file__).parent))
from config import Config


def calculate_vegetation_indices(rgb_pixels: np.ndarray) -> Dict[str, float]:
    """Calculate RGB-based vegetation indices"""
    if len(rgb_pixels) == 0:
        return {vi: np.nan for vi in Config.VEGETATION_INDICES}
    
    # Normalize to [0, 1]
    R = rgb_pixels[:, 0].astype(float) / 255.0
    G = rgb_pixels[:, 1].astype(float) / 255.0
    B = rgb_pixels[:, 2].astype(float) / 255.0
    
    # Avoid division by zero
    epsilon = 1e-7
    
    indices = {}
    
    # Excess Green (ExG)
    # ExG = 2*G - R - B
    indices['ExG'] = float(np.mean(2*G - R - B))
    
    # Visible Atmospherically Resistant Index (VARI)
    # VARI = (G - R) / (G + R - B)
    denominator = G + R - B
    denominator[np.abs(denominator) < epsilon] = epsilon
    vari = (G - R) / denominator
    indices['VARI'] = float(np.mean(vari))
    
    # Green Leaf Index (GLI)
    # GLI = (2*G - R - B) / (2*G + R + B)
    denominator = 2*G + R + B
    denominator[denominator < epsilon] = epsilon
    gli = (2*G - R - B) / denominator
    indices['GLI'] = float(np.mean(gli))
    
    # Normalized Green-Red Difference Index (NGRDI)
    # NGRDI = (G - R) / (G + R)
    denominator = G + R
    denominator[denominator < epsilon] = epsilon
    ngrdi = (G - R) / denominator
    indices['NGRDI'] = float(np.mean(ngrdi))
    
    return indices


def extract_color_statistics(rgb_pixels: np.ndarray) -> Dict[str, float]:
    """Extract color statistics from RGB pixels"""
    if len(rgb_pixels) == 0:
        return {
            'mean_R': np.nan, 'mean_G': np.nan, 'mean_B': np.nan,
            'std_R': np.nan, 'std_G': np.nan, 'std_B': np.nan,
            'median_R': np.nan, 'median_G': np.nan, 'median_B': np.nan
        }
    
    stats = {}
    
    # RGB statistics
    stats['mean_R'] = float(np.mean(rgb_pixels[:, 0]))
    stats['mean_G'] = float(np.mean(rgb_pixels[:, 1]))
    stats['mean_B'] = float(np.mean(rgb_pixels[:, 2]))
    
    stats['std_R'] = float(np.std(rgb_pixels[:, 0]))
    stats['std_G'] = float(np.std(rgb_pixels[:, 1]))
    stats['std_B'] = float(np.std(rgb_pixels[:, 2]))
    
    stats['median_R'] = float(np.median(rgb_pixels[:, 0]))
    stats['median_G'] = float(np.median(rgb_pixels[:, 1]))
    stats['median_B'] = float(np.median(rgb_pixels[:, 2]))
    
    # Convert to HSV for additional statistics
    rgb_mean = np.uint8([[rgb_pixels.mean(axis=0)]])
    hsv = cv2.cvtColor(rgb_mean, cv2.COLOR_RGB2HSV)[0, 0]
    
    stats['mean_H'] = float(hsv[0])
    stats['mean_S'] = float(hsv[1])
    stats['mean_V'] = float(hsv[2])
    
    return stats


def calculate_spatial_features(mask: np.ndarray, 
                               pot_center: Tuple[int, int]) -> Dict[str, float]:
    """Calculate spatial features of the plant"""
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return {
            'centroid_x': np.nan,
            'centroid_y': np.nan,
            'centroid_dist_from_pot': np.nan,
            'bounding_box_width': np.nan,
            'bounding_box_height': np.nan,
            'convex_hull_area': np.nan,
            'solidity': np.nan
        }
    
    # Get largest contour (main plant body)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate moments
    M = cv2.moments(largest_contour)
    
    features = {}
    
    # Centroid
    if M['m00'] > 0:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        features['centroid_x'] = float(cx)
        features['centroid_y'] = float(cy)
        
        # Distance from pot center
        dist = np.sqrt((cx - pot_center[0])**2 + (cy - pot_center[1])**2)
        features['centroid_dist_from_pot'] = float(dist)
    else:
        features['centroid_x'] = np.nan
        features['centroid_y'] = np.nan
        features['centroid_dist_from_pot'] = np.nan
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    features['bounding_box_width'] = float(w)
    features['bounding_box_height'] = float(h)
    
    # Convex hull and solidity
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(largest_contour)
    
    features['convex_hull_area'] = float(hull_area)
    features['solidity'] = float(contour_area / hull_area) if hull_area > 0 else np.nan
    
    return features


def calculate_pot_overlap(mask: np.ndarray,
                          pot_center: Tuple[int, int],
                          pot_radius: int = 50) -> Dict[str, float]:
    """Calculate how much of the plant extends beyond its pot"""
    h, w = mask.shape
    
    # Create circular pot mask
    y_coords, x_coords = np.ogrid[:h, :w]
    pot_mask = ((x_coords - pot_center[0])**2 + 
                (y_coords - pot_center[1])**2 <= pot_radius**2)
    
    # Calculate areas
    total_plant_area = mask.sum()
    plant_inside_pot = (mask & pot_mask).sum()
    plant_outside_pot = (mask & ~pot_mask).sum()
    
    spillover_ratio = float(plant_outside_pot / total_plant_area) if total_plant_area > 0 else 0.0
    
    return {
        'plant_inside_pot_area': int(plant_inside_pot),
        'plant_outside_pot_area': int(plant_outside_pot),
        'spillover_ratio': spillover_ratio
    }


def extract_plant_traits(rgb_image: np.ndarray,
                         nir_image: np.ndarray,
                         rgb_mask: np.ndarray,
                         nir_mask: np.ndarray,
                         pot_center: Tuple[int, int],
                         plant_id: int,
                         pot_radius: int = 50,
                         pixels_to_cm2: float = None) -> Dict:
    """Extract all traits for a single plant"""
    traits = {'plant_id': plant_id}
    
    # Basic area measurements
    rgb_area_pixels = int(rgb_mask.sum())
    nir_area_pixels = int(nir_mask.sum())
    
    traits['rgb_leaf_area_pixels'] = rgb_area_pixels
    traits['nir_leaf_area_pixels'] = nir_area_pixels
    
    # Convert to cm² if conversion factor provided
    if pixels_to_cm2 is not None:
        traits['rgb_leaf_area_cm2'] = float(rgb_area_pixels * pixels_to_cm2)
        traits['nir_leaf_area_cm2'] = float(nir_area_pixels * pixels_to_cm2)
    
    # Extract RGB pixels
    rgb_pixels = rgb_image[rgb_mask > 0]
    
    # Color statistics
    color_stats = extract_color_statistics(rgb_pixels)
    traits.update(color_stats)
    
    # Vegetation indices
    vi_values = calculate_vegetation_indices(rgb_pixels)
    traits.update(vi_values)
    
    # Spatial features (use RGB mask as reference)
    spatial_features = calculate_spatial_features(rgb_mask, pot_center)
    traits.update(spatial_features)
    
    # Pot overlap/spillover
    spillover = calculate_pot_overlap(rgb_mask, pot_center, pot_radius)
    traits.update(spillover)
    
    # Pot center location
    traits['pot_center_x'] = pot_center[0]
    traits['pot_center_y'] = pot_center[1]
    
    return traits


def process_all_masks(mask_dir: Path, 
                      pixels_to_cm2: float = None,
                      pot_radius: int = 50) -> pd.DataFrame:
    """Process all mask files and extract traits"""
    all_traits = []
    
    # Find all mask files
    mask_files = sorted(mask_dir.glob("masks_*/masks.pkl"))
    
    if not mask_files:
        print(f"No mask files found in {mask_dir}")
        return pd.DataFrame()
    
    print(f"Processing {len(mask_files)} mask files...")
    
    for mask_file in mask_files:
        print(f"\nProcessing {mask_file.parent.name}...")
        
        # Load mask data
        with open(mask_file, 'rb') as f:
            mask_data = pickle.load(f)
        
        date = mask_data['date']
        pot_centers = mask_data['pot_centers']
        rgb_masks = mask_data['rgb_masks']
        nir_masks = mask_data['nir_masks']
        
        # Load images
        rgb_image = cv2.imread(mask_data['rgb_image'])
        nir_image = cv2.imread(mask_data['nir_image'])
        
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        nir_image = cv2.cvtColor(nir_image, cv2.COLOR_BGR2RGB)
        
        # Process each plant
        for i, (rgb_mask, nir_mask, pot_center) in enumerate(zip(rgb_masks, nir_masks, pot_centers)):
            plant_id = i + 1
            
            # Extract traits
            traits = extract_plant_traits(
                rgb_image, nir_image,
                rgb_mask, nir_mask,
                pot_center, plant_id,
                pot_radius, pixels_to_cm2
            )
            
            # Add metadata
            traits['date'] = date
            traits['image_date'] = mask_data['rgb_image'].split('_')[-1].replace('.jpg', '')
            
            all_traits.append(traits)
        
        print(f"  Extracted traits from {len(pot_centers)} plants")
    
    # Create DataFrame
    df = pd.DataFrame(all_traits)
    
    # Reorder columns for better readability
    metadata_cols = ['date', 'image_date', 'plant_id', 'pot_center_x', 'pot_center_y']
    area_cols = [col for col in df.columns if 'area' in col]
    color_cols = [col for col in df.columns if any(x in col for x in ['mean_', 'std_', 'median_'])]
    vi_cols = Config.VEGETATION_INDICES
    spatial_cols = [col for col in df.columns if 'centroid' in col or 'bounding' in col or 'solidity' in col or 'hull' in col]
    spillover_cols = [col for col in df.columns if 'spillover' in col or 'inside' in col or 'outside' in col]
    
    ordered_cols = []
    for col_list in [metadata_cols, area_cols, color_cols, vi_cols, spatial_cols, spillover_cols]:
        ordered_cols.extend([col for col in col_list if col in df.columns])
    
    # Add any remaining columns
    remaining = [col for col in df.columns if col not in ordered_cols]
    ordered_cols.extend(remaining)
    
    df = df[ordered_cols]
    
    return df


def main():
    """Main function"""
    Config.ensure_output_dir()
    
    # Check if segmentation results exist
    seg_dir = Config.OUTPUT_DIR / "segmentation_results"
    if not seg_dir.exists():
        print(f"Error: Segmentation results not found at {seg_dir}")
        print("Please run 02_segment_plants.py first")
        return
    
    # Process all masks and extract traits
    # TODO: Measure actual pot radius and pixels_to_cm2 from calibration target
    pot_radius = 80  # Approximate, should be measured from images
    pixels_to_cm2 = None  # Will calculate from calibration target
    
    print("Extracting traits from all plants...")
    df = process_all_masks(seg_dir, pixels_to_cm2, pot_radius)
    
    if df.empty:
        print("No data extracted")
        return
    
    # Save results
    output_file = Config.OUTPUT_DIR / "plant_traits.csv"
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print("Trait extraction complete!")
    print(f"Extracted traits from {len(df)} plants across {df['date'].nunique()} dates")
    print(f"Results saved to: {output_file}")
    print("="*60)
    
    # Print summary statistics
    print("\nSummary statistics:")
    print(f"  Total plants: {len(df)}")
    print(f"  Dates: {df['date'].nunique()}")
    print(f"  Mean RGB leaf area: {df['rgb_leaf_area_pixels'].mean():.1f} pixels")
    print(f"  Mean spillover ratio: {df['spillover_ratio'].mean():.2%}")
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()