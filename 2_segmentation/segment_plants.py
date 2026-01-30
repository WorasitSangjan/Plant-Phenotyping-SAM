"""
SAM-based plant segmentation with pot-anchored prompts

This script segments each plant using Segment Anything Model (SAM),
using pot centers as positive prompts to ensure biological correctness.

Author: Worasit Sangjan
Date Created: 30 January 2026
Version: 1.1   
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import pickle

sys.path.append(str(Path(__file__).parent))
from config import Config, load_pot_centers, get_image_pairs

# Import SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Error: segment_anything not installed.")
    print("Install with: pip install segment-anything")
    sys.exit(1)


class SAMPlantSegmenter:
    """Segment plants using SAM with pot-anchored prompts"""
    
    def __init__(self, sam_checkpoint: Optional[str] = None):
        """Initialize SAM segmenter"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load SAM model
        if sam_checkpoint is None:
            sam_checkpoint = self._download_sam_checkpoint()
        
        print(f"Loading SAM model from {sam_checkpoint}")
        sam = sam_model_registry[Config.SAM_MODEL_TYPE](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        
        self.predictor = SamPredictor(sam)
        print("SAM model loaded successfully")
    
    def _download_sam_checkpoint(self) -> str:
        """Download SAM checkpoint if not exists"""
        checkpoint_path = Path("/home/claude") / Config.SAM_CHECKPOINT
        
        if not checkpoint_path.exists():
            print("Downloading SAM checkpoint...")
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            urllib.request.urlretrieve(url, checkpoint_path)
            print(f"Downloaded SAM checkpoint to {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def segment_plant(self, 
                      image: np.ndarray,
                      pot_center: Tuple[int, int],
                      neighbor_centers: Optional[List[Tuple[int, int]]] = None,
                      use_nir: bool = False) -> np.ndarray:
        """Segment a single plant using SAM"""
        # Set image in predictor
        self.predictor.set_image(image)
        
        # Prepare prompts
        # Positive prompt: slightly offset toward plant center (not exact pot center)
        # This helps SAM focus on the stem/base
        positive_points = np.array([pot_center])
        positive_labels = np.array([1])  # 1 = foreground
        
        # Optional: Add negative prompts at neighboring pots
        if neighbor_centers and len(neighbor_centers) > 0:
            negative_points = np.array(neighbor_centers)
            negative_labels = np.array([0] * len(neighbor_centers))  # 0 = background
            
            input_points = np.vstack([positive_points, negative_points])
            input_labels = np.concatenate([positive_labels, negative_labels])
        else:
            input_points = positive_points
            input_labels = positive_labels
        
        # Predict mask
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True  # Get multiple mask proposals
        )
        
        # Select best mask (highest score)
        best_mask_idx = scores.argmax()
        mask = masks[best_mask_idx]
        
        return mask.astype(np.uint8)
    
    def segment_all_plants(self,
                          image_path: Path,
                          pot_centers: List[Tuple[int, int]],
                          use_negative_prompts: bool = True) -> List[np.ndarray]:
        """Segment all plants in an image"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"Segmenting {len(pot_centers)} plants in {image_path.name}...")
        
        masks = []
        for i, pot_center in enumerate(pot_centers):
            # Get neighboring pot centers for negative prompts
            neighbors = None
            if use_negative_prompts:
                # Find nearby pots (within 200 pixels)
                neighbors = []
                for j, other_center in enumerate(pot_centers):
                    if i != j:
                        dist = np.sqrt((pot_center[0] - other_center[0])**2 + 
                                     (pot_center[1] - other_center[1])**2)
                        if dist < 200:  # Threshold for "neighboring"
                            neighbors.append(other_center)
            
            # Segment this plant
            mask = self.segment_plant(image_rgb, pot_center, neighbors)
            masks.append(mask)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(pot_centers)} plants")
        
        print(f"  Completed all {len(pot_centers)} plants")
        return masks
    
    def create_overlap_visualization(self,
                                     image: np.ndarray,
                                     masks: List[np.ndarray],
                                     pot_centers: List[Tuple[int, int]]) -> np.ndarray:
        """Create visualization showing plant masks with different colors"""
        vis = image.copy()
        
        # Generate distinct colors for each plant
        colors = []
        for i in range(len(masks)):
            hue = int(180 * i / len(masks))
            color = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
            colors.append(tuple(map(int, color)))
        
        # Overlay masks with transparency
        overlay = vis.copy()
        for mask, color, pot_center in zip(masks, colors, pot_centers):
            # Apply mask with color
            overlay[mask > 0] = color
            
            # Draw pot center
            cv2.circle(vis, pot_center, 5, (255, 255, 255), -1)
            cv2.circle(vis, pot_center, 6, (0, 0, 0), 1)
        
        # Blend original and overlay
        vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0)
        
        return vis


def segment_all_images(pot_centers_dict: Dict[str, List[Tuple[int, int]]],
                       output_dir: Path,
                       use_negative_prompts: bool = True):
    """Segment all images and save masks"""
    # Initialize SAM
    segmenter = SAMPlantSegmenter()
    
    # Get image pairs
    image_pairs = get_image_pairs(Config.DATA_DIR)
    
    # Process each date
    results = {}
    for rgb_path, nir_path, date in image_pairs:
        print(f"\n{'='*60}")
        print(f"Processing date: {date}")
        print(f"  RGB: {rgb_path.name}")
        print(f"  NIR: {nir_path.name}")
        print(f"{'='*60}")
        
        # Get pot centers for this image
        if rgb_path.name not in pot_centers_dict:
            print(f"Warning: No pot centers found for {rgb_path.name}, skipping")
            continue
        
        pot_centers = pot_centers_dict[rgb_path.name]
        
        # Segment RGB image
        print(f"\nSegmenting RGB image...")
        rgb_masks = segmenter.segment_all_plants(rgb_path, pot_centers, use_negative_prompts)
        
        # Segment NIR image (using same pot centers)
        print(f"\nSegmenting NIR image...")
        nir_masks = segmenter.segment_all_plants(nir_path, pot_centers, use_negative_prompts)
        
        # Save masks
        date_dir = output_dir / f"masks_{date}"
        date_dir.mkdir(exist_ok=True)
        
        # Save as pickle for easy loading
        mask_data = {
            'rgb_masks': rgb_masks,
            'nir_masks': nir_masks,
            'pot_centers': pot_centers,
            'rgb_image': str(rgb_path),
            'nir_image': str(nir_path),
            'date': date
        }
        
        mask_file = date_dir / "masks.pkl"
        with open(mask_file, 'wb') as f:
            pickle.dump(mask_data, f)
        print(f"Saved masks to {mask_file}")
        
        # Create visualizations
        print("Creating visualizations...")
        rgb_img = cv2.imread(str(rgb_path))
        nir_img = cv2.imread(str(nir_path))
        
        rgb_vis = segmenter.create_overlap_visualization(rgb_img, rgb_masks, pot_centers)
        nir_vis = segmenter.create_overlap_visualization(nir_img, nir_masks, pot_centers)
        
        cv2.imwrite(str(date_dir / "rgb_segmentation.jpg"), rgb_vis)
        cv2.imwrite(str(date_dir / "nir_segmentation.jpg"), nir_vis)
        print(f"Saved visualizations to {date_dir}")
        
        results[date] = {
            'n_plants': len(pot_centers),
            'rgb_masks': rgb_masks,
            'nir_masks': nir_masks,
            'pot_centers': pot_centers
        }
    
    return results


def main():
    """Main function"""
    Config.ensure_output_dir()
    
    # Load pot centers
    pot_centers_file = Config.OUTPUT_DIR / "pot_centers.json"
    if not pot_centers_file.exists():
        print(f"Error: Pot centers file not found at {pot_centers_file}")
        print("Please run 01_annotate_pots.py first to annotate pot centers")
        return
    
    pot_centers_dict = load_pot_centers(pot_centers_file)
    print(f"Loaded pot centers for {len(pot_centers_dict)} images")
    
    # Segment all images
    output_dir = Config.OUTPUT_DIR / "segmentation_results"
    output_dir.mkdir(exist_ok=True)
    
    results = segment_all_images(pot_centers_dict, output_dir, use_negative_prompts=True)
    
    print("\n" + "="*60)
    print("Segmentation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Processed {len(results)} dates")
    print("="*60)


if __name__ == "__main__":
    main()
