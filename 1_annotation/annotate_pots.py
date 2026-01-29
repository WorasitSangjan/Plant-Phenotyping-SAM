"""
Interactive pot center annotation tool

Usage:
    python 01_annotate_pots.py

Instructions:
    - Left click: Add pot center
    - Right click: Remove nearest pot center
    - Press 'n': Next image
    - Press 'p': Previous image
    - Press 's': Save and continue
    - Press 'q': Save and quit

Author: Worasit Sangjan
Date Created: 29 January 2026
Version: 1.2   
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import sys

sys.path.append(str(Path(__file__).parent))
from config import Config, save_pot_centers, load_pot_centers, get_image_pairs


class PotAnnotator:
    """Interactive tool for annotating pot centers"""
    
    def __init__(self, image_paths: List[Path]):
        self.image_paths = image_paths
        self.current_idx = 0
        self.pot_centers: Dict[str, List[Tuple[int, int]]] = {}
        self.window_name = "Pot Center Annotation"
        self.display_img = None
        
        # Try to load existing annotations
        self.save_path = Config.OUTPUT_DIR / "pot_centers.json"
        if self.save_path.exists():
            print(f"Loading existing annotations from {self.save_path}")
            self.pot_centers = load_pot_centers(self.save_path)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        image_name = self.image_paths[self.current_idx].name
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add pot center
            if image_name not in self.pot_centers:
                self.pot_centers[image_name] = []
            self.pot_centers[image_name].append((x, y))
            self.redraw()
            print(f"Added pot at ({x}, {y}) - Total: {len(self.pot_centers[image_name])}")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove nearest pot center
            if image_name in self.pot_centers and self.pot_centers[image_name]:
                centers = np.array(self.pot_centers[image_name])
                distances = np.sqrt(((centers - [x, y]) ** 2).sum(axis=1))
                nearest_idx = distances.argmin()
                
                if distances[nearest_idx] < 50:  # Within 50 pixels
                    removed = self.pot_centers[image_name].pop(nearest_idx)
                    self.redraw()
                    print(f"Removed pot at {removed} - Total: {len(self.pot_centers[image_name])}")
    
    def redraw(self):
        """Redraw the current image with annotations"""
        image_path = self.image_paths[self.current_idx]
        image_name = image_path.name
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error loading {image_path}")
            return
        
        # Resize if too large
        max_display_size = 1200
        h, w = img.shape[:2]
        if max(h, w) > max_display_size:
            scale = max_display_size / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        self.display_img = img.copy()
        
        # Draw existing pot centers
        if image_name in self.pot_centers:
            for i, (cx, cy) in enumerate(self.pot_centers[image_name]):
                # Draw circle
                cv2.circle(self.display_img, (cx, cy), 8, (0, 255, 0), 2)
                # Draw cross
                cv2.line(self.display_img, (cx-15, cy), (cx+15, cy), (0, 255, 0), 1)
                cv2.line(self.display_img, (cx, cy-15), (cx, cy+15), (0, 255, 0), 1)
                # Draw number
                cv2.putText(self.display_img, str(i+1), (cx+12, cy-12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw info text
        n_pots = len(self.pot_centers.get(image_name, []))
        info_text = f"Image {self.current_idx + 1}/{len(self.image_paths)}: {image_name} | Pots: {n_pots}"
        cv2.putText(self.display_img, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        instructions = "Left: add | Right: remove | n: next | p: prev | s: save | q: quit"
        cv2.putText(self.display_img, instructions, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(self.window_name, self.display_img)
    
    def next_image(self):
        """Move to next image"""
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.redraw()
    
    def prev_image(self):
        """Move to previous image"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.redraw()
    
    def save(self):
        """Save current annotations"""
        Config.ensure_output_dir()
        save_pot_centers(self.pot_centers, self.save_path)
    
    def run(self):
        """Run the annotation tool"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.redraw()
        
        print("\n" + "="*60)
        print("Pot Center Annotation Tool")
        print("="*60)
        print("Controls:")
        print("  Left click: Add pot center")
        print("  Right click: Remove nearest pot center")
        print("  'n': Next image")
        print("  'p': Previous image")
        print("  's': Save annotations")
        print("  'q': Save and quit")
        print("="*60 + "\n")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n'):
                self.next_image()
            elif key == ord('p'):
                self.prev_image()
            elif key == ord('s'):
                self.save()
            elif key == ord('q'):
                self.save()
                break
            elif key == 27:  # ESC
                self.save()
                break
        
        cv2.destroyAllWindows()


def main():
    """Main function"""
    Config.ensure_output_dir()
    
    # Get all RGB images for annotation
    # (We'll use RGB for annotation, then apply to both RGB and NIR)
    rgb_images = sorted(Config.DATA_DIR.glob(Config.RGB_PATTERN))
    
    if not rgb_images:
        print(f"No images found matching pattern {Config.RGB_PATTERN} in {Config.DATA_DIR}")
        return
    
    print(f"Found {len(rgb_images)} RGB images to annotate")
    
    annotator = PotAnnotator(rgb_images)
    annotator.run()
    
    print("\nAnnotation complete!")
    print(f"Annotations saved to: {annotator.save_path}")


if __name__ == "__main__":
    main()
