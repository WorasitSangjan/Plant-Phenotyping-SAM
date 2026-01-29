"""
Copy pot center coordinates from one reference image to all other images

Use this when your imaging setup is fixed (camera and tray do not move),
so pot positions are the same across all images.

Author: Worasit Sangjan
Date Created: 29 January 2026
Version: 1.0   
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import Config, load_pot_centers, save_pot_centers, get_image_pairs


def copy_pot_centers(reference_image: str, verify: bool = True):
    """Copy pot centers from reference image to all other images"""
    Config.ensure_output_dir()
    
    # Load existing pot centers
    pot_centers_file = Config.OUTPUT_DIR / "pot_centers.json"
    if not pot_centers_file.exists():
        print(f"Error: No pot centers file found at {pot_centers_file}")
        print("Please run 01_annotate_pots.py first to annotate at least one image")
        return
    
    pot_centers_dict = load_pot_centers(pot_centers_file)
    
    # Check reference image exists
    if reference_image not in pot_centers_dict:
        print(f"Error: Reference image '{reference_image}' not found in pot centers")
        print(f"Available images: {list(pot_centers_dict.keys())}")
        return
    
    reference_centers = pot_centers_dict[reference_image]
    print(f"Reference image: {reference_image}")
    print(f"Number of pot centers: {len(reference_centers)}")
    print(f"Pot centers: {reference_centers[:3]}... (showing first 3)")
    print()
    
    # Get all images to process
    all_rgb_images = sorted(Config.DATA_DIR.glob(Config.RGB_PATTERN))
    target_images = [img.name for img in all_rgb_images if img.name != reference_image]
    
    if not target_images:
        print("No other images found to copy pot centers to")
        return
    
    print(f"Found {len(target_images)} images to copy pot centers to:")
    for img in target_images:
        status = "exists" if img in pot_centers_dict else "✗ new"
        print(f"  - {img} [{status}]")
    print()
    
    # Verify with user
    if verify:
        print("WARNING: This will overwrite existing pot centers for these images!")
        response = input("Continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled")
            return
        print()
    
    # Copy pot centers
    n_copied = 0
    for target_image in target_images:
        pot_centers_dict[target_image] = reference_centers.copy()
        n_copied += 1
    
    # Save updated pot centers
    save_pot_centers(pot_centers_dict, pot_centers_file)
    
    print(f"Successfully copied pot centers to {n_copied} images")
    print(f"Total images with pot centers: {len(pot_centers_dict)}")
    print()
    print("Next steps:")
    print("  1. (Optional) Verify pot positions are correct:")
    print("     python 01_annotate_pots.py  # Navigate through images to check")
    print("  2. Run segmentation:")
    print("     python 02_segment_plants.py")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Copy pot centers from reference image to all other images'
    )
    parser.add_argument(
        '--reference', '-r',
        type=str,
        required=True,
        help='Reference image name (e.g., rgb_061325.jpg)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    copy_pot_centers(args.reference, verify=not args.no_verify)


if __name__ == "__main__":
    main()
