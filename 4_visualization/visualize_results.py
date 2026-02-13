"""
Visualization helpers for phenotyping results

Creates plots and summaries from extracted trait data

Author: Worasit Sangjan
Date Created: 9 February 2026
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import Config


def plot_growth_over_time(df: pd.DataFrame, output_dir: Path):
    """Plot leaf area growth over time"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # RGB leaf area
    ax = axes[0]
    for plant_id in df['plant_id'].unique()[:20]:  # Plot first 20 plants
        plant_data = df[df['plant_id'] == plant_id].sort_values('date')
        ax.plot(plant_data['date'], plant_data['rgb_leaf_area_pixels'], 
                marker='o', alpha=0.6, linewidth=1)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('RGB Leaf Area (pixels)')
    ax.set_title('Individual Plant Growth (RGB)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Average growth
    ax = axes[1]
    avg_growth = df.groupby('date')['rgb_leaf_area_pixels'].agg(['mean', 'std'])
    ax.errorbar(avg_growth.index, avg_growth['mean'], yerr=avg_growth['std'],
                marker='o', capsize=5, linewidth=2, markersize=8)
    ax.fill_between(avg_growth.index, 
                     avg_growth['mean'] - avg_growth['std'],
                     avg_growth['mean'] + avg_growth['std'],
                     alpha=0.2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('RGB Leaf Area (pixels)')
    ax.set_title('Average Growth Across All Plants')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'growth_over_time.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'growth_over_time.png'}")
    plt.close()


def plot_vegetation_indices(df: pd.DataFrame, output_dir: Path):
    """Plot vegetation indices over time"""
    
    vis = ['ExG', 'VARI', 'GLI', 'NGRDI']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, vi in enumerate(vis):
        if vi not in df.columns:
            continue
        
        ax = axes[i]
        avg_vi = df.groupby('date')[vi].agg(['mean', 'std'])
        
        ax.errorbar(avg_vi.index, avg_vi['mean'], yerr=avg_vi['std'],
                   marker='o', capsize=5, linewidth=2, markersize=8)
        ax.fill_between(avg_vi.index,
                       avg_vi['mean'] - avg_vi['std'],
                       avg_vi['mean'] + avg_vi['std'],
                       alpha=0.2)
        
        ax.set_xlabel('Date')
        ax.set_ylabel(vi)
        ax.set_title(f'{vi} Over Time')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'vegetation_indices.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'vegetation_indices.png'}")
    plt.close()


def plot_spatial_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot spatial distribution of plants"""
    
    # Use last date
    last_date = df['date'].max()
    df_last = df[df['date'] == last_date]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Leaf area by position
    ax = axes[0]
    scatter = ax.scatter(df_last['pot_center_x'], df_last['pot_center_y'],
                        c=df_last['rgb_leaf_area_pixels'], s=200,
                        cmap='viridis', edgecolors='black', linewidth=1)
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(f'Leaf Area by Position ({last_date})')
    ax.invert_yaxis()  # Image coordinates
    plt.colorbar(scatter, ax=ax, label='RGB Leaf Area (pixels)')
    
    # Spillover ratio by position
    ax = axes[1]
    scatter = ax.scatter(df_last['pot_center_x'], df_last['pot_center_y'],
                        c=df_last['spillover_ratio'] * 100, s=200,
                        cmap='RdYlGn', vmin=0, vmax=100,
                        edgecolors='black', linewidth=1)
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(f'Spillover Ratio by Position ({last_date})')
    ax.invert_yaxis()
    plt.colorbar(scatter, ax=ax, label='Spillover (%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'spatial_distribution.png'}")
    plt.close()


def plot_color_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot color statistics over time"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ['mean_R', 'mean_G', 'mean_B']
    color_names = ['Red', 'Green', 'Blue']
    color_rgb = ['red', 'green', 'blue']
    
    for i, (col, name, rgb) in enumerate(zip(colors, color_names, color_rgb)):
        ax = axes[i]
        avg_color = df.groupby('date')[col].agg(['mean', 'std'])
        
        ax.errorbar(avg_color.index, avg_color['mean'], yerr=avg_color['std'],
                   marker='o', capsize=5, linewidth=2, markersize=8,
                   color=rgb)
        ax.fill_between(avg_color.index,
                       avg_color['mean'] - avg_color['std'],
                       avg_color['mean'] + avg_color['std'],
                       alpha=0.2, color=rgb)
        
        ax.set_xlabel('Date')
        ax.set_ylabel(f'Mean {name} Channel')
        ax.set_title(f'{name} Channel Over Time')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 255])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'color_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'color_distribution.png'}")
    plt.close()


def plot_spillover_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze leaf spillover patterns"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Spillover ratio over time
    ax = axes[0]
    avg_spillover = df.groupby('date')['spillover_ratio'].agg(['mean', 'std'])
    avg_spillover *= 100  # Convert to percentage
    
    ax.errorbar(avg_spillover.index, avg_spillover['mean'], 
               yerr=avg_spillover['std'],
               marker='o', capsize=5, linewidth=2, markersize=8)
    ax.fill_between(avg_spillover.index,
                   avg_spillover['mean'] - avg_spillover['std'],
                   avg_spillover['mean'] + avg_spillover['std'],
                   alpha=0.2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Spillover Ratio (%)')
    ax.set_title('Average Leaf Spillover Over Time')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Relationship between area and spillover
    ax = axes[1]
    for date in df['date'].unique():
        df_date = df[df['date'] == date]
        ax.scatter(df_date['rgb_leaf_area_pixels'], 
                  df_date['spillover_ratio'] * 100,
                  alpha=0.6, s=30, label=date)
    
    ax.set_xlabel('RGB Leaf Area (pixels)')
    ax.set_ylabel('Spillover Ratio (%)')
    ax.set_title('Leaf Area vs Spillover')
    ax.legend(title='Date', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spillover_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'spillover_analysis.png'}")
    plt.close()


def generate_summary_report(df: pd.DataFrame, output_dir: Path):
    """Generate text summary report"""
    
    report = []
    report.append("="*60)
    report.append("PHENOTYPING SUMMARY REPORT")
    report.append("="*60)
    report.append("")
    
    # Overall statistics
    report.append("Overall Statistics:")
    report.append(f"  Total plants measured: {len(df)}")
    report.append(f"  Number of dates: {df['date'].nunique()}")
    report.append(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    report.append("")
    
    # Growth statistics
    report.append("Growth Statistics (RGB Leaf Area):")
    report.append(f"  Mean: {df['rgb_leaf_area_pixels'].mean():.1f} pixels")
    report.append(f"  Std: {df['rgb_leaf_area_pixels'].std():.1f} pixels")
    report.append(f"  Min: {df['rgb_leaf_area_pixels'].min():.1f} pixels")
    report.append(f"  Max: {df['rgb_leaf_area_pixels'].max():.1f} pixels")
    report.append("")
    
    # Spillover statistics
    report.append("Spillover Statistics:")
    report.append(f"  Mean spillover: {df['spillover_ratio'].mean()*100:.1f}%")
    report.append(f"  Plants with >50% spillover: {(df['spillover_ratio'] > 0.5).sum()}")
    report.append("")
    
    # Vegetation indices
    report.append("Vegetation Indices (Mean):")
    for vi in ['ExG', 'VARI', 'GLI', 'NGRDI']:
        if vi in df.columns:
            report.append(f"  {vi}: {df[vi].mean():.3f}")
    report.append("")
    
    # Per-date summary
    report.append("Per-Date Summary:")
    for date in sorted(df['date'].unique()):
        df_date = df[df['date'] == date]
        report.append(f"\n  {date}:")
        report.append(f"    Plants: {len(df_date)}")
        report.append(f"    Mean area: {df_date['rgb_leaf_area_pixels'].mean():.1f} pixels")
        report.append(f"    Mean spillover: {df_date['spillover_ratio'].mean()*100:.1f}%")
    
    report.append("")
    report.append("="*60)
    
    # Save report
    report_text = "\n".join(report)
    report_file = output_dir / "summary_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nSaved: {report_file}")


def main():
    """Main visualization function"""
    Config.ensure_output_dir()
    
    # Load trait data
    trait_file = Config.OUTPUT_DIR / "plant_traits.csv"
    if not trait_file.exists():
        print(f"Error: Trait file not found at {trait_file}")
        print("Please run 03_extract_traits.py first")
        return
    
    print(f"Loading trait data from {trait_file}...")
    df = pd.read_csv(trait_file)
    print(f"Loaded data for {len(df)} plants across {df['date'].nunique()} dates")
    
    # Create visualization directory
    vis_dir = Config.OUTPUT_DIR / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    print("\nGenerating visualizations...")
    
    # Generate all plots
    plot_growth_over_time(df, vis_dir)
    plot_vegetation_indices(df, vis_dir)
    plot_spatial_distribution(df, vis_dir)
    plot_color_distribution(df, vis_dir)
    plot_spillover_analysis(df, vis_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df, vis_dir)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"All plots saved to: {vis_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
