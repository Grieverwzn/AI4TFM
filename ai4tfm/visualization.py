"""
Visualization tools for speed heatmaps and traffic analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from typing import Optional, Tuple, List, Dict


class SpeedHeatmap:
    """
    Create speed heatmaps to visualize observed vs. modeled speeds.
    
    This class provides methods to create various types of heatmaps
    for traffic speed analysis.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 6),
        cmap: str = 'RdYlGn'
    ):
        """
        Initialize SpeedHeatmap visualizer.
        
        Args:
            figsize: Figure size (width, height) in inches
            cmap: Colormap name for heatmap
        """
        self.figsize = figsize
        self.cmap = cmap
    
    def plot_single_heatmap(
        self,
        speeds: np.ndarray,
        time_labels: List[str],
        location_labels: List[str],
        title: str = "Speed Heatmap",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        xlabel: str = "Time",
        ylabel: str = "Location"
    ) -> Figure:
        """
        Create a single speed heatmap.
        
        Args:
            speeds: 2D array of speeds (locations x time)
            time_labels: List of time labels
            location_labels: List of location labels
            title: Plot title
            vmin: Minimum value for colorbar
            vmax: Maximum value for colorbar
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(
            speeds,
            cmap=self.cmap,
            aspect='auto',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(time_labels)))
        ax.set_yticks(np.arange(len(location_labels)))
        ax.set_xticklabels(time_labels, rotation=45, ha='right')
        ax.set_yticklabels(location_labels)
        
        # Labels and title
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Speed (mph)', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_comparison_heatmap(
        self,
        observed_speeds: np.ndarray,
        modeled_speeds: np.ndarray,
        time_labels: List[str],
        location_labels: List[str],
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> Figure:
        """
        Create side-by-side comparison of observed vs. modeled speeds.
        
        Args:
            observed_speeds: 2D array of observed speeds
            modeled_speeds: 2D array of modeled speeds
            time_labels: List of time labels
            location_labels: List of location labels
            vmin: Minimum value for colorbar
            vmax: Maximum value for colorbar
            
        Returns:
            Matplotlib Figure object
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Observed speeds
        im1 = ax1.imshow(
            observed_speeds,
            cmap=self.cmap,
            aspect='auto',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )
        ax1.set_title('Observed Speeds', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Location', fontsize=12)
        ax1.set_xticks(np.arange(len(time_labels)))
        ax1.set_yticks(np.arange(len(location_labels)))
        ax1.set_xticklabels(time_labels, rotation=45, ha='right')
        ax1.set_yticklabels(location_labels)
        
        # Modeled speeds
        im2 = ax2.imshow(
            modeled_speeds,
            cmap=self.cmap,
            aspect='auto',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )
        ax2.set_title('Modeled Speeds', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Location', fontsize=12)
        ax2.set_xticks(np.arange(len(time_labels)))
        ax2.set_yticks(np.arange(len(location_labels)))
        ax2.set_xticklabels(time_labels, rotation=45, ha='right')
        ax2.set_yticklabels(location_labels)
        
        # Difference (error)
        difference = observed_speeds - modeled_speeds
        im3 = ax3.imshow(
            difference,
            cmap='RdBu_r',
            aspect='auto',
            interpolation='nearest',
            vmin=-np.max(np.abs(difference)),
            vmax=np.max(np.abs(difference))
        )
        ax3.set_title('Difference (Observed - Modeled)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time', fontsize=12)
        ax3.set_ylabel('Location', fontsize=12)
        ax3.set_xticks(np.arange(len(time_labels)))
        ax3.set_yticks(np.arange(len(location_labels)))
        ax3.set_xticklabels(time_labels, rotation=45, ha='right')
        ax3.set_yticklabels(location_labels)
        
        # Colorbars
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Speed (mph)', fontsize=10)
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Speed (mph)', fontsize=10)
        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('Speed Difference (mph)', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_time_series(
        self,
        time_steps: np.ndarray,
        observed_speeds: Optional[np.ndarray] = None,
        modeled_speeds: Optional[np.ndarray] = None,
        title: str = "Speed Time Series",
        location: str = "Location"
    ) -> Figure:
        """
        Plot time series of speeds for a single location.
        
        Args:
            time_steps: Array of time values
            observed_speeds: Array of observed speeds (optional)
            modeled_speeds: Array of modeled speeds (optional)
            title: Plot title
            location: Location identifier
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if observed_speeds is not None:
            ax.plot(time_steps, observed_speeds, 'o-', 
                   label='Observed', linewidth=2, markersize=4)
        
        if modeled_speeds is not None:
            ax.plot(time_steps, modeled_speeds, 's--', 
                   label='Modeled', linewidth=2, markersize=4)
        
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Speed (mph)', fontsize=12)
        ax.set_title(f'{title} - {location}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_congestion_profile(
        self,
        time_steps: np.ndarray,
        speeds: np.ndarray,
        congestion_threshold: float = 45.0,
        free_flow_speed: float = 60.0
    ) -> Figure:
        """
        Plot congestion profile with color-coded regions.
        
        Args:
            time_steps: Array of time values
            speeds: Array of speeds
            congestion_threshold: Speed threshold for congestion
            free_flow_speed: Free flow speed reference
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot speed profile
        ax.plot(time_steps, speeds, 'b-', linewidth=2, label='Speed')
        
        # Add threshold lines
        ax.axhline(y=congestion_threshold, color='r', linestyle='--', 
                  linewidth=1.5, label=f'Congestion Threshold ({congestion_threshold} mph)')
        ax.axhline(y=free_flow_speed, color='g', linestyle='--', 
                  linewidth=1.5, label=f'Free Flow ({free_flow_speed} mph)')
        
        # Shade congested regions
        congested = speeds < congestion_threshold
        if np.any(congested):
            ax.fill_between(time_steps, 0, speeds, where=congested, 
                           alpha=0.3, color='red', label='Congested Period')
        
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Speed (mph)', fontsize=12)
        ax.set_title('Congestion Profile', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: Figure, filename: str, dpi: int = 300):
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib Figure object
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
