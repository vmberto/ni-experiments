"""
Overfitting Illustration
========================
Generates a clean illustration comparing overfitting vs non-overfitting behavior.
Uses synthetic curves for clarity.

Output: overfitting_illustration.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Use Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'


def generate_overfitting_curves(epochs=100):
    """Generate synthetic curves that show overfitting behavior."""
    x = np.linspace(0, epochs, epochs)
    
    # Training loss: keeps decreasing
    train_loss = 0.8 * np.exp(-0.08 * x) + 0.02
    
    # Validation loss: decreases then increases (overfitting)
    val_loss = 0.9 * np.exp(-0.05 * x) + 0.15
    # Add the characteristic upturn after some point
    upturn_start = 25
    upturn = np.maximum(0, (x - upturn_start) * 0.008)
    val_loss = val_loss + upturn
    
    return x, train_loss, val_loss


def generate_non_overfitting_curves(epochs=100):
    """Generate synthetic curves that show proper generalization (no overfitting)."""
    x = np.linspace(0, epochs, epochs)
    
    # Training loss: decreases and plateaus
    train_loss = 0.8 * np.exp(-0.06 * x) + 0.05
    
    # Validation loss: follows training closely, slightly higher
    val_loss = 0.85 * np.exp(-0.055 * x) + 0.08
    
    return x, train_loss, val_loss


def plot_overfitting_illustration(save_path='overfitting_illustration.pdf'):
    """Generate the overfitting vs non-overfitting illustration."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Colors matching the reference
    train_color = '#3A91DA'  # Cornflower blue
    val_color = '#75DC5E'    # Flame orange
    
    # =========================================================================
    # (a) Overfitting
    # =========================================================================
    x, train_loss, val_loss = generate_overfitting_curves()
    
    line_train, = ax2.plot(x, train_loss, color=train_color, linewidth=3.5)
    line_val, = ax2.plot(x, val_loss, color=val_color, linewidth=3.5)
    
    # Find and mark the overfitting threshold
    min_val_idx = np.argmin(val_loss)
    threshold_x = x[min_val_idx]
    
    # Add vertical dashed line at overfitting point
    ax2.axvline(x=threshold_x, color='gray', linestyle='--', linewidth=2.5, alpha=0.7)
    
    # Add annotation for the threshold
    ax2.annotate('Early stopping\npoint',
                xy=(threshold_x, val_loss[min_val_idx]), 
                xytext=(threshold_x + 15, val_loss[min_val_idx] + 0.15),
                fontsize=18,
                arrowprops=dict(arrowstyle='->', color='gray', lw=2),
                ha='left')
    
    # Shade the overfitting region
    ax2.axvspan(threshold_x, x[-1], alpha=0.1, color='red')
    ax2.text(threshold_x + (x[-1] - threshold_x)/2, 0.83, 'Overfitting\nRegion',
             ha='center', va='center', fontsize=24, color='#8B0000', style='italic')
    
    ax2.set_xlim(0, x[-1])
    ax2.set_ylim(0, 0.95)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)
    
    # =========================================================================
    # (b) Non-overfitting
    # =========================================================================
    x, train_loss, val_loss = generate_non_overfitting_curves()
    
    ax1.plot(x, train_loss, color=train_color, linewidth=3.5)
    ax1.plot(x, val_loss, color=val_color, linewidth=3.5)
    
    ax1.set_xlim(0, x[-1])
    ax1.set_ylim(0, 0.95)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    
    # =========================================================================
    # Subfigure labels
    # =========================================================================
    ax2.set_xlabel('(b) Overfitting', fontsize=32, fontweight='normal', labelpad=18)
    ax1.set_xlabel('(a) Non-Overfitting', fontsize=32, fontweight='normal', labelpad=18)
    
    # =========================================================================
    # Unified legend at the bottom
    # =========================================================================
    fig.legend([line_train, line_val], ['Training loss', 'Validation loss'],
               loc='lower center', ncol=2, fontsize=32, frameon=False,
               bbox_to_anchor=(0.5, -0.18))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)  # Make room for legend
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nSaved: {save_path}")


if __name__ == '__main__':
    plot_overfitting_illustration()
