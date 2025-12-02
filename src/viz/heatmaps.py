import matplotlib.pyplot as plt
import numpy as np

def plot_distortion(Y, distortion_values, title="Distortion Heatmap", log_scale=True):
    """
    Plots the 2D embedding colored by a distortion metric.
    """
    plt.figure(figsize=(10, 8))
    
    values = distortion_values.copy()
    
    if log_scale:
        # Log scale helps visualize orders of magnitude differences (common in Condition Number)
        values = np.log1p(values)
        title += " (Log Scale)"
        
    sc = plt.scatter(Y[:, 0], Y[:, 1], c=values, cmap='magma', s=5, alpha=0.8)
    plt.colorbar(sc, label="Distortion Magnitude")
    plt.title(title)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Save directly
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    # plt.show() # Uncomment if running locally with UI
