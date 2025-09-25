import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

# Set the style
sns.set_context("talk")
sns.set_style("white")

# Generate in-distribution clusters
np.random.seed(42)
blue = np.random.randn(20, 2) * 0.4 + np.array([1, 1])
green = np.random.randn(20, 2) * 0.4 + np.array([3, 1])
ood = np.random.randn(10, 2) * 0.3 + np.array([2, -1])

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# ========== Left: Discriminator ==========
ax = axes[0]
ax.scatter(*blue.T, color="royalblue", label="In-distribution (Class 1)")
ax.scatter(*green.T, color="mediumseagreen", label="In-distribution (Class 2)")
ax.scatter(*ood.T, color="crimson", label="Out-of-distribution")

# Fake decision boundary
x = np.linspace(0, 4, 100)
y = 2 - 0.5 * x
ax.plot(x, y, 'k--', label="Decision Boundary")

ax.set_title("Discriminator")
ax.legend(loc="lower right")
ax.axis("off")

# ========== Right: Density Estimator ==========
ax = axes[1]
ax.scatter(*blue.T, color="royalblue")
ax.scatter(*green.T, color="mediumseagreen")
ax.scatter(*ood.T, color="crimson")

# Add contours or ellipses for in-dist regions
for center, color in zip([[1,1], [3,1]], ['royalblue', 'mediumseagreen']):
    ellipse = Ellipse(center, width=2, height=1.2, alpha=0.1, color=color)
    ax.add_patch(ellipse)

# Add ellipse around OoD region
ellipse = Ellipse([2, -1], width=1.2, height=0.8, alpha=0.1, color='crimson')
ax.add_patch(ellipse)

ax.set_title("Density Estimator")
ax.axis("off")

plt.suptitle("In-Distribution vs Out-of-Distribution")
plt.tight_layout()
plt.show()
