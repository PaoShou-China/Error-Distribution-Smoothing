import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x, y)
def f(x, y):
    """
    Compute the value of the function f(x, y) = 1 / (0.33 + x^2 + y^2).
    """
    return 1 / (0.33 + x ** 2 + y ** 2)

# Create grid data
x = np.linspace(-3, 3, 400)  # Reduce the number of points to improve performance
y = np.linspace(-3, 3, 400)  # Reduce the number of points to improve performance
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Set font sizes
title_font_size = 16
label_font_size = 12
tick_label_size = 12

# Plot the contour filled plot (heatmap)
plt.figure(figsize=(5, 4))
contour = plt.contourf(X, Y, Z, levels=30, cmap='plasma')
plt.colorbar(contour)

# Set title and axis labels with specified font sizes
# plt.title('Contour Filled Plot of $f(x, y) = (0.33 + x^2 + y^2)^{-1}$',fontsize=title_font_size)
plt.xlabel('x', fontsize=label_font_size)
plt.ylabel('y', fontsize=label_font_size)

# Set tick label font sizes
plt.xticks(fontsize=tick_label_size)
plt.yticks(fontsize=tick_label_size)

plt.tight_layout()
plt.show()