import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x, y)
def f(x, y):
    return 1 / (0.33 + x ** 2 + y ** 2)

# Create mesh grid data
x = np.linspace(-3, 3, 400)  # Reduce the number of points for better performance
y = np.linspace(-3, 3, 400)  # Reduce the number of points for better performance
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

title_font_size = 16
label_font_size = 12
tick_label_size = 12

# Draw filled contour plot
plt.figure(figsize=(5, 4))
contour = plt.contourf(X, Y, Z, levels=30, cmap='plasma')
plt.colorbar(contour)

# Set axis labels and font sizes
plt.xlabel('x', fontsize=label_font_size)
plt.ylabel('y', fontsize=label_font_size)
plt.xticks(fontsize=tick_label_size)
plt.yticks(fontsize=tick_label_size)

plt.tight_layout()
plt.show()