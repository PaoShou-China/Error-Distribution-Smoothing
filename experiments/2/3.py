import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set global font sizes for plots
plt.rcParams.update({
    'font.size': 14,        # Default font size
    'axes.titlesize': 16,   # Title font size
    'axes.labelsize': 14,   # Axis label font size
    'legend.fontsize': 12,  # Legend font size
    'xtick.labelsize': 12,  # X-axis tick label font size
    'ytick.labelsize': 12,  # Y-axis tick label font size
})

# True Lorenz system derivative function
def lorenz_system(t, state, sigma=10, rho=28, beta=8 / 3):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Identified model 1 derivative function
def sindy_model_1(t, state):
    x, y, z = state
    dx_dt = -9.969 * x + 9.980 * y
    dy_dt = 27.969 * x - 0.980 * y - 0.999 * x * z
    dz_dt = -2.665 * z + 1.000 * x * y
    return [dx_dt, dy_dt, dz_dt]

# Identified model 3 derivative function
def sindy_model_3(t, state):
    x, y, z = state
    dx_dt = -9.950 * x + 9.968 * y - 0.001 * x * z
    dy_dt = 27.949 * x - 0.967 * y - 0.999 * x * z
    dz_dt = -2.665 * z + 1.000 * x * y
    return [dx_dt, dy_dt, dz_dt]

# Identified model 4 derivative function
def sindy_model_4(t, state):
    x, y, z = state
    dx_dt = -9.998 * x + 9.999 * y
    dy_dt = 27.994 * x - 0.997 * y - 1.000 * x * z
    dz_dt = -2.666 * z + 1.000 * x * y
    return [dx_dt, dy_dt, dz_dt]

# Identified model 2 derivative function
def sindy_model_2(t, state):
    x, y, z = state
    dx_dt = -9.981 * x + 9.989 * y
    dy_dt = 27.899 * x - 0.971 * y - 0.007 * z - 0.002 * x * y - 0.998 * x * z + 0.001 * y * y
    dz_dt = -4.280 + 0.090 * x - 0.028 * y - 2.074 * z + 0.079 * x * x + 0.933 * x * y - 0.003 * x * z + 0.012 * y * y - 0.018 * z * z
    return [dx_dt, dy_dt, dz_dt]

# Define initial condition and time span
initial_state = [1.0, 1.0, 1.0]  # Initial state
t_span = (0, 15)  # Time span
t_eval = np.linspace(t_span[0], t_span[1], 10000)  # Time points

# Solve the true Lorenz system
sol_true = solve_ivp(lorenz_system, t_span, initial_state, t_eval=t_eval)

# Solve the four identified SINDy models
sindy_models = [sindy_model_1, sindy_model_2, sindy_model_3, sindy_model_4]
model_names = ['Model via $\mathcal{D}$', 'Model via $\mathcal{D}_{S}$', 'Model via $\mathcal{D}_{M}$', 'Model via $\mathcal{D}_{R}$']

solutions = {}
for model, name in zip(sindy_models, model_names):
    sol_sindy = solve_ivp(model, t_span, initial_state, t_eval=t_eval)
    solutions[name] = sol_sindy

# Create 2x2 subplot layout
fig = plt.figure(figsize=(15, 12))
axes = []
for i in range(4):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    axes.append(ax)

# Set axis ranges
x_min, x_max = np.min(sol_true.y[0]), np.max(sol_true.y[0])
y_min, y_max = np.min(sol_true.y[1]), np.max(sol_true.y[1])
z_min, z_max = np.min(sol_true.y[2]), np.max(sol_true.y[2])

padding = -0.01  # Add a small margin
x_lim = [x_min - padding * (x_max - x_min), x_max + padding * (x_max - x_min)]
y_lim = [y_min - padding * (y_max - y_min), y_max + padding * (y_max - y_min)]
z_lim = [z_min - padding * (z_max - z_min), z_max + padding * (z_max - z_min)]

# Plot true system and each identified model in each subplot
colors = ['purple','blue', 'green',  'red']
for i, (name, sol) in enumerate(solutions.items()):
    ax = axes[i]
    # Plot true Lorenz system
    ax.plot(sol_true.y[0], sol_true.y[1], sol_true.y[2], 
            label='True Lorenz System', lw=2, color='black')
    # Plot identified model
    ax.plot(sol.y[0], sol.y[1], sol.y[2], '--', 
            label=f'{name}', lw=2, alpha=1.0, color=colors[i])
    ax.set_title(f'True and SINDy {name}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # Set axis ranges
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    # Adjust view angle
    ax.view_init(elev=20, azim=-45)
    # Add legend
    ax.legend(loc='upper right')

# Adjust subplot spacing
plt.tight_layout()
plt.savefig("results/figures/2/3.png", dpi=300)
# plt.show()