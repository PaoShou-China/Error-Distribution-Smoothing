import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Set global font sizes
plt.rcParams.update({
    'font.size': 14,  # Default font size
    'axes.titlesize': 16,  # Title font size
    'axes.labelsize': 14,  # Axis label font size
    'legend.fontsize': 14,  # Legend font size
    'xtick.labelsize': 14,  # X-axis tick label font size
    'ytick.labelsize': 14,  # Y-axis tick label font size
})


# Define the derivative function for the true Lorenz system
def lorenz_system(t, state, sigma=10, rho=28, beta=8 / 3):
    """
    Define the derivative function for the true Lorenz system.
    """
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]


def sindy_model_1(t, state):
    """
    Define the derivative function for the first identified SINDy model.
    """
    x, y, z = state
    dx_dt = -9.969 * x + 9.980 * y
    dy_dt = 27.969 * x - 0.980 * y - 0.999 * x * z
    dz_dt = -2.665 * z + 1.000 * x * y
    return [dx_dt, dy_dt, dz_dt]


def sindy_model_2(t, state):
    """
    Define the derivative function for the second identified SINDy model.
    """
    x, y, z = state
    dx_dt = -9.950 * x + 9.968 * y - 0.001 * x * z
    dy_dt = 27.949 * x - 0.967 * y - 0.999 * x * z
    dz_dt = -2.665 * z + 1.000 * x * y
    return [dx_dt, dy_dt, dz_dt]


def sindy_model_3(t, state):
    """
    Define the derivative function for the third identified SINDy model.
    """
    x, y, z = state
    dx_dt = -9.991 * x + 9.995 * y
    dy_dt = 27.991 * x - 0.995 * y - 1.000 * x * z
    dz_dt = -2.666 * z + 1.000 * x * y
    return [dx_dt, dy_dt, dz_dt]


# Define initial conditions and time span
initial_state = [1.0, 1.0, 1.0]  # Initial state
t_span = (0, 15)  # Time span
t_eval = np.linspace(t_span[0], t_span[1], 10000)  # Time points

# Solve the true Lorenz system using solve_ivp
sol_true = solve_ivp(lorenz_system, t_span, initial_state, t_eval=t_eval)

# Solve the identified SINDy models using solve_ivp
sindy_models = [sindy_model_1, sindy_model_2, sindy_model_3]
model_names = ['Model via $\mathcal{D}$', 'Model via $\mathcal{D}_{M}$', 'Model via $\mathcal{D}_{R}$']

solutions = {}
for model, name in zip(sindy_models, model_names):
    sol_sindy = solve_ivp(model, t_span, initial_state, t_eval=t_eval)
    solutions[name] = sol_sindy

# Plot 3D trajectories
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory of the true Lorenz system for enhanced visual effect
ax.plot(sol_true.y[0], sol_true.y[1], sol_true.y[2], label='True Lorenz System', lw=2, color='black')

# Plot the trajectories of each SINDy model with transparency
colors = ['blue', 'green', 'red']
for i, (name, sol) in enumerate(solutions.items()):
    ax.plot(sol.y[0], sol.y[1], sol.y[2], '--', label=f'{name}', lw=2, alpha=1.0, color=colors[i])

# Set chart title and axis labels
ax.set_title('3D Trajectories of True and SINDy Models')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# Set axis limits to zoom in the image
x_min, x_max = np.min(sol_true.y[0]), np.max(sol_true.y[0])
y_min, y_max = np.min(sol_true.y[1]), np.max(sol_true.y[1])
z_min, z_max = np.min(sol_true.y[2]), np.max(sol_true.y[2])

padding = -0.01  # Add some extra space
ax.set_xlim([x_min - padding * (x_max - x_min), x_max + padding * (x_max - x_min)])
ax.set_ylim([y_min - padding * (y_max - y_min), y_max + padding * (y_max - y_min)])
ax.set_zlim([z_min - padding * (z_max - z_min), z_max + padding * (z_max - z_min)])

# Adjust the view to better display the trajectories
ax.view_init(elev=20, azim=-45)

# Add a horizontally arranged legend
ax.legend(loc='upper center', ncol=2)

plt.tight_layout()
plt.subplots_adjust(bottom=0.0)  # Adjust the bottom margin to fit the legend
plt.show()
