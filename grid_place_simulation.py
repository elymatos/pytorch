
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation parameters
arena_size = 2.0  # meters
duration = 20     # seconds
dt = 0.1
steps = int(duration / dt)

# Random walk for navigation
np.random.seed(42)
angles = np.cumsum(np.random.randn(steps) * 0.1)
x = np.cumsum(np.cos(angles) * dt * 0.1)
y = np.cumsum(np.sin(angles) * dt * 0.1)
x = np.mod(x + arena_size, arena_size)
y = np.mod(y + arena_size, arena_size)

# Grid cell activation function
def grid_cell_activity(x_grid, y_grid, spacing, phase, orientation):
    lambda_ = spacing
    theta = orientation
    k = (4 * np.pi) / (np.sqrt(3) * lambda_)

    u = np.array([np.cos(theta), np.sin(theta)])
    v = np.array([np.cos(theta + np.pi / 3), np.sin(theta + np.pi / 3)])
    w = np.array([np.cos(theta + 2 * np.pi / 3), np.sin(theta + 2 * np.pi / 3)])

    px = x_grid - phase[0]
    py = y_grid - phase[1]

    activation = (
        np.cos(k * (px * u[0] + py * u[1])) +
        np.cos(k * (px * v[0] + py * v[1])) +
        np.cos(k * (px * w[0] + py * w[1]))
    ) / 3.0
    return np.maximum(activation, 0)

# Grid cell definitions
grid_params = [
    {"spacing": 0.4, "phase": np.array([0.0, 0.0]), "orientation": 0.0},
    {"spacing": 0.6, "phase": np.array([0.1, 0.1]), "orientation": 0.1},
    {"spacing": 0.8, "phase": np.array([0.2, 0.2]), "orientation": -0.2}
]

# Plot setup
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
trail_line, = axs[0].plot([], [], 'k-', alpha=0.3)
agent_dot, = axs[0].plot([], [], 'ro')
axs[0].set_xlim(0, arena_size)
axs[0].set_ylim(0, arena_size)
axs[0].set_title("Agent Path and Place Cell Activation")

heat = axs[1].imshow(np.zeros((50, 50)), extent=(0, arena_size, 0, arena_size),
                     origin='lower', cmap='hot', interpolation='bilinear', vmin=0, vmax=1)
axs[1].set_title("Combined Grid → Place Cell Activity")

# Heatmap meshgrid
res = 50
x_edges = np.linspace(0, arena_size, res)
y_edges = np.linspace(0, arena_size, res)
xx, yy = np.meshgrid(x_edges, y_edges)

# Animation update
def update(frame):
    xi, yi = x[frame], y[frame]
    trail_line.set_data(x[:frame], y[:frame])
    agent_dot.set_data([xi], [yi])  # FIXED: wrap in list to avoid RuntimeError

    act = np.ones_like(xx)
    for gp in grid_params:
        gc = grid_cell_activity(xx, yy, gp["spacing"], gp["phase"], gp["orientation"])
        act *= gc

    heat.set_array(act)
    return trail_line, agent_dot, heat

ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=True)
plt.tight_layout()
plt.show()
