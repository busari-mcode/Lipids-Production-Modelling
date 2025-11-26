import numpy as np
import matplotlib.pyplot as plt
import math

# Removed: import seaborn as sns

# Set a clean Matplotlib style manually (simulating the Seaborn 'whitegrid' look)
plt.style.use('default') # Start with a clean default
plt.rcParams.update({
    'axes.grid': True,              # Add grid lines
    'grid.linestyle': '--',         # Use dashed lines for the grid
    'axes.facecolor': 'white',      # White plot background
    'figure.facecolor': 'white',    # White figure background
    'lines.linewidth': 2.2,         # Set default line width (matching lw in the plot call)
    'font.size': 14,                # Increase font size (like seaborn 'talk' context)
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

# Poisson PMF function
def poisson_pmf(n, mu):
    n = np.asarray(n, dtype=float)
    # Using the log-gamma function for numerical stability
    return np.exp(n*np.log(mu) - mu - np.vectorize(math.lgamma)(n+1))

# Parameters
n = np.arange(0, 80)
mu_values = [10, 20, 40]

# Plot
plt.figure(figsize=(8, 6))
for mu in mu_values:
    plt.plot(n, poisson_pmf(n, mu), label=f"$\\mu$={mu}") # lw=2.2 is now set by rcParams

# plt.title("Case 1: Narrow Enzyme Distribution → Poisson($\\mu$)")
plt.xlabel("Product count ($n_p$)")
plt.ylabel("Probability [P($n_p$)]")
plt.legend()
plt.tight_layout()
plt.show()