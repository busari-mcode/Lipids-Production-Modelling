import numpy as np
import matplotlib.pyplot as plt
# Removed: import seaborn as sns
import math

# Set a clean Matplotlib style manually (simulating the Seaborn 'whitegrid' look)
plt.style.use('default') # Start with a clean default
plt.rcParams.update({
    'axes.grid': True,              # Add grid lines
    'grid.linestyle': '--',         # Use dashed lines for the grid
    'axes.facecolor': 'white',      # White plot background
    'figure.facecolor': 'white',    # White figure background
    'lines.linewidth': 2.2,         # Set default line width
    'font.size': 14,                # Increase font size
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

# Negative Binomial PMF function
def nb_pmf(n, mu, r):
    # p is the probability of success in the underlying geometric process
    p = r / (r + mu)
    n = np.asarray(n, dtype=float)
    
    # Calculate the log of the binomial coefficient (n + r - 1 choose n)
    log_coef = (np.vectorize(math.lgamma)(n + r)
                 - np.vectorize(math.lgamma)(r)
                 - np.vectorize(math.lgamma)(n + 1))
    
    # The full PMF: P(n) = (n + r - 1 choose n) * p^r * (1 - p)^n
    return np.exp(log_coef + r*np.log(p) + n*np.log(1 - p))

# Parameters
n = np.arange(0, 100)
mu_nb = 20 # Mean product count is fixed
r_values = [1, 5, 20]  # Burstiness levels (r is the dispersion parameter)

# Plot
plt.figure(figsize=(8, 6))
for r in r_values:
    # Note: mu is fixed, r is varied to show the effect of burstiness
    plt.plot(n, nb_pmf(n, mu_nb, r), label=f"$\\mu$={mu_nb}, r$={r}$")

# plt.title("Case 2: Bursty Enzyme Expression → Negative Binomial($\\mu$, r)")
plt.xlabel("Product count ($n_p$)")
plt.ylabel("Probability [P($n_p$)]")
plt.legend()
plt.tight_layout()
plt.show()