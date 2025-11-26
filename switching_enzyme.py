import numpy as np
import matplotlib.pyplot as plt
# Removed: import seaborn as sns
import math

# Set a clean Matplotlib style manually (simulating a clean, academic look)
plt.style.use('default') 
plt.rcParams.update({
    'axes.grid': True,              
    'grid.linestyle': '--',         
    'axes.facecolor': 'white',      
    'figure.facecolor': 'white',    
    'lines.linewidth': 2.2,         
    'font.size': 14,                
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

# Poisson PMF function (standard formula)
def poisson_pmf(n, mu):
    n = np.asarray(n, dtype=float)
    # Using the log-gamma function for numerical stability
    return np.exp(n*np.log(mu) - mu - np.vectorize(math.lgamma)(n+1))

# Poisson Mixture PMF function (weighted average of two Poisson distributions)
def poisson_mixture_pmf(n, mus, weights):
    pmf = np.zeros_like(n, dtype=float)
    for w, mu in zip(weights, mus):
        pmf += w * poisson_pmf(n, mu)
    return pmf

# Parameters
n = np.arange(0, 80)
# Low mean (OFF state) and High mean (ON state)
mu_off, mu_on = 2, 30
# Probability of being in the ON state
p_ON_values = [0.2, 0.5, 0.8]

# Plot
plt.figure(figsize=(8, 5))
for p_ON in p_ON_values:
    # weights are [1 - p_ON, p_ON] for the [mu_off, mu_on] states
    pmf_mix = poisson_mixture_pmf(n, [mu_off, mu_on], [1 - p_ON, p_ON])
    plt.plot(n, pmf_mix, label=f"$p_{{ON}}$={p_ON}")

# plt.title("Case 3: ON/OFF Switching → Mixture of Poissons")
plt.xlabel("Product count ($n_p$)")
plt.ylabel("Probability [P($n_p$)]")
plt.legend(title="ON-State Probability")
plt.tight_layout()
plt.show()

# Probability of off-state = 1 - P(on-state)