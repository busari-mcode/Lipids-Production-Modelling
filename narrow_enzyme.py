import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# --- Fixed Parameters (Illustrative Values) ---

# k_cat: catalytic rate constant (units: s^-1)
k_cat = 100.0

# S: substrate concentration (units: M)
S = 1e-4

# delta_p: product degradation/dilution rate (units: s^-1)
delta_p = 0.1

# K_m: Michaelis constant (units: M)
K_m = 5e-5

# n_p: The specific number of product molecules for which we calculate the probability P(n_p)
# We must pick one integer value for the plot.
n_p_value = 5

# --- Function Definition ---

def calculate_P_np(n_e_tot, k_cat, S, delta_p, K_m, n_p):
    """
    Calculates the probability P(n_p) using the Poisson distribution formula.
    
    Parameters:
    - n_e_tot (array or float): Total enzyme concentration.
    - k_cat, S, delta_p, K_m (float): Fixed parameters.
    - n_p (int): The specific number of product molecules (np).
    
    Returns:
    - P_np (array or float): Probability P(n_p).
    """
    
    # Calculate the mean lambda (average number of products)
    # lambda = (k_cat * n_e_tot * S) / (delta_p * (K_m + S))
    numerator = k_cat * n_e_tot * S
    denominator = delta_p * (K_m + S)
    
    lambda_val = numerator / denominator
    
    # Calculate P(n_p) using the Poisson formula: 
    # P(n_p) = (lambda^n_p / n_p!) * exp(-lambda)
    
    P_np = (lambda_val**n_p / factorial(n_p)) * np.exp(-lambda_val)
    
    return P_np

# --- Data Generation ---

# Range for n_e_tot (Total enzyme concentration). 
# Choose a range appropriate for your system, e.g., from 0 to 1e-7 M.
n_e_tot_range = np.linspace(0, 1e-7, 500)

# Calculate P(n_p) for the chosen range of n_e_tot
P_np_results = calculate_P_np(n_e_tot_range, k_cat, S, delta_p, K_m, n_p_value)

# --- Plotting ---

plt.figure(figsize=(10, 6))

plt.plot(n_e_tot_range, P_np_results, label=f'$P(n_p = {n_p_value})$')

# plt.title(f'Probability $P(n_p)$ vs. Total Enzyme Concentration $n_e^{{tot}}$', fontsize=14)
plt.xlabel('Total Enzyme Concentration, $n_e^{tot}$ (M)', fontsize=12)
plt.ylabel(f'Probability, $P(n_p = {n_p_value})$', fontsize=12)

# Display the fixed parameter values in the plot
param_text = (
    f'$k_{{cat}} = {k_cat}$ s$^{{-1}}$\n'
    f'$S = {S:.1e}$ M\n'
    f'$K_m = {K_m:.1e}$ M\n'
    f'$\\delta_p = {delta_p}$ s$^{{-1}}$\n'
    f'$n_p = {n_p_value}$'
)
plt.text(0.7, 0.95, param_text, transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", alpha=0.1))

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()