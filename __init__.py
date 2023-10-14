# multi-targeted DMRG

# from flux_cu import Flux
import flux_cu
import cupy as np
from tqdm import tqdm
n_j = 12
n_excited = 3
phi_ext_range = np.linspace(0, np.pi / 2, 41)  # the sweep range of phi_ext
energies = np.zeros((len(phi_ext_range), n_excited))
f = flux_cu.Flux(N_J=n_j, N_excited=n_excited, Ds=12, N_Lanczos=6, basis='flux')
for n, var in enumerate(tqdm(phi_ext_range)):
    f.set_phiExt(var)
    energies[n] = np.array(f.optimizeT(usingLanczos=True, Eng_criterion=0.01, N_criterion=12, Quiet=False))
N = energies.shape[1]  # the number of lines in the spectrum (n_excited)
# Plotting the energy spectrum
energies = energies[::-1, :]
phi_e_range = phi_ext_range / np.pi