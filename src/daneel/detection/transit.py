import batman, numpy as np
import matplotlib.pyplot as plt

p = batman.TransitParams()
p.t0 = 0.0 # relative time, days            # Sing at al. 2024
p.per = 1.27492504 # orbital period, days   # Sing at al. 2024
p.rp = 0.122551 # Rp / Rstar                # Sing at al. 2024
p.a = 3.7844 # a / Rstar                    # Sing at al. 2024
p.inc = 88.49 # degrees                     # Bourrier et al. 2020
p.ecc = 0.0085; p.w = 88.95                 # Pellettier et al. 2026
p.limb_dark = "quadratic"
p.u = [0.086066667, 0.1542]
t = np.linspace(-0.1, 0.1, 1000)
flux = batman.TransitModel(p,t).light_curve(p)

# Quick, approximate transit depth for an un-darkened stellar disc.
print(f"Approx. transit depth: {(p.rp**2)*100:.2f}%")

# 1,000 simulated time samples over 0.15 days (about 3.6 hours).
print(f"Time coverage: {t[0]:.3f} to {t[-1]:.3f} days, {len(t)} samples")

model = batman.TransitModel(p, t)  # initialise the transit model
flux = model.light_curve(p)           # calculate the normalised flux

assert flux.shape == t.shape
assert np.all(np.isfinite(flux)), "The light curve contains non-finite values"
print(f"Minimum model flux: {flux.min():.6f}")
print(f"Maximum model flux: {flux.max():.6f}")

plt.figure(figsize=(8, 6))
plt.plot(t,flux, color='r', label="BATMAN transit model")
plt.xlabel('Time from central transit [days]', fontsize=12, color='black')
plt.ylabel('Normalized Flux', fontsize=12, color='black')
plt.axhline(1.0, color="0.6", lw=1, ls="--", label="Out-of-transit flux")
plt.ylim(0.982, 1.002) 
plt.grid(True, color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.title('Light curve - WASP-121b')
plt.legend(loc='lower left')
# plt.show()
plt.savefig("WASP-121b_assignment1_taskF.png")