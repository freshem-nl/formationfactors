#%%
print("script has been made by copilot, has not been checked")

import numpy as np
import matplotlib.pyplot as plt

# parameters van de onderliggende normale verdeling
mu_ln_getransformeerd = 0.5
sigma_ln_getransformeerd = 0.7

n = 100_000
np.random.seed(42)

# 1. onderliggende normale verdeling
normal = np.random.normal(
    mu_ln_getransformeerd,
    sigma_ln_getransformeerd,
    n,
)

# 2. lognormale verdeling via numpy
lognormal = np.random.lognormal(
    mu_ln_getransformeerd,
    sigma_ln_getransformeerd,
    n,
)

# 3. lognormale verdeling via exp(normal)
exp_normal = np.exp(
    np.random.normal(
        mu_ln_getransformeerd,
        sigma_ln_getransformeerd,
        n,
    )
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(normal, bins=75, density=True, color="steelblue")
axes[0].set_title(
    "Onderliggende normaal verdeelde dataset\n"
    "np.random.normal(\n"
    "    mu_ln_getransformeerd,\n"
    "    sigma_ln_getransformeerd\n"
    ")"
)

axes[1].hist(lognormal, bins=75, density=True, color="orange")
axes[1].set_title(
    "Lognormaal verdeelde dataset\n"
    "np.random.lognormal(\n"
    "    mu_ln_getransformeerd,\n"
    "    sigma_ln_getransformeerd\n"
    ")"
)

axes[2].hist(exp_normal, bins=75, density=True, color="green")
axes[2].set_title(
    "exp(normaal verdeelde dataset)\n"
    "np.exp(\n"
    "    np.random.normal(\n"
    "        mu_ln_getransformeerd,\n"
    "        sigma_ln_getransformeerd\n"
    "    )\n"
    ")"
)

for ax in axes:
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

#%%

np.random.seed(42)

x1 = np.random.lognormal(mu_ln_getransformeerd, sigma_ln_getransformeerd, 100000)

np.random.seed(42)
x2 = np.exp(np.random.normal(mu_ln_getransformeerd, sigma_ln_getransformeerd, 100000))


np.allclose(x1, x2)

#%%

GM = np.exp(mu_ln_getransformeerd) # Geometrisch gemiddelde (GM) = mediaan originele ruimte
MSD = np.exp(sigma_ln_getransformeerd) # Multiplicatieve standaardafwijking

np.random.seed(42)

x3 = np.random.lognormal(
    np.log(GM),
    np.log(MSD),
    1000000000,
)

np.allclose(x2, x3)