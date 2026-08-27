import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

m_shells = [1.7, 3.]
poros_shells = [.44, .502]    # breder: [.36, .502]
n_samples_shells = 14

m_gravel = [1.4, 1.8]
poros_gravel = [.245,.346]
avg_poros_gravel = .304
n_samples_gravel = 16

def archie(poros, m):
    return poros**-m

assert abs(archie(0.35, 1.3) - 3.914) < 1e-3



# observed information
# n = 14
# xmin = .44
# xmax = .502
n=16
xmin=.245
xmax=.346



R = xmax - xmin
midrange = (xmin + xmax) / 2

# Blom approximation
z = st.norm.ppf((n - 0.375) / (n + 0.25))

sigma_hat = R / (2 * z)
mu_hat = midrange

print(f"mu ≈ {mu_hat:.3f}")
print(f"sigma ≈ {sigma_hat:.3f}")


# Blom approximation
z = st.norm.ppf((n - 0.375)/(n + 0.25))

# estimate lognormal parameters
sigma_hat = np.log(xmax/xmin)/(2*z)
mu_hat = (np.log(xmin) + np.log(xmax))/2

print(f"mu = {mu_hat:.3f}")
print(f"sigma = {sigma_hat:.3f}")

# corresponding scipy distribution
dist = st.lognorm(
    s=sigma_hat,      # shape
    scale=np.exp(mu_hat)
)

print("median =", dist.median())
print("mean   =", dist.mean())
print("std    =", dist.std())
print("68confint    =", dist.interval(.68))