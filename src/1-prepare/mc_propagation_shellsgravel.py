import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.integrate import quad
from pathlib import Path

## from information Ronald(measured porosities) and literature review on ranges of m (copilot)
m_shells = [1.7, 3.]
poros_shells = [.44, .502]    # breder: [.36, .502]
n_samples_shells = 14

m_gravel = [1.4, 1.8]
poros_gravel = [.245,.346]
avg_poros_gravel = .304
n_samples_gravel = 16



path_output = Path("data/4-output/unc_propagation_shellsgravel")
path_output.mkdir(parents=True, exist_ok=True)



characteristics = {"shells":{"m":{"range":m_shells, "n":n_samples_shells},
                             "poros":{"range":poros_shells, "n":n_samples_shells},
                             },
                   "gravel":{"m":{"range":m_gravel, "n":n_samples_gravel},
                             "poros":{"range":poros_gravel, "n":n_samples_gravel},
                             },
                    }

def archie(poros, m):
    return poros**-m

assert abs(archie(0.35, 1.3) - 3.914) < 1e-3

def d_n(n):
    """
    Expected range of n samples from N(0,1):
        d_n = E[max - min]
    """

    def integrand(z):
        return (
            z
            * n
            * st.norm.pdf(z)
            * st.norm.cdf(z) ** (n - 1)
        )

    ezmax, _ = quad(
        integrand,
        -np.inf,
        np.inf,
        epsabs=1e-10,
        epsrel=1e-10
    )

    return 2 * ezmax

def plot(litho, F, dist, fn=None):
    plt.hist(F, 100)
    ymin,ymax = plt.ylim()
    mu = dist.mean()
    p50 = dist.median()
    sigma = dist.std()
    confint = dist.interval(.68)
    plt.vlines(mu, ymin,ymax, ls="-", label=f"mu = {mu:.1f}")
    plt.vlines(p50, ymin,ymax, ls="-.", label=f"median = {p50:.1f}")
    plt.vlines(confint, ymin,ymax, ls=":", label=f"stdev = {sigma:.1f}")
    plt.grid()
    plt.legend()
    plt.title(f"Estimated formation factor for {litho}")
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        plt.close()




# First step is to infer a distribution from observed ranges
# Let's say X ~ LogNormal(mu,sigma),
# Then Y = ln(X) ~ N(mu,sigma)
# mu is geometric midpoint of range
# d_n is de range van een standaardnormale steekproef: d_n = E[Z(n) - Z(1)], dat geeft: sigma_hat = ln(xmax/xmin) / d_n
# alternatief is Blom's approximation. Maakt weinig uit...

# transform endpoints to log-space
for litho in ["shells","gravel"]:
    for prm in ["m", "poros"]:
        n = characteristics[litho][prm]["n"]
        xmin,xmax = characteristics[litho][prm]["range"]
        # log-space bounds
        ymin = np.log(xmin)
        ymax = np.log(xmax)

        # estimate mu from geometric midrange
        mu_hat = 0.5 * (ymin + ymax)

        # Blom approximation
        z = st.norm.ppf((n - 0.375)/(n + 0.25))  # Blom's approximation is: E(r,n) = -CDF-1 ((r - a) / (n - 2a+1))   --> so, r expected largest order. rangnummer r van het maximum is n
        # estimate lognormal parameters
        sigma_blom = np.log(xmax/xmin)/(2*z)

        # wat ook kan: integratie van d_n
        dn = d_n(n)
        sigma_dn = (
                ymax - ymin
            ) / dn
        
        # corresponding distribution:
        print(f"{litho},{prm}: {mu_hat}, {sigma_blom}, {sigma_dn}")
        dist = st.lognorm(s=sigma_dn, scale=np.exp(mu_hat))
        characteristics[litho][prm]["dist"] = dist



#### Dan: MonteCarlo om F te karakteriseren
# draw n samples and calculate F
n = 10_000
for litho in ["shells","gravel"]:
    m = characteristics[litho]["m"]["dist"].rvs(n)
    poros = characteristics[litho]["poros"]["dist"].rvs(n)
    F = archie(poros, m)
    res = st.lognorm.fit(F)
    fitted_dist = st.lognorm(*res)
