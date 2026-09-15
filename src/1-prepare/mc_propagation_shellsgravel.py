#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.integrate import quad
from pathlib import Path
import pandas as pd


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

def plot_dist(litho, F, dist, fn=None, title=None):
    plt.hist(F, 100, alpha=.5)
    ymin,ymax = plt.ylim()
    mu = dist.mean()
    p50 = dist.median()
    sigma = dist.std()
    confint = dist.interval(.68)
    plt.vlines(mu, ymin,ymax, ls="-", label=f"mu = {mu:.3f}")
    plt.vlines(p50, ymin,ymax, ls="-.", label=f"median = {p50:.3f}")
    plt.vlines(confint, ymin,ymax, ls=":", label=f"stdev = {sigma:.3f}")
    plt.ylim(ymin,ymax)
    plt.grid()
    plt.legend()
    ax2 = plt.twinx()
    X = np.linspace(0,np.max(F),1000)
    ax2.plot(X,dist.pdf(X))
    ymin2,ymax2 = plt.ylim()
    plt.ylim(0,ymax2)

    if title is None:
        plt.title(f"Estimated formation factor for {litho}")
    else: 
        plt.title(title)
        
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        plt.close()


## from information Ronald(measured porosities) and literature review on ranges of m (copilot)
m_shells = [1.7, 3.]
poros_shells = [.36, .502]    # breder: [.36, .502] #[.44, .502]
n_samples_shells = 14

m_gravel = [1.4, 1.8]
poros_gravel = [.245,.346]
avg_poros_gravel = .304
n_samples_gravel = 16

characteristics = {"shells":{"m":{"range":m_shells, "n":n_samples_shells},
                             "poros":{"range":poros_shells, "n":n_samples_shells},
                             },
                   "gravel":{"m":{"range":m_gravel, "n":n_samples_gravel},
                             "poros":{"range":poros_gravel, "n":n_samples_gravel},
                             },
                    }

path_output = Path("data/4-output/unc_propagation_shellsgravel")

if __name__ == "__main__":

    path_output.mkdir(parents=True, exist_ok=True)


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
            # TODO: volgens mij moet de -CDF-1 positief zijn (dus CDF-1) in E(r,n) = -CDF-1 ((r - a) / (n - 2a+1))
            z = st.norm.ppf((n - 0.375)/(n + 0.25))  # Blom's approximation is: E(r,n) = -CDF-1 ((r - a) / (n - 2a+1))   --> so, r expected largest order. rangnummer r van het maximum is n
            # estimate lognormal parameters
            sigma_blom = np.log(xmax/xmin)/(2*z) # geeft sigma_log (sigma van onderliggende normaal verdeling van log getransformeerde dataset)

            # wat ook kan: integratie van d_n
            dn = d_n(n)
            sigma_dn = (
                    ymax - ymin
                ) / dn
            
            # corresponding distribution:
            print(f"{litho},{prm}: {mu_hat}, {sigma_blom}, {sigma_dn}")
            dist = st.lognorm(s=sigma_dn, scale=np.exp(mu_hat)) # st.lognorm(s=sigma_log, scale=np.exp(mu_log)) -> geeft waardes terug in de oorspronkeljke ruimte
            characteristics[litho][prm]["dist"] = dist



    #### Dan: MonteCarlo om F te karakteriseren
    # draw n samples and calculate F
    rows = []
    n = 1000000
    for litho in ["shells","gravel"]:
        m = characteristics[litho]["m"]["dist"].rvs(n)
        poros = characteristics[litho]["poros"]["dist"].rvs(n)
        F = archie(poros, m)
        res = st.lognorm.fit(F)
        shape, loc, scale = res
        fitted_dist = st.lognorm(*res)

        print(f"FF {litho}: sigma_log = {shape}, scale or exp(mu_log) = {scale}, mu_log = {np.log(scale)}")

        rows.append(
            {
                "parameter": litho,
                "FF scale": round(scale,1),
                "FF mu_log": round(np.log(scale),1),
                "FF MSD": round(np.exp(shape),1),
                "FF sigma_log": round(shape,1),
            }
        )

    results_df = pd.DataFrame(rows)

    results_df.to_csv(f"{path_output}/FF_shells_gravel")

    #%%
