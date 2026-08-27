import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# hoe werken al die distributies nu
rho = 5 # 28.47
rho_std = 1.362

confint68 = [rho/rho_std, rho*rho_std]


samples = np.random.lognormal(
    mean=np.log(rho),
    sigma=np.log(rho_std),
    size=10000,
    )


fitted_prms = st.lognorm.fit(samples)


# normal in logspace?
samples2 = rho * np.exp(np.random.normal(
        loc=0.0,
        scale=np.log(rho_std),
        size=10000,
        )
    )
fitted_prms2 = st.lognorm.fit(samples2)


#### Deze dus niet...
# lognorm with log10
samples3 = rho * 10**(np.random.normal(
        loc=0.0,
        scale=np.log10(rho_std),
        size=10000,
        )
    )
fitted_prms3 = st.lognorm.fit(samples3)


# direct obv rho en rho_std
def aarhus_lognormal(rho, rho_std):
    s=np.log(rho_std)
    loc=0.0
    scale=rho
    return st.lognorm(
        s=s,
        loc=loc,
        scale=scale,
        ), [s,loc,scale]

dist, dist_prms = aarhus_lognormal(rho, rho_std)
samples4 = dist.rvs(10000)


# maak overzichts figuren
def plot_cdf(samples, fitted_prms=None, ax=None, color="b", ls="-", label=""):
    if ax is None:
        fig,ax = plt.subplots()
    if fitted_prms is not None:
        median = st.lognorm.median(*fitted_prms)
        mu = st.lognorm.mean(*fitted_prms)
        stdev = st.lognorm.std(*fitted_prms)
        confint = st.lognorm.interval(.68,*fitted_prms)
        ax.axvline(mu,color=color, ls="-")
        ax.axvline(median,color=color, ls="-.")
        ax.axvline(confint[0],color=color, ls=":")
        ax.axvline(confint[1],color=color, ls=":")

        label = f"{label} ({mu:.1f}, {stdev:.1f})"
    ax.plot(sorted(samples), np.arange(1,len(samples)+1)/len(samples),color=color, ls=ls, label=label)
    return ax

ax = plot_cdf(samples, fitted_prms, label="lognorm")
plot_cdf(samples2, fitted_prms2, color="green",ls="--",label="ln(norm)", ax=ax)
plot_cdf(samples3, fitted_prms3, color="orange",ls="--",label="log10(norm)", ax=ax)
plot_cdf(samples4, dist_prms, color="red",ls=":",label="dist prms", ax=ax)
plt.grid()
plt.legend()
plt.show()



