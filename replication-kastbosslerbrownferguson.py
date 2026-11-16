import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA_FILE = "data-set.csv"

df = pd.read_csv(DATA_FILE, encoding="latin-1")

dv = "osvAll"
x_vars = [
    "troopLag",              # UN Military Troops (t-1), in 1,000s
    "policeLag",             # UN Police (t-1), in 1,000s
    "militaryobserversLag",  # UN Observers (t-1), in 1,000s
    "epduration",            # Conflict Duration (months)
    "incomp",                # Government Conflict (1 = territory, 2 = government)
    "lntpop",                # Population (logged)
    "brv_AllLag",            # All Battle Deaths (t-1)
    "osvAllLagDum",          # All OSV Dummy (t-1)
]

labels = {
    "troopLag": "UN Military Troops (t-1)",
    "policeLag": "UN Police (t-1)",
    "militaryobserversLag": "UN Observers (t-1)",
    "epduration": "Conflict Duration",
    "incomp": "Government Conflict",
    "lntpop": "Population",
    "brv_AllLag": "All Battle Deaths (t-1)",
    "osvAllLagDum": "All OSV Dummy (t-1)",
    "const": "Constant",
    "alpha": "Alpha",
}

decimals = {
    "epduration": 4,
    "brv_AllLag": 3,
}

data = df.dropna(subset=[dv] + x_vars + ["conflict_id"])
print("Observations:", len(data))
print("Conflicts (clusters):", data["conflict_id"].nunique())
print("Conflict episodes:", data["id_ep"].nunique())

# part a
y = data[dv]
print()
print("DV mean:", round(y.mean(), 2))
print("DV variance:", round(y.var(), 2))
print("Share of months with zero killings:", round((y == 0).mean(), 3))
print("DV max:", y.max())

# fit negative binomial model
X = sm.add_constant(data[x_vars])
model = sm.NegativeBinomial(y, X, loglike_method="nb2")

start = np.array([-10.0, -0.5, -10.0, 20.0, 0.0, 2.0, 0.7, 0.0, 2.0, 15.0])

result = model.fit(
    start_params=start,
    method="newton",
    maxiter=500,
    disp=0,
    cov_type="cluster",
    cov_kwds={"groups": data["conflict_id"], "use_correction": False},
)

if not result.mle_retvals["converged"]:
    print("WARNING: model did not converge")

G = data["conflict_id"].nunique()
vcov = result.cov_params() * (G / (G - 1))

coefs = result.params
ses = pd.Series(np.sqrt(np.diag(vcov)), index=coefs.index)
z = coefs / ses
pvals = pd.Series(2 * stats.norm.sf(np.abs(z)), index=coefs.index)

# wald chi-square
b = coefs[x_vars].values
V = vcov.loc[x_vars, x_vars].values
wald = b @ np.linalg.inv(V) @ b
wald_df = len(x_vars)
wald_p = stats.chi2.sf(wald, wald_df)


def stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def fmt(number, name=None):
    d = decimals.get(name, 2)
    return f"{number:.{d}f}"


row_order = x_vars + ["const", "alpha"]

rows = []
for name in row_order:
    coef_text = fmt(coefs[name], name) + stars(pvals[name])
    se_text = "(" + fmt(ses[name], name) + ")"
    rows.append([labels[name], coef_text, se_text])

print()
print("Table 1, Model 1: Effect of Peacekeeping Personnel on All OSV")
print("Negative binomial, robust SEs clustered on conflict")
print("-" * 52)
print(f"{'Variable':<30}{'Coef.':>12}{'(SE)':>10}")
print("-" * 52)
for r in rows:
    print(f"{r[0]:<30}{r[1]:>12}{r[2]:>10}")
print("-" * 52)
print(f"{'Observations':<30}{len(data):>12,}")
print(f"{'Conflicts (clusters)':<30}{G:>12}")
print(f"{'Wald chi2(' + str(wald_df) + ')':<30}{wald:>12.2f}{stars(wald_p)}")
print(f"{'Prob > chi2':<30}{wald_p:>12.3g}")
print(f"{'Log pseudo-likelihood':<30}{result.llf:>12.2f}")
print("-" * 52)
print("* p < 0.05, ** p < 0.01, *** p < 0.001 (two-tailed)")

def p_text(p):
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"

with open("model1_table.md", "w", encoding="utf-8") as f:
    f.write("**Table: Replication of Hultman et al. (2013), Table 1, Model 1**\n\n")
    f.write("DV: monthly count of civilians killed (all one-sided violence)\n\n")
    f.write("| Variable | Coefficient | Std. Error |\n")
    f.write("|---|---:|---:|\n")
    for r in rows:
        coef_md = r[1].replace("*", "\\*")
        f.write(f"| {r[0]} | {coef_md} | {r[2]} |\n")
    f.write(f"| Observations | {len(data):,} | |\n")
    f.write(f"| Conflicts (clusters) | {G} | |\n")
    f.write(f"| Conflict episodes | {data['id_ep'].nunique()} | |\n")
    wald_md = f"{wald:.2f}" + stars(wald_p).replace("*", "\\*")
    f.write(f"| Wald χ²({wald_df}) | {wald_md} | |\n")
    f.write(f"| Wald χ²({wald_df}) p-value | {p_text(wald_p)} | |\n")
    f.write(f"| Log pseudo-likelihood | {result.llf:.2f} | |\n")
    f.write("\nNegative binomial regression. Robust standard errors clustered on "
            "conflict in parentheses. UN personnel measured in thousands. "
            "\\* p < 0.05, \\*\\* p < 0.01, \\*\\*\\* p < 0.001 (two-tailed).\n")
print("\nSaved table to model1_table.md")

lines = ["Effects on expected monthly civilian killings"]
for name in ["troopLag", "policeLag", "militaryobserversLag"]:
    b = coefs[name]
    for n in [100, 1000]:
        irr = np.exp(b * n / 1000)
        pct = (irr - 1) * 100
        lines.append(f"{labels[name]}, +{n} personnel: IRR = {irr:.4f}, change = {pct:+.1f}%")

with open("model1_effects.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

import pandas as pd, numpy as np, statsmodels.api as sm
d=pd.read_csv('/mnt/user-data/uploads/a4fdf94e-data-set_1.csv',encoding='latin1')
d['govconf']=(d.incomp==2).astype(int)
X=['troopLag','policeLag','militaryobserversLag','epduration','govconf','lntpop','brv_AllLag','osvAllLagDum']
m=d.dropna(subset=['osvAll']+X).copy()
mod=sm.NegativeBinomial(m.osvAll, sm.add_constant(m[X]), loglike_method='nb2')
sp=np.array([-9.24,-.53,-9.9,21.76,-.0006,2.38,.70,.001,2.18,16.89])
r=mod.fit(start_params=sp,cov_type='cluster',cov_kwds={'groups':m.conflict_id},maxiter=200,disp=0,method='newton')
print(len(m), m.conflict_id.nunique()); print(r.summary2().tables[1].iloc[:,:4].round(4)); print(r.llf)
import pickle; pickle.dump((r.params,r.cov_params(),m[X].mean(),m),open('/home/claude/work/fit.pkl','wb'))
import pickle, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
b,V,means,m=pickle.load(open('fit.pkl','rb'))
names=list(b.index[:-1]); bb=b.values[:-1]; VV=V.values[:-1,:-1]
rng=np.random.default_rng(2013); draws=rng.multivariate_normal(bb,VV,10000)
base=means.copy(); base['govconf']=1; base['osvAllLagDum']=1   # paper footnote 5
def curve(var,grid):
    X=np.tile(np.r_[1,base.values],(len(grid),1)); X[:,names.index(var)]=grid
    sims=np.exp(draws@X.T)
    return np.exp(X@bb), sims.mean(0), np.percentile(sims,2.5,0), np.percentile(sims,97.5,0)
out={}
for var,lab,fn,mx in [('troopLag','UN Military Troops','fig4_troops.png',m.troopLag.max()),
                      ('policeLag','UN Police','fig5_police.png',m.policeLag.max())]:
    g=np.linspace(0,mx,600); pt,ev,lo,hi=curve(var,g); out[var]=(g,pt,ev,lo,hi)
    fig,ax=plt.subplots(figsize=(8,5),dpi=150)
    ax.fill_between(g*1000,lo,hi,color='#9ec5e8',alpha=.6,lw=0,label='95% Confidence Interval')
    ax.plot(g*1000,ev,color='#1f5f99',lw=2,label='Predicted Civilian Deaths')
    ax.set_xlabel(lab); ax.set_ylabel('Number of Predicted Civilian Deaths'); ax.set_ylim(0,180); ax.set_xlim(0,mx*1000)
    ax.set_title(f'Predicted Civilian Deaths as {lab} Increase\n(Model 1 negative binomial; other variables at means)',fontsize=11)
    ax.grid(axis='y',color='#ddd',lw=.6); [ax.spines[s].set_visible(False) for s in ['top','right']]
    ax.legend(frameon=False,loc='upper right'); fig.tight_layout(); fig.savefig(fn)
    # zoomed inset version for low range
    fig,ax=plt.subplots(figsize=(8,5),dpi=150); z=g<= (8 if var=='troopLag' else 1)
    ax.fill_between(g[z]*1000,lo[z],hi[z],color='#9ec5e8',alpha=.6,lw=0,label='95% Confidence Interval')
    ax.plot(g[z]*1000,ev[z],color='#1f5f99',lw=2,label='Predicted Civilian Deaths')
    ax.set_xlabel(lab); ax.set_ylabel('Number of Predicted Civilian Deaths'); ax.set_ylim(0,180); ax.set_xlim(0,g[z].max()*1000)
    ax.set_title(f'{lab}: low deployment range',fontsize=11); ax.grid(axis='y',color='#ddd',lw=.6)
    [ax.spines[s].set_visible(False) for s in ['top','right']]; ax.legend(frameon=False); fig.tight_layout(); fig.savefig(fn.replace('.png','_zoom.png'))
for var,pts in [('troopLag',[0,1,2,4,8]),('policeLag',[0,.1,.2,.5,1])]:
    for p in pts:
        X=np.r_[1,base.values]; X[names.index(var)]=p; s=np.exp(draws@X)
        print(var,p*1000,round(np.exp(X@bb),2),round(s.mean(),2),np.percentile(s,[2.5,97.5]).round(2))
print(means.round(3).to_dict())
