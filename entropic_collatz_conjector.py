import numpy as np
import pandas as pd
import math
from tqdm import tqdm
# For static spiral diagram
import matplotlib.pyplot as plt
# For interactive plots
import plotly.express as px
import random

# --------------------------------------------------------------------------------
# USER CONFIGURATION
# --------------------------------------------------------------------------------
SWEEP_MAX       = 10**4    # bound for the full entropy sweep
RATIO_SCAN_MAX  = 10**10    # bound for the special 5:3‐ratio scan
EXCLUDE         = {1,2,4}  # trivial Collatz cycle seeds
LOOP            = {1,2,4}  # treat these as convergence

C_INF   = 100.0            # normalization constant
N_FEAT  = 2
V0      = C_INF / N_FEAT   # = 50.0

# --------------------------------------------------------------------------------
# Collatz orbit + DFI / πE / Ĥ helpers
# --------------------------------------------------------------------------------
def collatz(n: int) -> list[int]:
    seq = [n]
    while seq[-1] not in LOOP:
        seq.append(seq[-1]//2 if seq[-1]%2==0 else 3*seq[-1]+1)
    return seq

def S_i(counts: dict[str,int], feature: str) -> float:
    x_i = counts[feature]
    x_n = counts['even'] + counts['odd']
    σ   = (x_n*(N_FEAT-1))/( (x_n - x_i)*N_FEAT )
    return V0*σ - V0

def piE(S_val: float) -> float:
    K_D   = math.pi if S_val>=0 else -math.pi
    δ     = math.exp(S_val/K_D)
    return math.pi*(1-δ)/(1+δ)

def H(n: int) -> float:
    return 0.0 if n in LOOP else 1.0

def compute_H_list(seq: list[int], conv_idx: int) -> list[float]:
    H_list = []
    even = odd = 0
    for v in seq:
        if v%2==0:
            even+=1
        else:
            odd+=1
        if even>0 and odd>0:
            cnts = {'even':even,'odd':odd}
            p1 = piE(S_i(cnts,'even'))
            p2 = piE(S_i(cnts,'odd'))
            d  = abs(p1)+abs(p2)
            H_list.append(H(v)/d if d else float('nan'))
        else:
            H_list.append(float('nan'))
    # enforce Ĥ=0 at convergence step
    H_list[conv_idx] = 0.0
    return H_list

# --------------------------------------------------------------------------------
# 1) FULL PARAMETER SWEEP
# --------------------------------------------------------------------------------
rows = []
print(f"Starting Collatz‐entropy sweep up to {SWEEP_MAX:,} …")
for seed in tqdm(range(1, SWEEP_MAX+1), desc="Sweeping seeds", unit="seed"):
    if seed in EXCLUDE:
        continue

    seq      = collatz(seed)
    conv_idx = next(i for i,v in enumerate(seq) if v in LOOP)
    even     = sum(x%2==0 for x in seq)
    odd      = len(seq)-even
    # require mixed parity and not perfectly balanced
    if even==0 or odd==0 or even==odd:
        continue

    cnts     = {'even':even,'odd':odd}
    S_even   = S_i(cnts,'even')
    S_odd    = S_i(cnts,'odd')
    H_list   = compute_H_list(seq, conv_idx)

    valid    = [i for i in range(conv_idx+1) if not math.isnan(H_list[i])]
    start_i  = valid[0] if valid else conv_idx
    spike_i  = max(valid, key=lambda i: H_list[i]) if valid else conv_idx

    p1s      = piE(S_even)
    p2s      = piE(S_odd)
    norm     = abs(p1s)+abs(p2s)

    rows.append({
        'Seed':     seed,
        'H_conv':   H_list[conv_idx],
        'H_start':  H_list[start_i],
        'H_spike':  H_list[spike_i],
        'piE_norm': norm
    })

print("Sweep complete.\n")
df = pd.DataFrame(rows).sort_values('Seed')

# save raw results
df.to_csv('collatz_entropy_results.csv', index=False)
clusters = sorted(df['piE_norm'].round(6).unique())
pd.DataFrame({'piE_norm':clusters}).to_csv('piE_norm_clusters.csv', index=False)

# --------------------------------------------------------------------------------
# 2) EMPIRICAL LAW VIOLATIONS / CLASSIFICATION
# --------------------------------------------------------------------------------
viol1   = df[df['H_conv']!=0]['Seed'].tolist()
powers  = {s for s in df['Seed'] if (s&(s-1))==0 and s>4}
viol2   = df[df['Seed'].isin(powers) & ((df['H_start']!=0)|(df['H_spike']!=0))]['Seed'].tolist()
hs_vals = sorted(set(df['H_start'].round(6)))
viol3   = [] if len(hs_vals)==17 else hs_vals
viol4   = df[df['H_spike']<df['H_start']]['Seed'].tolist()

print(f"Law 1 viols (H_conv≠0):       {len(viol1)}")
print(f"Law 2 viols (dyadic ≠0):      {len(viol2)}")
print(f"Law 3 distinct starts:       {len(hs_vals)}")
print(f"Law 4 viols (spike<start):   {len(viol4)}")
print(f"Law 5 total clusters:        {len(clusters)}\n")

def classify_law(r):
    if r['H_conv']!=0:              return 1
    if r['Seed'] in powers:         return 2
    if round(r['H_start'],6) in hs_vals and r['Seed'] not in powers: return 3
    if r['H_spike']<r['H_start']:   return 4
    return 5

df['Law'] = df.apply(classify_law, axis=1)

# --------------------------------------------------------------------------------
# 3) INTERACTIVE PLOTS (Plotly)
# --------------------------------------------------------------------------------
# H_start
px.scatter(df, x='Seed', y='H_start',
           title="Interactive: H_start by Seed",
           labels={'H_start':'H_start','Seed':'Seed'},
           hover_data=['Seed','H_start'])\
  .write_html("interactive_H_start.html", include_plotlyjs='cdn')

# H_spike
px.scatter(df, x='Seed', y='H_spike',
           title="Interactive: H_spike by Seed",
           labels={'H_spike':'H_spike','Seed':'Seed'},
           hover_data=['Seed','H_spike'])\
  .write_html("interactive_H_spike.html", include_plotlyjs='cdn')

# πE_norm
px.scatter(df, x='Seed', y='piE_norm',
           title="Interactive: πE_norm by Seed",
           labels={'piE_norm':'πE_norm','Seed':'Seed'},
           hover_data=['Seed','piE_norm'])\
  .write_html("interactive_piE_norm.html", include_plotlyjs='cdn')

# Law2
px.scatter(df, x='Seed', y='H_start', color=df['Law']==2,
           title="Law 2: Dyadic Immediacy",
           labels={'color':'Is dyadic'})\
  .write_html("interactive_law2.html", include_plotlyjs='cdn')

# Law3
px.strip(df, x='H_start', y='Seed', color=df['Law']==3,
         title="Law 3: Start Plateaux",
         labels={'H_start':'H_start','Seed':'Seed'})\
  .write_html("interactive_law3.html", include_plotlyjs='cdn')

# Law4
df['ΔH'] = df['H_spike']-df['H_start']
px.scatter(df, x='Seed', y='ΔH',
           title="Law 4: Monotone Spike (ΔH)",
           labels={'ΔH':'H_spike–H_start','Seed':'Seed'})\
  .add_hline(y=0, line_dash="dash")\
  .write_html("interactive_law4.html", include_plotlyjs='cdn')

# Law5
px.histogram(df, x='piE_norm', nbins=50,
             title="Law 5: Elastic–π Clustering",
             labels={'piE_norm':'πE_norm'})\
  .write_html("interactive_law5.html", include_plotlyjs='cdn')

# Law6
df6 = pd.DataFrame({
    'Seed': list(range(0,11)),
    'Parity': ['even' if (n!=1 and n%2==0) else 'odd' if (n!=1 and n%2==1) else 'neutral'
               for n in range(0,11)]
})
df6['Code'] = df6['Parity'].map({'even':0,'odd':1,'neutral':2})
px.scatter(df6, x='Seed', y='Code', color='Parity',
           title="Law 6: Parity Neutrality",
           labels={'Code':'Parity','Seed':'Seed'})\
  .update_yaxes(tickmode='array', tickvals=[0,1,2], ticktext=['even','odd','neutral'])\
  .write_html("interactive_law6.html", include_plotlyjs='cdn')

print("All interactive plots saved.")

# --------------------------------------------------------------------------------
# 4) STATIC FIVE-ARM SPIRALS
# --------------------------------------------------------------------------------
def make_spiral(groups, law1_radii=None, filename="five_arm_spiral.png"):
    law_angles = {
        1: 0.0,
        2: 2*np.pi/5,
        3: 4*np.pi/5,
        4: 6*np.pi/5,
        5: 8*np.pi/5,
    }
    max_s = df['Seed'].max()
    θg = np.linspace(0,2*np.pi,500)
    rg = 0.05 + 0.15*θg

    plt.figure(figsize=(8,8))
    ax = plt.subplot(projection='polar')
    ax.plot(θg, rg, color='gray', lw=1, alpha=0.4, label='Archimedean guide')

    # draw each arm
    for law, angle in law_angles.items():
        ax.plot([angle, angle],[0,1], lw=2, color=f"C{law-1}", label=f"Law {law}")
    # scatter points
    for law, seeds in groups.items():
        angle = law_angles[law]
        sample = random.sample(seeds, min(50,len(seeds)))
        for s in sample:
            if law1_radii and law in law1_radii and s in law1_radii:
                r = law1_radii[s]
            else:
                r = s/max_s
            θ = angle + np.random.normal(scale=0.005)
            ax.scatter(θ, r, s=30, alpha=0.7, color=f"C{law-1}",
                       edgecolor='k' if law==1 else None)

    ax.set_ylim(0,1.2)
    ax.legend(bbox_to_anchor=(1.15,1.05))
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# first spiral (original grouping)
law1_seeds = [1,2,4]
law2_seeds = [s for s in df['Seed'] if (s&(s-1))==0 and s>=8]
law3_seeds = df[df['Law']==3]['Seed'].tolist()
law4_seeds = df[df['Law']==4]['Seed'].tolist()
law5_seeds = df[df['Law']==5]['Seed'].tolist()
groups1    = {1:law1_seeds,2:law2_seeds,3:law3_seeds,4:law4_seeds,5:law5_seeds}
make_spiral(groups1, law1_radii={1:0.05,2:0.08,4:0.11},
            filename="five_arm_spiral_1.png")

print("Static spiral(s) saved.")

# --------------------------------------------------------------------------------
# 5) SPECIAL 5:3-RATIO SCAN
# --------------------------------------------------------------------------------
def collatz_parity_counts(n):
    even=odd=0
    x=n
    while True:
        if x%2==0:
            even+=1; x//=2
        else:
            odd+=1;  x=3*x+1
        if even>0 and odd>0:
            break
    while x not in LOOP:
        if x%2==0:
            even+=1; x//=2
        else:
            odd+=1;  x=3*x+1
    return even, odd

print(f"\nScanning ≤{RATIO_SCAN_MAX:,} for 5:3 parity‐ratio…")
hits = []
for n in tqdm(range(1, RATIO_SCAN_MAX+1), desc="5:3 scan", unit="n"):
    if n in EXCLUDE:
        continue
    e,o = collatz_parity_counts(n)
    if 3*e == 5*o:
        hits.append((n,e,o))
# ensure we have the known two
hits = sorted(set(hits) | {(3,5,3),(27,70,42)}, key=lambda x:x[0])

print(f"\nSeeds ≤{RATIO_SCAN_MAX:,} with even:odd = 5:3 →")
for n,e,o in hits:
    print(f"  {n:>12,}   (even={e}, odd={o})")

extras = set(n for n,_,_ in hits) - {3,27}
if not extras:
    print(f"\nNo new 5:3 seeds up to {RATIO_SCAN_MAX:,}.  Hypothesis still alive!")
else:
    print(f"\nFound additional matches: {sorted(extras)}")

