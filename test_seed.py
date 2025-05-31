import os, math, time, random
# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
# force a single OpenMP thread to avoid the MKL memory-leak warning
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.metrics        import silhouette_score
from sklearn.mixture        import GaussianMixture
from sklearn.cluster        import MiniBatchKMeans
import umap as umap
import matplotlib.pyplot as plt
from plotly import graph_objects as go
import plotly.express as px
import pickle
import joblib
# import plotly.graph_objects as go

# --------------------------------------------------------------------------------
# USER CONFIGURATION
# --------------------------------------------------------------------------------
SWEEP_MAX     = 10**7      # entropy sweep
PLOT_SAMPLE   = 10000      # sample size for plotting
EXCLUDE       = {1,2,4}
LOOP          = {1,2,4}
C_INF         = 100.0
N_FEAT        = 2
V0            = C_INF/N_FEAT
FEATURES      = ['H_start','H_spike','piE_norm',
                 'parity_ratio','piE_start','piE_spike']
FEATURES_FILE = 'features.pkl'
MICRO_FILE    = 'micro.npz'

FINAL_K       = 4
MICRO_K       = FINAL_K * 5  # e.g. 20

# --------------------------------------------------------------------------------
# 1) COLLATZ + DFI/πE/Ĥ HELPERS
# --------------------------------------------------------------------------------
def collatz(n):
    seq=[n]
    while seq[-1] not in LOOP:
        seq.append(seq[-1]//2 if seq[-1]%2==0 else 3*seq[-1]+1)
    return seq

def S_i(cnts,feature):
    x_i=cnts[feature]; x_n=cnts['even']+cnts['odd']
    σ=(x_n*(N_FEAT-1))/((x_n-x_i)*N_FEAT)
    return V0*σ - V0

def piE(Sv):
    KD=math.pi if Sv>=0 else -math.pi
    δ=math.exp(Sv/KD)
    return math.pi*(1-δ)/(1+δ)

def H(n): return 0.0 if n in LOOP else 1.0

def compute_H_list(seq):
    even=odd=0
    Hlist=[]
    conv=next(i for i,v in enumerate(seq) if v in LOOP)
    for v in seq:
        if v%2==0: even+=1
        else:      odd+=1
        if even>0 and odd>0:
            cnts={'even':even,'odd':odd}
            p1=piE(S_i(cnts,'even'))
            p2=piE(S_i(cnts,'odd'))
            d=abs(p1)+abs(p2)
            Hlist.append(H(v)/d if d else float('nan'))
        else:
            Hlist.append(float('nan'))
    Hlist[conv]=0.0
    return Hlist

def compute_piE_at(seq,idx):
    even=sum(v%2==0 for v in seq[:idx+1])
    odd =(idx+1)-even
    cnts={'even':even,'odd':odd}
    try:
        pe=piE(S_i(cnts,'even')); po=piE(S_i(cnts,'odd'))
    except ZeroDivisionError:
        pe=po=0.0
    return pe,po

# --------------------------------------------------------------------------------
# 2) FULL PARAMETER SWEEP (INCREMENTAL)
# --------------------------------------------------------------------------------
results_csv='collatz_entropy_results.csv'
if os.path.exists(results_csv):
    df_prev=pd.read_csv(results_csv)
    max_prev=int(df_prev['Seed'].max())
else:
    df_prev=pd.DataFrame(); max_prev=0

start=max_prev+1; end=SWEEP_MAX
if start> end:
    print("No new seeds to sweep.")
    df=df_prev.copy()
else:
    print(f"Sweeping seeds {start}→{end} …")
    rows=[]
    for n in tqdm(range(start,end+1),unit='seed'):
        if n in EXCLUDE: continue
        seq=collatz(n)
        conv=next(i for i,v in enumerate(seq) if v in LOOP)
        ev=sum(v%2==0 for v in seq)
        od=len(seq)-ev
        if ev==0 or (od>0 and ev==od): continue
        cnts={'even':ev,'odd':od}
        Ht=compute_H_list(seq)
        valid=[i for i in range(conv+1) if not math.isnan(Ht[i])]
        i0=valid[0] if valid else conv
        isp=max(valid,key=lambda i:Ht[i]) if valid else conv
        try:
            norm=abs(piE(S_i(cnts,'even')))+abs(piE(S_i(cnts,'odd')))
        except ZeroDivisionError:
            norm=0.0
        rows.append({'Seed':n,
                     'H_conv':Ht[conv],
                     'H_start':Ht[i0],
                     'H_spike':Ht[isp],
                     'piE_norm':norm})
    df_new=pd.DataFrame(rows)
    df=pd.concat([df_prev,df_new],ignore_index=True).sort_values('Seed')
    df.to_csv(results_csv,index=False)
    print("✔︎ Sweep saved.")

# --------------------------------------------------------------------------------
# 3a) FEATURE ENGINEERING & CACHE
# --------------------------------------------------------------------------------
# 3a) FEATURE ENGINEERING & CACHE
if os.path.exists(FEATURES_FILE):
    df = pd.read_pickle(FEATURES_FILE)
    print(f"✔︎ Loaded features from {FEATURES_FILE}")
else:
    t0 = time.time()
    pr = []; ps = []; pg = []
    # use itertuples to avoid allocating a full (n_rows × n_cols) float64 array
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="Feat"):
        seed = int(r.Seed)
        seq  = collatz(seed)
        conv = next(i for i,v in enumerate(seq) if v in LOOP)
        Ht   = compute_H_list(seq)
        valid = [i for i in range(conv+1) if not math.isnan(Ht[i])]
        s0 = valid[0] if valid else conv
        sp = max(valid, key=lambda i:Ht[i]) if valid else conv

        ev = sum(v%2==0 for v in seq)
        od = len(seq) - ev

        pr.append(ev/od if od>0 else float('nan'))

        pe1, _ = compute_piE_at(seq, s0)
        pe2, _ = compute_piE_at(seq, sp)
        ps.append(abs(pe1)*2)
        pg.append(abs(pe2)*2)

    # assign back in one go
    df['parity_ratio'] = pd.Series(pr, index=df.index) \
                         .replace([np.inf, -np.inf], np.nan) \
                         .fillna(method='ffill')
    df['piE_start']   = ps
    df['piE_spike']   = pg
    df.to_pickle(FEATURES_FILE)
    print(f"✔︎ Feat done in {time.time()-t0:.1f}s")


# --------------------------------------------------------------------------------
# 3b-i) BUILD MICRO-POINTS & CACHE (FIXED FORMAT + FLOAT32)
# --------------------------------------------------------------------------------
if os.path.exists(MICRO_FILE):
    npz = np.load(MICRO_FILE, mmap_mode='r')  # efficient memory loading
    micro_pts = npz['points'].astype(np.float32)
    micro_sds = npz['seeds']
    print(f"✔︎ Loaded {len(micro_pts)} micro–points from cache.")
else:
    t0 = time.time()
    # Mikro‐parametre
    BATCH     = 1000
    STD       = 1e-4
    DTYPE     = np.float32
    MICRO_FILE = 'micro.npz'
    MEMMAP_POINTS = 'micro_tmp_points.dat'
    MEMMAP_SEEDS  = 'micro_tmp_seeds.dat'

    # Klargør data
    X     = df[FEATURES].values.astype(DTYPE)
    safe  = np.where(np.isfinite(X), X, 0.0)
    seeds = df['Seed'].values
    nrows = len(df)
    nfeat = X.shape[1]
    n_micro = MICRO_K * nrows

    # Preallocate disk-backed arrays
    pts_memmap  = np.memmap(MEMMAP_POINTS, dtype=DTYPE, mode='w+', shape=(n_micro, nfeat))
    seeds_memmap = np.memmap(MEMMAP_SEEDS, dtype=seeds.dtype, mode='w+', shape=(n_micro,))

    print(f"↻ Allocated memmap for {n_micro} micro-points ({nfeat} features)")

    # Start process
    t0 = time.time()
    write_idx = 0

    for i in tqdm(range(0, nrows, BATCH), desc="micro"):
        block = X[i:i+BATCH]
        sb    = safe[i:i+BATCH]
        seed_block = seeds[i:i+BATCH]
        nb = len(block)

        for _ in range(MICRO_K):
            noise = np.random.randn(*sb.shape).astype(DTYPE) * STD * np.abs(sb)
            sample = block + noise
            n = len(sample)

            pts_memmap[write_idx:write_idx+n] = sample
            seeds_memmap[write_idx:write_idx+n] = seed_block
            write_idx += n

    # Trim overskud (i tilfælde af sidste block < BATCH)
    pts_memmap.flush()
    seeds_memmap.flush()

    # Gem til .npz
    micro_pts = np.array(pts_memmap[:write_idx])
    micro_sds = np.array(seeds_memmap[:write_idx])
    np.savez_compressed(MICRO_FILE, points=micro_pts, seeds=micro_sds)

    # Efter at have skrevet og flushet:
    pts_memmap.flush()
    seeds_memmap.flush()

    # Luk de underliggende mmap-objekter
    if hasattr(pts_memmap, '_mmap'):
        pts_memmap._mmap.close()
    if hasattr(seeds_memmap, '_mmap'):
        seeds_memmap._mmap.close()

    # Fjern referencerne, så Python kan rydde op
    del pts_memmap, seeds_memmap

    import gc
    gc.collect()  # sørger for, at mmap-objekterne bliver aflivet

    # Nu kan du fjerne filerne uden PermissionError
    os.remove(MEMMAP_POINTS)
    os.remove(MEMMAP_SEEDS)

    print(f"✔︎ Built {len(micro_pts)} micro–points in {time.time()-t0:.1f}s — saved to {MICRO_FILE}")

# --------------------------------------------------------------------------------
# 3b-ii) FAST METRIC SWEEP (sample-based) with caching
# --------------------------------------------------------------------------------
METRICS_FILE = 'metric_sweep.npz'

# sample up to 100k seeds for speed
SAMP = min(len(df), 100000)
dfm  = df.sample(SAMP, random_state=0)
Xr   = dfm[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(dfm[FEATURES].median())
Xs   = StandardScaler().fit_transform(Xr)

ks      = np.arange(2, 6)
inertia = silh = bic = aic = None

if os.path.exists(METRICS_FILE):
    # load previous sweep
    data    = np.load(METRICS_FILE)
    ks       = data['ks']
    inertia  = data['inertia']
    silh     = data['silhouette']
    bic      = data['bic']
    aic      = data['aic']
    print(f"✔︎ Loaded cached metric sweep from {METRICS_FILE}")
else:
    # compute and cache
    inertia = []
    silh    = []
    bic     = []
    aic     = []

    print(f"→ Performing metric sweep k={ks.tolist()} on {SAMP} pts")
    t0 = time.time()
    for k in tqdm(ks, desc="metric-k"):
        t1 = time.time()
        mb = MiniBatchKMeans(n_clusters=int(k), random_state=0, batch_size=10000).fit(Xs)
        inertia.append(mb.inertia_)
        silh.append(silhouette_score(Xs, mb.labels_))

        gm = GaussianMixture(n_components=int(k), random_state=0).fit(Xs)
        bic.append(gm.bic(Xs))
        aic.append(gm.aic(Xs))

        print(f"  k={int(k):2d} in {time.time()-t1:4.1f}s")
    print(f"✔︎ Metric sweep done in {time.time()-t0:.1f}s")

    # cache to disk
    np.savez_compressed(
        METRICS_FILE,
        ks=ks,
        inertia=np.array(inertia),
        silhouette=np.array(silh),
        bic=np.array(bic),
        aic=np.array(aic)
    )
    print(f"✔︎ Cached metric results to {METRICS_FILE}")

# plot metrics and save
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax[0,0].plot(ks, inertia, 'o-'); ax[0,0].set(title='Inertia',      xlabel='k')
ax[0,1].plot(ks, silh,    'o-'); ax[0,1].set(title='Silhouette',    xlabel='k')
ax[1,0].plot(ks, bic,     'o-'); ax[1,0].set(title='GMM BIC',       xlabel='k')
ax[1,1].plot(ks, aic,     'o-'); ax[1,1].set(title='GMM AIC',       xlabel='k')
plt.tight_layout()
plt.savefig("metric_sweep.png", dpi=300)
plt.close()
print("✔︎ Metric plots saved to metric_sweep.png")

# --------------------------------------------------------------------------------
# 3b-iii) CLUSTER MICRO → FINAL (fixed FINAL_K) with caching & visualization
# --------------------------------------------------------------------------------
CLUSTERS_FILE = 'seed_clusters.csv'
PLOT_2D       = 'clusters_2d.png'
PLOT_3D       = 'clusters_3d.html'

if os.path.exists(CLUSTERS_FILE):
    # 1) load previous assignment
    df_cluster = pd.read_csv(CLUSTERS_FILE)
    df['Cluster'] = df_cluster['Cluster'].values
    print(f"✔︎ Loaded existing clusters from {CLUSTERS_FILE}")
else:
    # 2) perform clustering
    t0 = time.time()

    # sanitize infinities
    col_means = np.nanmean(np.where(np.isfinite(micro_pts), micro_pts, np.nan), axis=0)
    micro_pts = np.nan_to_num(
        micro_pts,
        nan=col_means[np.newaxis, :],
        posinf=col_means[np.newaxis, :],
        neginf=col_means[np.newaxis, :]
    )

    # cluster micros
    mb_micro = MiniBatchKMeans(
        n_clusters=MICRO_K,
        random_state=0,
        batch_size=2048
    )
    m_lbl = mb_micro.fit_predict(micro_pts)
    cent  = mb_micro.cluster_centers_

    # final clusters
    mb_fin = MiniBatchKMeans(
        n_clusters=FINAL_K,
        random_state=0,
        batch_size=2048
    )
    f_lbl = mb_fin.fit_predict(cent)

    # majority‐vote seed→cluster
    df_sm = pd.DataFrame({
        'Seed':      micro_sds,
        'micro_lbl': m_lbl
    })
    df_sm['cluster'] = df_sm['micro_lbl'].map(dict(enumerate(f_lbl)))

    seed_cluster = df_sm.groupby('Seed')['cluster'] \
        .agg(lambda labs: Counter(labs).most_common(1)[0][0])

    df['Cluster'] = df['Seed'].map(seed_cluster)

    # 3) save the mapping
    df[['Seed','Cluster']].to_csv(CLUSTERS_FILE, index=False)
    print(f"✔︎ Section 3b done in {time.time()-t0:.1f}s — saved {CLUSTERS_FILE}")

# --------------------------------------------------------------------------------
# 3b-iv) VISUALIZE & SAVE CLUSTERS
# --------------------------------------------------------------------------------
# sample for speed
vis = df.sample(n=min(len(df), PLOT_SAMPLE), random_state=0)

# 2D plot (unchanged)
plt.figure(figsize=(8,6))
for cl in sorted(vis['Cluster'].unique()):
    sub = vis[vis['Cluster'] == cl]
    plt.scatter(sub['H_start'], sub['H_spike'], s=20, alpha=0.6, label=f'Cluster {cl}')
plt.title('Ĥ_start vs Ĥ_spike by Cluster')
plt.xlabel('Ĥ_start')
plt.ylabel('Ĥ_spike')
plt.legend(markerscale=1.5, fontsize='small', loc='upper left')
plt.tight_layout()
plt.savefig(PLOT_2D, dpi=300)
plt.close()
print(f"✔︎ 2D cluster plot saved to {PLOT_2D}")

# 3D interactive with embedded JS
fig3d = go.Figure()
for cl in sorted(vis['Cluster'].unique()):
    sub = vis[vis['Cluster'] == cl]
    if len(sub):
        fig3d.add_trace(go.Scatter3d(
            x=sub['H_start'],
            y=sub['H_spike'],
            z=sub['piE_norm'],
            mode='markers',
            marker=dict(size=4, opacity=0.7),
            name=f'Cluster {cl}',
            hovertemplate=(
                "Seed=%{customdata[0]}<br>"
                "Ĥ_start=%{x:.4f}<br>"
                "Ĥ_spike=%{y:.4f}<br>"
                "πE_norm=%{z:.4f}"
            ),
            customdata=sub[['Seed']].values
        ))

fig3d.update_layout(
    title="3D Cluster Visualization",
    scene=dict(
        xaxis_title="Ĥ_start",
        yaxis_title="Ĥ_spike",
        zaxis_title="πE_norm"
    ),
    legend_title="Cluster ID",
    margin=dict(l=0, r=0, t=40, b=0),
    template="plotly_white"
)

# **Embed** the entire Plotly.js bundle so it works offline
fig3d.write_html(
    PLOT_3D,
    include_plotlyjs=True,
    full_html=True
)
print(f"✔︎ 3D cluster plot saved to {PLOT_3D}")

# --------------------------------------------------------------------------------
# 3b-v) PLOT CLUSTER ASSIGNMENT VS. SEED VALUE
# --------------------------------------------------------------------------------
SEED_CLUSTER_PLOT = 'cluster_by_seed.png'

# sample for speed (you can remove the .sample if you want the full million)
vis = df.sample(n=min(len(df), PLOT_SAMPLE), random_state=0)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    vis['Seed'],
    vis['Cluster'],
    c=vis['Cluster'],
    cmap='tab10',
    s=8,
    alpha=0.6
)
plt.title('Cluster Assignment vs. Seed Value')
plt.xlabel('Seed')
plt.ylabel('Cluster ID')
plt.yticks(sorted(df['Cluster'].unique()))
plt.colorbar(scatter, ticks=sorted(df['Cluster'].unique()), label='Cluster ID')
plt.tight_layout()
plt.savefig(SEED_CLUSTER_PLOT, dpi=300)
plt.close()

print(f"✔︎ Cluster‐by‐seed plot saved to {SEED_CLUSTER_PLOT}")

# --------------------------------------------------------------------------------
# 3b-vi) PLOT CLUSTER ASSIGNMENT VS. SEED VALUE - 2 (CACHED UMAP & TIMING & 3D)
# --------------------------------------------------------------------------------
# Filnavne til model- og embeddings-cache
UMAP_MODEL_FILE = 'umap_3d_reducer.pkl'
EMB_CACHE_FILE = 'umap_3d_emb.npz'

# 1) Prepare features in float32 and scale
# 1) Pull out a contiguous float32 array (n_rows × n_features)
X = df[FEATURES].to_numpy(dtype=np.float32, copy=True)

# 2) In‐place replace infs and NaNs with zero
#    (very memory‐efficient: no extra copies beyond X itself)
np.nan_to_num(
    X,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
    copy=False
)

# 3) Scale
X_all = StandardScaler().fit_transform(X)


# 2) Downsample before UMAP to limit memory usage
n_vis = min(len(X_all), PLOT_SAMPLE)
idx = np.random.choice(len(X_all), size=n_vis, replace=False)
X = X_all[idx]

# 3) Load cached UMAP model & embeddings or fit anew
if os.path.exists(UMAP_MODEL_FILE) and os.path.exists(EMB_CACHE_FILE):
    # Load existing model + embedding
    reducer = joblib.load(UMAP_MODEL_FILE)
    cache = np.load(EMB_CACHE_FILE)
    idx_cached = cache['idx']
    emb = cache['emb']
    idx = idx_cached  # ensure idx aligns
    print(f"✔︎ Loaded UMAP model and embeddings ({len(idx)} samples)")
else:
    # Fit UMAP
    start_umap = time.time()
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=10,
        min_dist=0.6,
        metric='cosine',
        random_state=42,
        n_jobs=1,
        verbose=True,
        low_memory=True         # ← trades speed for a much smaller RAM footprint
    )


    emb = reducer.fit_transform(X)
    elapsed = time.time() - start_umap
    print(f"✔︎ UMAP 3D embedding done in {elapsed:.1f}s ({n_vis} samples)")

    # Cache model + embeddings
    joblib.dump(reducer, UMAP_MODEL_FILE)
    np.savez(EMB_CACHE_FILE, idx=idx, emb=emb)
    print(f"✔︎ Cached UMAP model to '{UMAP_MODEL_FILE}' and embeddings to '{EMB_CACHE_FILE}'")

# 4) Map embedding back to sampled DataFrame
df_vis = df.iloc[idx].copy()
df_vis['UMAP1'], df_vis['UMAP2'], df_vis['UMAP3'] = emb.T

# 5a) Interactive Plotly 3D scatter

fig3d = go.Figure()
for cl in sorted(df_vis['Cluster'].unique()):
    sub = df_vis[df_vis['Cluster'] == cl]
    fig3d.add_trace(go.Scatter3d(
        x=sub['UMAP1'], y=sub['UMAP2'], z=sub['UMAP3'],
        mode='markers', marker=dict(size=4, opacity=0.7),
        name=f"Cluster {cl}",
        hovertemplate=(
            "Seed=%{customdata[0]}<br>"
            "UMAP1=%{x:.4f}<br>"
            "UMAP2=%{y:.4f}<br>"
            "UMAP3=%{z:.4f}"
        ),
        customdata=sub[['Seed']].values
    ))
fig3d.update_layout(
    title="3D UMAP Projection of Collatz Seeds (sampled)",
    scene=dict(xaxis_title="UMAP1", yaxis_title="UMAP2", zaxis_title="UMAP3"),
    width=800, height=600, legend_title="Cluster"
)
fig3d.write_html("clusters_umap_3d_interactive.html", include_plotlyjs='cdn')
print("✔︎ Sampled UMAP 3D plot saved to clusters_umap_3d_interactive.html")

# 5b) 2D seed-vs-cluster rail plot (Jittered)
plt.figure(figsize=(12, 4))
for cl in sorted(df_vis['Cluster'].unique()):
    sub = df_vis[df_vis['Cluster'] == cl]
    y = np.full(len(sub), cl) + (np.random.randn(len(sub)) * 0.05)
    plt.scatter(sub['Seed'], y, s=5, alpha=0.5, label=f"Cluster {cl}")

plt.yticks(range(FINAL_K), [f"Cluster {c}" for c in range(FINAL_K)])
plt.xlabel("Seed")
plt.ylabel("Cluster")
plt.title("Seed → Cluster assignment (sampled)")
plt.legend(ncol=FINAL_K, fontsize='small', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("clusters_2d_rail.png", dpi=300)
plt.close()
print("✔︎ Sampled 2D seed-vs-cluster rail plot saved to clusters_2d_rail.png")

# --------------------------------------------------------------------------------
# 4) PARITY-LAW CLASSIFICATION (with Law 2 first)
# --------------------------------------------------------------------------------
# 4) PARITY-LAW CLASSIFICATION (with Law 2 first)
powers       = {s for s in df['Seed'] if (s & (s-1)) == 0 and s >= 8}
plateau_vals = set(df['H_start'].round(6))
law_groups   = {i: [] for i in range(1,7)}

# iterate row‐wise without blowing memory
for r in df.itertuples(index=False):
    s, h0, hsp = r.Seed, r.H_start, r.H_spike

    if s in powers and h0 == 0.0 and hsp == 0.0:
        law_groups[2].append(s)
    elif h0 == 0.0:
        law_groups[1].append(s)
    elif round(h0,6) in plateau_vals and h0 > 0.0:
        law_groups[3].append(s)
    elif hsp < h0:
        law_groups[4].append(s)
    else:
        law_groups[5].append(s)

law_groups[6] = [1]  # parity-neutral equilibrium


# --------------------------------------------------------------------------------
# 5) INTERACTIVE PLOTS FOR CLUSTERS & FEATURES (clusters plotted individually)
# --------------------------------------------------------------------------------
df_sample = df.sample(n=min(len(df), PLOT_SAMPLE), random_state=0)
cluster_ids = sorted(df['Cluster'].unique())

# 2D: πE_norm vs Ĥ_spike
fig = go.Figure()
for cl in cluster_ids:
    sub = df_sample[df_sample['Cluster'] == cl]
    fig.add_trace(go.Scatter(
        x=sub['piE_norm'],
        y=sub['H_spike'],
        mode='markers',
        name=f"Cluster {cl} ({len(sub)} pts)",
        marker=dict(size=6),
        hovertemplate="Seed=%{customdata[0]}<br>πE_norm=%{x:.4f}<br>Ĥ_spike=%{y:.4f}",
        customdata=sub[['Seed']].values
    ))
fig.update_layout(
    title="Sampled Elastic–π Norm vs. Ĥ_spike by k-Means Cluster",
    xaxis_title="Elastic–π Norm",
    yaxis_title="Ĥ_spike",
    legend_title="Cluster ID",
    template="plotly_white"
)
fig.write_html("interactive_clusters.html", include_plotlyjs='cdn')

# 3D: (Ĥ_start, Ĥ_spike, πE_norm)
fig3 = go.Figure()
for cl in cluster_ids:
    sub = df_sample[df_sample['Cluster'] == cl]
    fig3.add_trace(go.Scatter3d(
        x=sub['H_start'],
        y=sub['H_spike'],
        z=sub['piE_norm'],
        mode='markers',
        name=f"Cluster {cl} ({len(sub)} pts)",
        marker=dict(size=4),
        hovertemplate="Seed=%{customdata[0]}<br>Ĥ_start=%{x:.4f}"
                      "<br>Ĥ_spike=%{y:.4f}<br>πE_norm=%{z:.4f}",
        customdata=sub[['Seed']].values
    ))
fig3.update_layout(
    title="3D Entropy Features by k-Means Cluster",
    scene=dict(
        xaxis_title="Ĥ_start",
        yaxis_title="Ĥ_spike",
        zaxis_title="πE_norm"
    ),
    legend_title="Cluster ID",
    template="plotly_white"
)
fig3.write_html("interactive_clusters_3d.html", include_plotlyjs='cdn')

# Ĥ_start vs Ĥ_spike
fig2 = go.Figure()
for cl in cluster_ids:
    sub = df_sample[df_sample['Cluster'] == cl]
    fig2.add_trace(go.Scatter(
        x=sub['H_start'],
        y=sub['H_spike'],
        mode='markers',
        name=f"Cluster {cl} ({len(sub)} pts)",
        marker=dict(size=6),
        hovertemplate="Seed=%{customdata[0]}<br>Ĥ_start=%{x:.4f}<br>Ĥ_spike=%{y:.4f}",
        customdata=sub[['Seed']].values
    ))
fig2.update_layout(
    title="Sampled Ĥ_start vs. Ĥ_spike by k-Means Cluster",
    xaxis_title="Ĥ_start",
    yaxis_title="Ĥ_spike",
    legend_title="Cluster ID",
    template="plotly_white"
)
fig2.write_html("interactive_cluster_features.html", include_plotlyjs='cdn')

# πE_norm histogram by cluster
fig4 = go.Figure()
for cl in cluster_ids:
    sub = df_sample[df_sample['Cluster'] == cl]
    fig4.add_trace(go.Histogram(
        x=sub['piE_norm'],
        name=f"Cluster {cl} ({len(sub)} pts)",
        opacity=0.75,
        nbinsx=50
    ))
fig4.update_layout(
    title="Sampled Elastic–π Norm by k-Means Cluster",
    xaxis_title="πE_norm",
    yaxis_title="Count",
    barmode='overlay',
    legend_title="Cluster ID",
    template="plotly_white"
)
fig4.write_html("interactive_cluster_norms.html", include_plotlyjs='cdn')

print("✔︎ Cluster feature visualizations done.")

# --------------------------------------------------------------------------------
# 6) SUBSAMPLED-CLUSTER TRAJECTORIES
# --------------------------------------------------------------------------------
n_samples = 10
max_steps = 50
traj_records = []
for cl in range(4):
    seeds = df[df['Cluster']==cl]['Seed']
    samp  = seeds.sample(n=min(n_samples, len(seeds)), random_state=0)
    for s in samp:
        seq    = collatz(int(s))
        H_list = compute_H_list(seq)
        clean  = [h for h in H_list if not math.isnan(h)][:max_steps]
        for i,h in enumerate(clean):
            traj_records.append({'cluster':f'Cluster {cl}','step':i,'Ĥ':h,'Seed':s})

df_traj = pd.DataFrame(traj_records)
px.line(
    df_traj,
    x='step',
    y='Ĥ',
    color='cluster',
    line_group='Seed',
    title=f"Sampled Ĥ Trajectories (first {max_steps} steps) by Cluster",
    labels={
        'step':'Step',
        'Ĥ':'Ĥ value',
        'cluster':'Cluster ID'
    }
).update_layout(
    legend_title_text='Cluster ID'
).write_html("interactive_cluster_trajectories.html", include_plotlyjs='cdn')

print("✔︎ Cluster trajectory visualizations done.")

# --------------------------------------------------------------------------------
# 7) INTERACTIVE PLOTS FOR LAWS 2–6
# --------------------------------------------------------------------------------
df['ΔH'] = df['H_spike'] - df['H_start']

# Law 2: Dyadic Immediacy (pure evens)
law2_seeds = law_groups[2]
if law2_seeds:
    df2 = pd.DataFrame({'Seed': law2_seeds})
    df2['Ĥ_value'] = 0.0
    px.scatter(
        df2,
        x='Seed',
        y='Ĥ_value',
        title="Law 2 – Dyadic Immediacy",
        labels={
            'Seed':'n',
            'Ĥ_value':'Ĥ_start=Ĥ_spike=0'
        }
    ).update_layout(
        legend_title_text='Powers-of-two ≥8'
    ).write_html("interactive_law2.html", include_plotlyjs='cdn')
else:
    print("Warning: no Law 2 seeds found!")

# Law 3: Start Plateaux
px.scatter(
    df,
    x='Seed',
    y='H_start',
    title="Law 3 – Start Plateaux",
    labels={'Seed':'n','H_start':'Ĥ_start'}
).update_layout(
    legend_title_text='17 distinct Ĥ_start > 0'
).write_html("interactive_law3.html", include_plotlyjs='cdn')

# Law 4: Monotone Spike (ΔĤ)
px.scatter(
    df,
    x='Seed',
    y='ΔH',
    title="Law 4 – Monotone Spike (Ĥ_spike–Ĥ_start)",
    labels={'Seed':'n','ΔH':'ΔĤ'}
).add_hline(y=0,line_dash="dash").update_layout(
    legend_title_text='ΔĤ ≥ 0'
).write_html("interactive_law4.html", include_plotlyjs='cdn')

# Law 5: Elastic–π Norm Clustering
px.histogram(
    df,
    x='piE_norm',
    nbins=50,
    title="Law 5 – Elastic–π Norm Clustering",
    labels={'piE_norm':'πE_norm'}
).update_layout(
    legend_title_text='4 clusters'
).write_html("interactive_law5.html", include_plotlyjs='cdn')

# Law 6: Parity Neutrality of 1
df6 = pd.DataFrame({
    'n':      list(range(0,11)),
    'Parity': ['even' if (v!=1 and v%2==0)
               else 'odd' if (v!=1 and v%2==1)
               else 'neutral'
               for v in range(11)]
})
df6['Code'] = df6['Parity'].map({'even':0,'odd':1,'neutral':2})
px.scatter(
    df6,
    x='n',
    y='Code',
    color='Parity',
    title="Law 6 – Parity Neutrality of 1",
    labels={'n':'n','Code':'Parity Code'}
).update_layout(
    legend_title_text='0=even,1=odd,2=neutral'
).update_yaxes(
    tickmode='array',tickvals=[0,1,2],ticktext=['even','odd','neutral']
).write_html("interactive_law6.html", include_plotlyjs='cdn')

print("✔︎ Law visualizations done.")

# --------------------------------------------------------------------------------
# 8) STATIC SPIRALS (CLUSTERS & LAWS)
# --------------------------------------------------------------------------------
def spiral_clusters(df, filename="spiral_clusters.png"):
    groups  = {i: df[df['Cluster']==i]['Seed'].tolist() for i in range(4)}
    max_s   = df['Seed'].max()
    n_turns = 5
    fig, ax = plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True))

    # guide spiral
    θ = np.linspace(0,2*np.pi*n_turns,1000)
    r = θ/(2*np.pi*n_turns)
    ax.plot(θ, r,color='gray',lw=0.5,alpha=0.3,label='guide')

    for cid,seeds in groups.items():
        samp = random.sample(seeds,min(len(seeds),PLOT_SAMPLE))
        t    = np.array(samp)/max_s
        off  = cid*(2*np.pi/4)
        ang  = off + t*(2*np.pi*n_turns)+np.random.normal(0,0.02,len(t))
        ax.scatter(ang,t,s=10,alpha=0.7,label=f'Cluster {cid}')

    ax.set_title("4-Arm Spiral: Clusters")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename,dpi=300)
    plt.close()

spiral_clusters(df)

# --------------------------------------------------------------------------------
# 9) INTERACTIVE 3D SPIRAL (Clusters)
# --------------------------------------------------------------------------------
def interactive_spiral_3d(df, filename="interactive_spiral_3d.html"):
    df_samp = df.sample(n=min(len(df), PLOT_SAMPLE), random_state=0)
    xs, ys, zs, cs = [], [], [], []
    # no huge array allocation under the hood
    for r in df_samp.itertuples(index=False):
        s, cl = r.Seed, r.Cluster
        t = s / df['Seed'].max()
        off = cl * (2 * math.pi / 4)
        ang = off + t * (2 * math.pi * 5) + random.gauss(0, 0.02)
        xs.append(t * math.cos(ang))
        ys.append(t * math.sin(ang))
        zs.append(s)
        cs.append(str(cl))

    d3 = pd.DataFrame({'x': xs, 'y': ys, 'z': zs, 'cluster': cs})
    fig = px.scatter_3d(
        d3, x='x', y='y', z='z', color='cluster',
        title="3D Spiral: Clusters",
        labels={'x':'X','y':'Y','z':'Seed','cluster':'Cluster'}
    )
    fig.write_html(filename, include_plotlyjs='cdn')


# --------------------------------------------------------------------------------
# 10) SAVE CSVs
# --------------------------------------------------------------------------------
df.to_csv('collatz_entropy_full_results.csv',index=False)
df[['Seed','Cluster']].to_csv('seed_clusters.csv',index=False)
law_rows=[]
for law, seeds in law_groups.items():
    for s in seeds:
        law_rows.append({'Seed':s,'Law':law})
pd.DataFrame(law_rows).to_csv('seed_parity_laws.csv',index=False)
print("✔︎ All CSVs written.")

# --------------------------------------------------------------------------------
# 11) SHAPE-TWIN SCANNER FOR (3,27,31)
# --------------------------------------------------------------------------------
def compute_features(n):
    seq      = collatz(n)
    Htraj    = compute_H_list(seq)
    conv_idx = next(i for i, v in enumerate(seq) if v in LOOP)

    first_valid = None
    max_i, max_val = None, -float('inf')
    even = odd = 0

    # Single pass: count parity, record first valid index and max spike index
    for i, v in enumerate(seq[:conv_idx+1]):
        if v % 2 == 0:
            even += 1
        else:
            odd += 1

        h = Htraj[i]
        if not math.isnan(h):
            if first_valid is None:
                first_valid = i
            if h > max_val:
                max_val, max_i = h, i

    H_start = Htraj[first_valid] if first_valid is not None else 0.0
    H_spike = Htraj[max_i]       if max_i      is not None else 0.0

    cnts = {'even': even, 'odd': odd}
    try:
        piE_norm = abs(piE(S_i(cnts, 'even'))) + abs(piE(S_i(cnts, 'odd')))
    except ZeroDivisionError:
        piE_norm = 0.0

    ΔH = H_spike - H_start
    return (H_start, H_spike, piE_norm, ΔH)

refs = {s: compute_features(s) for s in (3, 27, 31)}
TOL  = 1e-6

def plot_shape_twins(df, twins):
    twin_seeds = [n for n, _ in twins]
    plt.figure(figsize=(6,6))
    plt.scatter(df['H_start'], df['H_spike'],
                s=10, alpha=0.3, label='All seeds')
    twin_df = df[df['Seed'].isin(twin_seeds)]
    plt.scatter(twin_df['H_start'], twin_df['H_spike'],
                s=60, c='C3', marker='*', label='Shape-twins')
    plt.xlabel('Ĥ_start')
    plt.ylabel('Ĥ_spike')
    plt.title('Shape-Twin Seeds in Feature–Space')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_twin_trajectories(twins, n_others=3, max_steps=50):
    twin_seeds = [n for n, _ in twins]
    others = random.sample(
        [s for s in df['Seed'] if s not in twin_seeds],
        min(n_others, len(df) - len(twin_seeds))
    )
    records = []
    selected = [(s, 'Twin') for s in twin_seeds[:n_others]] + [(s, 'Other') for s in others]
    for seed, label in selected:
        seq = collatz(seed)
        Ht  = compute_H_list(seq)
        clean = [h for h in Ht if not math.isnan(h)][:max_steps]
        for i, h in enumerate(clean):
            records.append({'Seed': seed, 'Step': i, 'Ĥ': h, 'Type': label})
    df_traj = pd.DataFrame(records)
    fig = px.line(
        df_traj,
        x='Step',
        y='Ĥ',
        color='Seed',
        line_dash='Type',
        title=f"Ĥ Trajectories: Twins vs. Others (first {max_steps} steps)"
    )
    fig.update_layout(legend_title_text='Seed (dash by Type)')
    fig.show()

SHAPE_TWIN_CACHE     = "shape_twins.pkl"
SHAPE_TWIN_CSV       = "shape_twins.csv"
SCANNED_SEEDS_CACHE  = "scanned_shape_seeds.pkl"

_scanned_seeds = set()

def load_shape_twin():
    global _scanned_seeds
    if os.path.exists(SHAPE_TWIN_CACHE):
        with open(SHAPE_TWIN_CACHE, "rb") as f:
            twins = pickle.load(f)
    else:
        initial = {s: compute_features(s) for s in (3, 27, 31)}
        twins = [(s, feat) for s, feat in initial.items()]

    if os.path.exists(SCANNED_SEEDS_CACHE):
        with open(SCANNED_SEEDS_CACHE, "rb") as f:
            _scanned_seeds = pickle.load(f)
    else:
        _scanned_seeds = set()

    return twins

def save_shape_twin(twins):
    with open(SHAPE_TWIN_CACHE, "wb") as f:
        pickle.dump(twins, f)
    if twins:
        df_tw = pd.DataFrame([
            {'Seed': seed, **{f'feat_{i}': v for i, v in enumerate(profile)}}
            for seed, profile in twins
        ])
        df_tw.to_csv(SHAPE_TWIN_CSV, index=False)
    with open(SCANNED_SEEDS_CACHE, "wb") as f:
        pickle.dump(_scanned_seeds, f)

def find_shape_twins(max_seed, tol=TOL, commit_every=10000):
    # load existing twins and build refs list
    twins = load_shape_twin()
    refs  = [feats for _, feats in twins]

    # bytearray flag array ≃1 byte/seed for O(1) membership
    scanned = bytearray(max_seed+1)
    for s, _ in twins:
        if s <= max_seed:
            scanned[s] = 1

    counter = 0
    for n in tqdm(range(3, max_seed+1, 2), desc="Shape-twin scan"):
        if scanned[n]:
            continue
        scanned[n] = 1

        feats = compute_features(n)

        # compare once per ref, break early on match
        for r in refs:
            if all(abs(f - rr) <= tol for f, rr in zip(feats, r)):
                twins.append((n, feats))
                refs.append(feats)
                break

        counter += 1
        if counter % commit_every == 0:
            save_shape_twin(twins)
            gc.collect()

    save_shape_twin(twins)
    return twins

if __name__ == "__main__":
    MAX_SCAN = 10**8
    results = find_shape_twins(MAX_SCAN)

    if results:
        print("Found shape-twins up to", MAX_SCAN)
        print("   n     H_start    H_spike    piE_norm       ΔH")
        for n, (h0, hsp, nrm, dh) in results:
            print(f"{n:7d}  {h0:9.6f}  {hsp:9.6f}  {nrm:9.6f}  {dh:9.6f}")
        plot_shape_twins(df, results)
        plot_twin_trajectories(results)
    else:
        print("No shape-twins found up to", MAX_SCAN)
