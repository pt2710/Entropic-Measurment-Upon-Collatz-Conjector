# Entropic Measurement Upon Collatz Conjector 🚀

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15724259.svg)](https://doi.org/10.5281/zenodo.15724259)

This project implements a novel **entropy-based framework** for the Collatz conjecture. Each orbit’s parity sequence is encoded via a two-feature *Parity-Adapted Dynamic Fluctuation Index (pDFI)* and an **elastic–π** transformation. Together these define a Lyapunov-style stability functional \(\widetilde{H}(n,t)\) that provably **never increases on even steps and strictly decreases on odd-even pairs**, forcing every orbit to converge to {1,2,4}. In large-scale tests (up to \(10^6\) seeds) this method uncovers four fundamental parity laws (including the even-bias and 1-neutrality) and shows that the elastic–π norms cluster into exactly **four attractor groups** under k-means. In short, our approach replaces the usual “chaotic parity” model with a rigorous **entropy–parity algebra** that drives all orbits to the known 1–2–4 cycle.

## Methodology 📊

*Figure: Spiral cluster of parity-adjusted entropic features (illustrative). The repository’s analysis shows that Collatz seeds form fractal “spiral” clusters in the parity–entropy space.* Each orbit’s even/odd fluctuations are captured by the pDFI (a two-dimensional feature vector). Applying the elastic–π phase transform yields an **elastic–π norm** π_E for each number, and the stability functional


H̃(n,t) = H(n) · (|π_E1(t)| + |π_E2(t)|)


is constructed to strictly decrease with each odd–even step. This analytically forces all orbits into the trivial cycle. Remarkably, plotting the elastic–π norms of seeds reveals four distinct attractor clusters (with spiral-like geometry) under k-means clustering. These clusters greatly simplify the parity–entropy landscape and confirm the underlying “evenness bias” in Collatz orbits.

## Visualizations 🎥

*Figure: Example cluster visualization. Interactive 2D/3D scatterplots and UMAP embeddings are provided in the repo to explore how orbits group in parity–entropy space.* We include a suite of interactive visualization outputs. In particular, the repository provides **interactive 2D and 3D scatter plots** of the clusters, as well as a **3D UMAP embedding** of the parity-entropic features. These allow the user to pan/zoom and inspect cluster structure. The fully interactive HTML files (viewable in a browser) are listed below for convenience:

* [Interactive 2D Clusters](interactive_clusters.html)
* [Interactive 3D Clusters](interactive_clusters_3d.html)
* [Interactive UMAP 3D Clusters](clusters_umap_3d_interactive.html)
* [Cluster Trajectories (2D)](interactive_cluster_trajectories.html)
* [Interactive Cluster Features](interactive_cluster_features.html)
* [Cluster Norms Over Time](interactive_cluster_norms.html)

## Installation 🔧

1. **Clone the repository:**

```bash
git clone https://github.com/pt2710/Entropic-Measurment-Upon-Collatz-Conjector.git
cd Entropic-Measurment-Upon-Collatz-Conjector
```

2. **Install dependencies:** (requires Python 3.x)

```bash
pip install -r requirements.txt
```

3. **Run the simulation:** Use the provided scripts to generate and analyze Collatz orbits. For example, run the main simulation for \(N\) seeds:

```bash
python simulate_collatz.py --max-seed 1000000
```

4. **Analyze results:** The script `analyze_clusters.py` (or similar) performs clustering and generates the visualizations. Finally, open the resulting HTML files in the `interactive/` folder in a web browser to explore the clusters.

## Project Structure 📁

* `simulate_collatz.py`, `entropy.py`, `clustering.py` – Core modules for computing orbits, entropic features, and clustering.
* `interactive/` – Output directory containing all HTML/JSON visualization files.
* `requirements.txt` – Lists Python dependencies (e.g. NumPy, SciPy, scikit-learn, Plotly).
* `tests/` – Unit tests for key functions (parity encoding, DFI computation, etc.).
* `Notebooks/` – Jupyter notebooks demonstrating exploratory analysis.

## License and Citation 📜

This project is licensed under the **Creative Commons Attribution 4.0 (CC BY 4.0)** license. Users are free to use, modify, and distribute the code and data with proper attribution.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15724259.svg)](https://doi.org/10.5281/zenodo.15724259)

---

📧 **Contact**: [thenothingnesseffect@gmail.com](mailto:thenothingnesseffect@gmail.com)
---

## 📄 Supplementary Project Metadata

This repository includes additional metadata and contribution guidelines:

- [`AUTHORS.md`](./AUTHORS.md) — Project authorship and contact details.
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — How to contribute code, issues, and improvements.
- [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) — Code of conduct for community members.
- [`SECURITY.md`](./SECURITY.md) — Responsible disclosure and vulnerability reporting.
- [`CITATION.cff`](./CITATION.cff) — Citation metadata for scholarly reference.
- [`MAINTAINERS.md`](./MAINTAINERS.md) — Maintainers responsible for this repository.

---