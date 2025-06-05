# Running the Scripts

The main analysis is performed by `entropic_collatz_conjector.py`. Execute it from the repository root:

```bash
python entropic_collatz_conjector.py
```

During execution you will see progress bars showing the sweep over starting seeds. Output files such as `collatz_entropy_results.csv`, various PNG images and interactive HTML plots will be generated in the repository directory.

If TensorFlow is installed you can also run `test_seed.py` for additional clustering experiments:

```bash
python test_seed.py
```
