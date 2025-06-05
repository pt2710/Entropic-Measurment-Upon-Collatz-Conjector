# Advanced Configuration

Most parameters used in the analysis can be adjusted directly in `entropic_collatz_conjector.py`. For convenience a sample configuration file is provided in `configs/default.yaml` which shows how custom values might be stored.

When modifying the sweep bounds or empirical law definitions keep in mind that large sweeps may take significant time to run. The script prints progress updates so you can monitor long runs.

To run the unit tests simply invoke:

```bash
pytest -q
```

The tests are lightweight and should finish quickly. If TensorFlow is missing the machine learning based tests in `test_seed.py` are skipped automatically.
