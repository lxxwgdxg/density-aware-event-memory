# Tested software environments

The tabular validation and figure-generation route was tested with:

- Python 3.13.9
- NumPy 2.3.5
- pandas 2.3.3
- SciPy 1.16.3
- scikit-learn 1.7.2
- Matplotlib 3.10.6
- openpyxl 3.1.5
- PyYAML 6.0.3

The portable dependency files specify compatible lower bounds. The original score-building route also used PyTorch 2.5.1; the appropriate CPU or CUDA build should be selected for the host system. Figure-level reproduction does not require a GPU.
