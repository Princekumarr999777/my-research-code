# Temporal Fairness Knowledge Tracing (UTFA)

**UTFA** (Temporal Fairness Knowledge Tracing) is a self‑contained research codebase that implements a suite of knowledge‑tracing models on the OULAD dataset with a focus on **temporal fairness** across gender and engagement groups.  The repository contains:

- A reproducible training pipeline that supports early‑stopping, dynamic‑lambda fairness control, adversarial GRL, re‑weighting baselines, and extensive evaluation.
- Automatic reporting that produces tables, figures, and a SHA‑256 manifest for result integrity.
- Parameter sweeps for lambda selection and split‑robustness analysis.

## Repository Structure
```
📦 my-research-code
├─ 📄 utfa_final.py          # Main script (executable entry point)
├─ 📄 README.md             # This document
└─ 📄 .gitignore            # Ignored files (e.g., data caches, checkpoints)
```

## Requirements
- Python ≥ 3.9
- PyTorch ≥ 1.13
- pandas, numpy, matplotlib
- (Optional) Google Colab for easy drive mounting

You can install the required packages via:
```bash
pip install torch pandas numpy matplotlib
```

## Usage
1. Clone the repository or pull the latest changes.
2. Ensure your OULAD CSV files are placed under a directory referenced by `DATA_DIR` in the script.
3. Run the pipeline:
```bash
python utfa_final.py
```
   The script will iterate over the two thresholds (`70` and `50`), train all methods, perform lambda sweeps, and generate a full report under `WORK_DIR`.

## License
This code is released under the MIT License. Feel free to adapt and extend it for your own research.
