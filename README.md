# RMDLDPA

Resources for **Retrieval-augmented Deep Learning Discovers Repurposed Drugs Against Parasitic Diseases**.

RMDLDPA predicts drug–parasitic disease associations by combining interaction-aware representations, historical evidence retrieval, and reliability-aware prediction.

## Files

- `association.csv`: drug–parasitic disease associations.
- `drug_embedding.csv` and `disease_embedding_pubmed.csv`: input features.
- `default.json`: model and training settings.
- `run_5cv.py`: five-fold cross-validation entry point.
- `encode_features.py`: feature encoding entry point.
- `verify_results.py`: result verification script.

## Dependencies

```bash
pip install -r requirements.txt
```

**Note:** The `rmdldpa` package imported by the scripts is currently missing from this repository and must be added before they can run.

## Data

See [RMDLDPA-data](https://github.com/lilei0203/RMDLDPA-data) for the associated data repository.
