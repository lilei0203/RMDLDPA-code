"""Recompute every fold metric and check splits, saved references and source hashes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rmdldpa.data import sha256
from rmdldpa.metrics import classification_metrics, summarize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args(argv)
    root = args.run_dir
    manifest = json.loads((root / 'run_manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'completed':
        raise ValueError('Run has not completed')
    for path, expected in manifest['source_sha256'].items():
        if sha256(root / 'source_snapshot' / path) != expected:
            raise ValueError(f'Source snapshot checksum mismatch: {path}')
    rows = pd.read_csv(root / 'fold_metrics.csv')
    run_means = []
    checked_folds = 0
    for run in summary['runs']:
        test_seen = set()
        all_pairs = None
        fold_metrics = []
        for fold in range(1, 6):
            directory = root / f'run_{run["run"]:02d}' / f'fold_{fold}'
            train = pd.read_csv(directory / 'train_pairs.csv')
            test = pd.read_csv(directory / 'test_pairs.csv')
            prediction = pd.read_csv(directory / 'predictions.csv')
            keys = ['drug_index', 'disease_index']
            train_keys = set(map(tuple, train[keys].to_numpy()))
            test_keys = set(map(tuple, test[keys].to_numpy()))
            if train_keys & test_keys or test_seen & test_keys:
                raise ValueError(f'Overlapping train/test folds in {directory}')
            if len(train_keys) != len(train) or len(test_keys) != len(test):
                raise ValueError('Duplicate pairs in a fold')
            if all_pairs is None:
                all_pairs = train_keys | test_keys
            if train_keys | test_keys != all_pairs:
                raise ValueError('Fold labeled pools differ within one run')
            test_seen |= test_keys
            pd.testing.assert_frame_equal(test, prediction.drop(columns=['probability']))
            checkpoint = directory / 'model.pt'
            if checkpoint.exists():
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                np.testing.assert_array_equal(state['reference_pairs'].numpy(), train[keys].to_numpy())
                np.testing.assert_array_equal(state['reference_labels'].numpy(), train.label.to_numpy())
                if not all(torch.isfinite(v).all() for v in state['model_state'].values()):
                    raise ValueError('Non-finite checkpoint parameters')
            metrics = classification_metrics(prediction.label, prediction.probability)
            saved = json.loads((directory / 'metrics.json').read_text())
            csv_row = rows[(rows.run == run['run']) & (rows.fold == fold)]
            if len(csv_row) != 1:
                raise ValueError('Duplicate or missing fold metrics')
            for name, value in metrics.items():
                np.testing.assert_allclose([saved[name], csv_row.iloc[0][name]], value, atol=1e-12, rtol=1e-12)
            history = pd.read_csv(directory / 'training.csv')
            if not np.isfinite(history.to_numpy()).all():
                raise ValueError('Non-finite training history')
            fold_metrics.append(metrics)
            checked_folds += 1
        if test_seen != all_pairs:
            raise ValueError('Held-out folds do not cover the full labeled pool')
        expected = summarize(fold_metrics)
        for statistic in ['mean', 'std']:
            for name, value in expected[statistic].items():
                np.testing.assert_allclose(run[statistic][name], value, atol=1e-12, rtol=1e-12)
        run_means.append(expected['mean'])
    expected = summarize(run_means)
    for statistic in ['mean', 'std']:
        for name, value in expected[statistic].items():
            actual = summary['across_run_means'][statistic][name]
            if value is None:
                if actual is not None:
                    raise ValueError('One-run standard deviation must be null')
            else:
                np.testing.assert_allclose(actual, value, atol=1e-12, rtol=1e-12)
    result = {'status': 'passed', 'folds_checked': checked_folds,
              'checks': ['source snapshot hashes', 'pair disjointness and full coverage', 'checkpoint reference corpus',
                         'finite parameters and training history', 'metrics recomputed from saved probabilities',
                         'CSV/JSON means and sample standard deviations']}
    (root / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
