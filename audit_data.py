"""Check aligned data, archive hashes, and available SMILES coverage without guessing missing chemistry."""
from collections import defaultdict
import json
from pathlib import Path
import pandas as pd
from rmdldpa.data import load_dataset, sha256


def main():
    root = Path(__file__).resolve().parent
    data = root / 'data'
    dataset = load_dataset(data / 'processed/association.csv', data / 'processed/drug_embedding.csv',
                           data / 'processed/disease_embedding_pubmed.csv')
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        if sha256(root / row['path']) != row['sha256']:
            raise ValueError(f'Archive/input checksum mismatch: {row["path"]}')
    candidate = defaultdict(lambda: defaultdict(list))
    sources = [('drug_smiles.csv', 'name', 'isosmiles'), ('dis_durg_smiles.csv', 'Drug', 'smiles'),
               ('simple_smiles_embedding_li.csv', 'Drug', 'smiles')]
    for name, entity_column, smiles_column in sources:
        path = data / 'raw/additional_sources' / name
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype=str)
        for _, row in frame.iterrows():
            entity, smiles = row[entity_column], row[smiles_column]
            if pd.notna(entity) and pd.notna(smiles) and smiles.strip():
                candidate[entity.strip().casefold()][smiles.strip()].append(name)
    aligned, missing, ambiguous = [], [], []
    for drug in dataset.drugs:
        choices = candidate.get(drug.strip().casefold(), {})
        if len(choices) == 1:
            smiles, sources = next(iter(choices.items()))
            aligned.append({'Drug': drug, 'SMILES': smiles, 'Source': ';'.join(sorted(set(sources)))})
        elif len(choices) == 0:
            missing.append(drug)
        else:
            ambiguous.append({'Drug': drug, 'candidate_SMILES': list(choices)})
    (data / 'text').mkdir(exist_ok=True)
    pd.DataFrame(aligned, columns=['Drug', 'SMILES', 'Source']).to_csv(data / 'text/drug_smiles_available.csv', index=False)
    report = {'dataset': dataset.provenance, 'archived_files_verified': len(manifest['files']),
              'smiles_match_rule': 'strip + casefold name equality only; conflicting strings are not silently resolved',
              'smiles_matched_count': len(aligned), 'smiles_missing': missing, 'smiles_conflicts': ambiguous,
              'paper_reported_counts': {'drugs': 383, 'diseases': 51, 'positive_associations': 742},
              'paper_original_inputs_available': False,
              'available_disease_text_source': 'PubMed, not the paper PubChem corpus'}
    (data / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Validated {len(dataset.drugs)} drugs x {len(dataset.diseases)} diseases, '
          f'{dataset.associations.sum()} positives, {len(manifest["files"])} archive hashes.')
    print(f'SMILES exact-name matches={len(aligned)}, missing={len(missing)}, conflicts={len(ambiguous)}')
    print('Detailed report: ' + str(data / 'audit.json'))


if __name__ == '__main__':
    main()
