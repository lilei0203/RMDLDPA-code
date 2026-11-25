"""Recompute saved inference and verify cohort exclusion, sources, and outputs."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from rmdldpa.config import ModelConfig
from rmdldpa.data import sha256
from rmdldpa.model import RMDLDPA
from rmdldpa.training import predict

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/candidate_screen_20261109'

def main():
    torch.set_num_threads(4)
    manifest=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    for name,h in manifest['hashes'].items():
        assert sha256(OUT/name)==h,name
    for name,h in manifest['source_hashes'].items():
        assert sha256(OUT/'source_snapshot'/name)==h,name
    source=json.loads((OUT/'input.json').read_text(encoding='utf-8'))
    frame=pd.read_csv(OUT/'predictions.csv')
    arrays=np.load(OUT/'prediction_arrays.npz')
    ds=SimpleNamespace(drug_features=arrays['drug_features'],disease_features=arrays['disease_features'])
    association=pd.read_csv(ROOT/'data/processed/association.csv',index_col=0)
    excluded=pd.read_csv(OUT/'excluded_training_drugs.csv')['index'].tolist()
    assert len(frame)==len(source['drugs'])==79
    assert frame.catalog.tolist()==[r['catalog'] for r in source['drugs']]
    assert frame.name.tolist()==[r['name'] for r in source['drugs']]
    assert frame.CE_score.notna().sum()==frame.AE_score.notna().sum()==78
    assert frame.loc[frame.CE_score.isna(),'catalog'].tolist()==['S6955']
    assert frame.loc[frame.out_of_domain_peptide,'catalog'].tolist()==['P1114']
    assert frame.loc[frame.out_of_domain_peptide,['CE_parent_rank','AE_parent_rank']].isna().all().all()
    errors=[]
    for k,seed in enumerate(manifest['seeds']):
        ckpt=torch.load(OUT/'models'/f'seed{seed}.pt',map_location='cpu',weights_only=True)
        pairs=ckpt['reference_pairs'].numpy()
        labels=ckpt['reference_labels'].numpy()
        assert not set(pairs[:,0])&set(excluded)
        assert not set(pairs[:,0])&set(arrays['queries'][:,0])
        assert len({tuple(x) for x in pairs})==len(pairs)
        truth=association.to_numpy()[pairs[:,0],pairs[:,1]]
        np.testing.assert_array_equal(truth,labels)
        assert sum(labels)==len(labels)/2==manifest['training_positives']
        model=RMDLDPA(ckpt['drug_dim'],ckpt['disease_dim'],ModelConfig(**ckpt['model_config']))
        model.load_state_dict(ckpt['model_state'],strict=True)
        values=predict(model,ds,arrays['queries'],pairs,labels,torch.device('cpu')).reshape(-1,2)
        np.testing.assert_allclose(values,arrays['scores'][k],rtol=0,atol=1e-7)
        errors.append(float(np.max(np.abs(values-arrays['scores'][k]))))
    for _,group in frame.dropna(subset=['parent_group']).groupby('parent_group'):
        assert group.CE_score.nunique()==group.AE_score.nunique()==1
    for r in frame.itertuples():
        if pd.notna(r.parent_group):
            idx=int(r.parent_group)
            np.testing.assert_allclose([r.CE_score,r.AE_score],arrays['scores'][:,idx,:].mean(0),rtol=0,atol=1e-12)
            np.testing.assert_allclose([r.CE_seed_sd,r.AE_seed_sd],arrays['scores'][:,idx,:].std(0,ddof=1),rtol=0,atol=1e-12)
    result=dict(passed=True,records=79,scored_records=78,excluded_drug_count=len(excluded),
                checkpoint_replay_max_abs_errors=errors,source_hashes_verified=True,
                reference_exclusion_verified=True,source_rows_preserved=True,salt_parent_scores_equal=True,
                missing_insulin_not_zero=True,peptide_separate_from_primary_ranking=True)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
