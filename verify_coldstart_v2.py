"""Independent result audit: identities, references, metrics, checkpoint replay."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rmdldpa.config import ModelConfig
from rmdldpa.data import load_dataset,sha256
from rmdldpa.cold_training import build_model
from rmdldpa.training import predict
from rmdldpa.metrics import classification_metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/coldstart_v2'
OLD=ROOT/'results/candidate_screen_20261109'

def main():
    torch.set_num_threads(4)
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha256(OUT/'retrieval_assets.npz')==protocol['assets_sha256']
    for p,h in protocol['source_hashes'].items():
        assert sha256(OUT/'source_snapshot'/p)==h,p
        assert sha256(ROOT/p)==h,p
    a=np.load(OUT/'retrieval_assets.npz'); groups=a['drug_groups']; eligible=set(a['eligible_drugs'].tolist())
    ds=load_dataset(ROOT/'data/processed/association.csv',ROOT/'data/processed/drug_embedding.csv',ROOT/'data/processed/disease_embedding_pubmed.csv')
    np.testing.assert_array_equal(ds.drug_features,a['drug_features'][:len(ds.drugs)])
    original_inputs=np.load(OLD/'prediction_arrays.npz')
    np.testing.assert_array_equal(original_inputs['drug_features'],a['drug_features'])
    np.testing.assert_array_equal(original_inputs['disease_features'],ds.disease_features)
    ds.drug_features=a['drug_features']; c=ModelConfig(**protocol['config']['model'])
    candidate_groups=set(groups[len(ds.drugs):])
    assert not set(groups[list(eligible)])&candidate_groups
    seen=set(); replay=[]; oof=pd.read_csv(OUT/'benchmark_oof.csv')
    summary=json.loads((OUT/'benchmark_summary.json').read_text())
    for fold in range(1,6):
        folder=OUT/'benchmark'/f'fold_{fold}'; split=json.loads((folder/'split.json').read_text())
        tr=set(split['train_drugs']); te=set(split['test_drugs'])
        assert tr|te==eligible and not tr&te
        assert not set(groups[list(tr)])&set(groups[list(te)])
        assert not seen&te; seen|=te
        reference=None
        for variant in protocol['variants']:
            sub=folder/variant; pred=pd.read_csv(sub/'predictions.csv')
            done=json.loads((sub/'complete.json').read_text()); assert sha256(sub/'model.pt')==done['checkpoint_sha256']
            checkpoint=torch.load(sub/'model.pt',map_location='cpu',weights_only=True)
            pairs=checkpoint['reference_pairs'].numpy(); labels=checkpoint['reference_labels'].numpy()
            assert set(pairs[:,0])<=tr
            np.testing.assert_array_equal(ds.associations[pairs[:,0],pairs[:,1]],labels)
            if reference is not None: np.testing.assert_array_equal(reference,np.column_stack((pairs,labels)))
            reference=np.column_stack((pairs,labels))
            assert set(pred.drug_index)==te and len(pred)==len(te)*len(ds.diseases)
            queries=pred[['drug_index','disease_index']].to_numpy()
            np.testing.assert_array_equal(ds.associations[queries[:,0],queries[:,1]],pred.label)
            observed=classification_metrics(pred.label,pred.score)
            stored=json.loads((sub/'metrics.json').read_text())['all_pairs']
            for key in stored: np.testing.assert_allclose(observed[key],stored[key],atol=1e-12,rtol=0)
            model=build_model(ds,c,a,variant); model.load_state_dict(checkpoint['model_state'],strict=True)
            # Replay all held-out predictions, not merely a convenient subset.
            reproduced=predict(model,ds,queries,pairs,labels,torch.device('cpu'))
            error=float(np.max(np.abs(reproduced-pred.score.to_numpy())))
            assert error<1e-7,(fold,variant,error)
            replay.append(dict(fold=fold,variant=variant,max_abs_error=error))
            print('Verified fold',fold,variant,flush=True)
    assert seen==eligible
    for variant in protocol['variants']:
        f=oof[oof.variant.eq(variant)]
        assert len(f)==len(eligible)*len(ds.diseases)
        assert not f.duplicated(['drug_index','disease_index']).any()
        for key,val in classification_metrics(f.label,f.score).items():
            np.testing.assert_allclose(val,summary[variant]['pooled_all_pairs'][key],atol=1e-12,rtol=0)
    arrays=np.load(OUT/'candidate_scores.npz'); frame=pd.read_csv(OUT/'candidate_comparison.csv')
    source=json.loads((OLD/'input.json').read_text(encoding='utf-8'))['drugs']
    assert len(frame)==len(source)==79
    assert frame.catalog.tolist()==[r['catalog'] for r in source]
    assert frame.loc[frame.CE_score.isna(),'catalog'].tolist()==['S6955']
    assert frame.loc[frame.out_of_domain_peptide,['CE_rank','AE_rank']].isna().all().all()
    for k,seed in enumerate([42,43,44]):
        paircheck=None
        for variant,subname,arrkey in [('original',f'original_seed_{seed}','control_scores'),('hybrid',f'seed_{seed}','scores')]:
            sub=OUT/'screening'/subname
            checkpoint=torch.load(sub/'model.pt',map_location='cpu',weights_only=True)
            assert sha256(sub/'model.pt')==json.loads((sub/'complete.json').read_text())['checkpoint_sha256']
            pairs=checkpoint['reference_pairs'].numpy(); labels=checkpoint['reference_labels'].numpy()
            assert set(pairs[:,0])<=eligible
            if paircheck is not None: np.testing.assert_array_equal(paircheck,np.column_stack((pairs,labels)))
            paircheck=np.column_stack((pairs,labels))
            model=build_model(ds,c,a,variant); model.load_state_dict(checkpoint['model_state'],strict=True)
            score=predict(model,ds,arrays['queries'],pairs,labels,torch.device('cpu')).reshape(-1,2)
            np.testing.assert_allclose(score,arrays[arrkey][k],atol=1e-7,rtol=0)
        neighbors=json.loads((OUT/'screening'/f'seed_{seed}'/'neighbors.json').read_text(encoding='utf-8'))
        for q in neighbors:
            assert not {groups[n['drug_index']] for n in q['neighbors']}&{groups[len(ds.drugs)+q['candidate_parent']]}
            assert len({(n['drug_index'],n['disease']) for n in q['neighbors']})==len(q['neighbors'])
    for r in frame.itertuples():
        if pd.isna(r.parent_group): continue
        for j,t in enumerate(['CE','AE']):
            scores=arrays['scores'][:,int(r.parent_group),j]; controls=arrays['control_scores'][:,int(r.parent_group),j]
            np.testing.assert_allclose(getattr(r,t+'_score'),scores.mean(),atol=1e-12,rtol=0)
            np.testing.assert_allclose(getattr(r,t+'_seed_sd'),scores.std(ddof=1),atol=1e-12,rtol=0)
            np.testing.assert_allclose(getattr(r,t+'_change'),scores.mean()-controls.mean(),atol=1e-12,rtol=0)
    for _,f in frame.dropna(subset=['parent_group']).groupby('parent_group'):
        assert f.CE_score.nunique()==f.AE_score.nunique()==1
    result=dict(passed=True,source_hashes_verified=True,fold_drug_group_disjointness=True,
        candidate_group_exclusion=True,matched_training_references=True,metric_recalculation=True,
        benchmark_checkpoint_replay=replay,all_six_candidate_models_replayed=True,
        duplicate_neighbor_exclusion=True,salt_parent_score_consistency=True,missing_input_not_zero=True)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
