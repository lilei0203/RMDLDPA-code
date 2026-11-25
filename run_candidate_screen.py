"""Drug-cold-start scoring for the user-supplied CE/AE candidate list.

All candidate-matching training drugs are removed from both positive and sampled
negative reference pairs. Diseases remain known. Predictions are uncalibrated
sigmoid scores under balanced training, not estimated clinical probabilities.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from rmdldpa.config import ModelConfig
from rmdldpa.data import load_dataset, sha256
from rmdldpa.training import fit_model, predict

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/candidate_screen_20261109'
TARGETS=['Cystic echinococcosis','Alveolar echinococcosis']

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    config=json.loads((ROOT/'configs/default.json').read_text())
    mc=ModelConfig(**config['model'])
    ds=load_dataset(ROOT/'data/processed/association.csv',ROOT/'data/processed/drug_embedding.csv',
                    ROOT/'data/processed/disease_embedding_pubmed.csv')
    candidates=json.loads((OUT/'candidates_encoded.json').read_text(encoding='utf-8'))
    cx=np.load(OUT/'candidate_features.npz')
    cohort_excluded=sorted({m['index'] for r in candidates for m in r['training_matches']})
    compatibility=pd.read_csv(OUT/'training_feature_compatibility.csv')
    unverified=compatibility.loc[compatibility.nearest_reencoded_max_abs_error>=1e-5,'Drug'].tolist()
    provenance_excluded=[ds.drugs.index(name) for name in unverified]
    excluded=sorted(set(cohort_excluded+provenance_excluded))
    retained=np.array([i for i in range(len(ds.drugs)) if i not in excluded])
    original_drug_count=len(ds.drugs)
    assert ds.drug_features.shape[1]==cx['features'].shape[1]
    ds.drug_features=np.concatenate([ds.drug_features,cx['features']])
    ds.drugs += [f'candidate_parent_{i:03}' for i in range(len(cx['features']))]
    target_ids=[ds.diseases.index(t) for t in TARGETS]
    queries=np.array([[original_drug_count+i,j] for i in range(len(cx['features'])) for j in target_ids],dtype=np.int64)
    allowed=np.zeros_like(ds.associations,dtype=bool)
    allowed[retained,:]=True
    positives=np.argwhere((ds.associations==1)&allowed)
    unknowns=np.argwhere((ds.associations==0)&allowed)
    model_dir=OUT/'models'
    model_dir.mkdir(exist_ok=True)
    pd.DataFrame([{'index':i,'Drug':ds.drugs[i],'reason':'candidate_parent' if i in cohort_excluded else 'unverified_feature_provenance'} for i in excluded]).to_csv(OUT/'excluded_training_drugs.csv',index=False)
    all_scores=[]
    histories=[]
    for seed in args.seeds:
        rng=np.random.default_rng(seed)
        negative=unknowns[rng.choice(len(unknowns),len(positives),replace=False)]
        pairs=np.concatenate([positives,negative])
        labels=np.concatenate([np.ones(len(positives)),np.zeros(len(negative))]).astype(np.int64)
        assert not set(pairs[:,0])&set(excluded)
        assert set(pairs[:,0])<=set(retained)
        assert not set(pairs[:,0])&set(queries[:,0])
        sample=pd.DataFrame(pairs,columns=['drug_index','disease_index'])
        sample['label']=labels
        sample.to_csv(OUT/f'train_pairs_seed{seed}.csv',index=False)
        print(f'Seed {seed}: {len(retained)} allowed training drugs; {len(positives)} positives + {len(negative)} sampled unknowns',flush=True)
        model,history=fit_model(ds,pairs,labels,mc,config['training'],seed,torch.device('cpu'),
            callback=lambda r:print(f'Seed {seed} epoch {r["epoch"]:02} loss {r["total"]:.5f} ({r["seconds"]:.1f}s)',flush=True) if r['epoch']%10==0 else None)
        score=predict(model,ds,queries,pairs,labels,torch.device('cpu')).reshape(-1,2)
        all_scores.append(score)
        pd.DataFrame(history).to_csv(OUT/f'training_seed{seed}.csv',index=False)
        histories.append(history)
        checkpoint=dict(model_state={k:v.cpu() for k,v in model.state_dict().items()},
            model_config=asdict(mc),training_config=config['training'],seed=seed,
            reference_pairs=torch.as_tensor(pairs),reference_labels=torch.as_tensor(labels),
            drugs=ds.drugs,diseases=ds.diseases,drug_dim=ds.drug_features.shape[1],disease_dim=ds.disease_features.shape[1])
        torch.save(checkpoint,model_dir/f'seed{seed}.pt')
    scores=np.stack(all_scores)
    assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    np.savez_compressed(OUT/'prediction_arrays.npz',scores=scores,queries=queries,seeds=args.seeds,
                        drug_features=ds.drug_features,disease_features=ds.disease_features)
    mean=scores.mean(0)
    std=scores.std(0,ddof=1 if len(scores)>1 else 0)
    rows=[]
    for r in candidates:
        row=dict(excel_row=r['excel_row'],catalog=r['catalog'],name=r['name'],cas=r['cas'],
                 status=r['prediction_status'],seen_in_original_training=r['seen_in_original_training'],
                 training_matches='; '.join(m['name'] for m in r['training_matches']),
                 known_CE_positive=any(m['CE_label']==1 for m in r['training_matches']),
                 known_AE_positive=any(m['AE_label']==1 for m in r['training_matches']),
                 parent_group=r.get('feature_index'),same_parent_catalogs='; '.join(r['same_parent_catalogs']),
                 notes='; '.join(r['notes']),parent_smiles=r.get('parent_smiles',''),
                 token_count=r.get('token_count'),out_of_domain_peptide=r.get('amide_bonds',0)>=10)
        for j,t in enumerate(['CE','AE']):
            if r['prediction_status']=='ok':
                idx=r['feature_index']
                row[t+'_score']=float(mean[idx,j])
                row[t+'_seed_sd']=float(std[idx,j])
                for k,seed in enumerate(args.seeds):
                    row[t+f'_seed{seed}']=float(scores[k,idx,j])
            else:
                row[t+'_score']=None
                row[t+'_seed_sd']=None
        rows.append(row)
    frame=pd.DataFrame(rows)
    # Rank active parents once. Keep peptide out of the small-molecule ranking.
    eligible=frame[frame.status.eq('ok')&~frame.out_of_domain_peptide].drop_duplicates('parent_group')
    for t in ['CE','AE']:
        ranks=eligible.set_index('parent_group')[t+'_score'].rank(ascending=False,method='min')
        frame[t+'_parent_rank']=frame.parent_group.map(ranks)
    frame.to_csv(OUT/'predictions.csv',index=False,encoding='utf-8-sig')
    (OUT/'predictions.json').write_text(frame.astype(object).where(frame.notna(),None).to_json(orient='records',force_ascii=False,indent=2),encoding='utf-8')
    # Snapshot precisely the implementation used in this run.
    snapshot=OUT/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for path in list((ROOT/'rmdldpa').glob('*.py'))+[Path(__file__),ROOT/'configs/default.json']:
        shutil.copy2(path,snapshot/path.name)
    manifest=dict(protocol='cohort drug-cold-start: all matched candidate drugs removed from all association/reference labels',
        seeds=args.seeds,epochs=config['training']['epochs'],training_drug_count=len(retained),
        training_positives=len(positives),sampled_unknowns_per_seed=len(positives),excluded_training_drugs=[ds.drugs[i] for i in excluded],
        cohort_excluded_drugs=[ds.drugs[i] for i in cohort_excluded],provenance_excluded_drugs=unverified,
        disease_names=TARGETS,original_target_positive_counts=ds.associations[:,target_ids].sum(0).tolist(),
        retained_target_positive_counts=ds.associations[retained][:,target_ids].sum(0).tolist(),
        candidate_records=len(frame),scored_records=int(frame.CE_score.notna().sum()),
        unique_scored_parents=len(cx['features']),unique_small_molecule_parents=len(eligible),
        score_definition='mean of sigmoid outputs across seeds; uncalibrated balanced-sampling score, NOT efficacy probability',
        uncertainty='SD across seed and sampled-negative variation; NOT a confidence interval',
        all_candidate_drugs_absent_from_training_and_retrieval=True,new_candidate_labels_used=False,
        old_5cv_metrics_applicable=False,inputs=ds.provenance,
        hashes={p.name:sha256(p) for p in [OUT/'source_drug_list.xlsx',OUT/'candidate_features.npz',OUT/'predictions.csv',OUT/'structures_audit.json']},
        source_hashes={p.name:sha256(p) for p in snapshot.glob('*')})
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPLETE',json.dumps({k:v for k,v in manifest.items() if k not in ['inputs','hashes','source_hashes']},ensure_ascii=True),flush=True)

if __name__=='__main__':
    main()
