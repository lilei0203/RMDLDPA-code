"""Fixed-protocol grouped drug-cold 5-CV, followed by candidate prediction.

Neither hyperparameters nor model choice are selected using candidate scores or
outer test labels. Original, mask-only, and hybrid are all reported, including
unfavorable results. Unknown pairs remain unverified negatives in evaluation.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from rmdldpa.config import ModelConfig
from rmdldpa.data import load_dataset,sha256
from rmdldpa.training import predict
from rmdldpa.cold_training import train_fixed,build_model
from rmdldpa.metrics import classification_metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/coldstart_v2'
OLD=ROOT/'results/candidate_screen_20261109'
VARIANTS=['original','mask','hybrid']
TARGETS=['Cystic echinococcosis','Alveolar echinococcosis']

def metrics(y,s):
    return classification_metrics(y,s) if len(np.unique(y))==2 else {'auroc':None,'aupr':None}

def get_model(ds,pairs,labels,c,tc,assets,variant,seed,folder):
    folder.mkdir(parents=True,exist_ok=True)
    sample=pd.DataFrame(pairs,columns=['drug_index','disease_index']); sample['label']=labels
    if (folder/'complete.json').exists():
        ckpt=torch.load(folder/'model.pt',weights_only=True,map_location='cpu')
        np.testing.assert_array_equal(ckpt['reference_pairs'],pairs)
        np.testing.assert_array_equal(ckpt['reference_labels'],labels)
        assert ckpt['variant']==variant and ckpt['seed']==seed
        model=build_model(ds,c,assets,variant); model.load_state_dict(ckpt['model_state'])
        return model
    sample.to_csv(folder/'reference_pairs.csv',index=False)
    return train_fixed(ds,pairs,labels,c,tc,assets,variant,seed,folder)

def benchmark(ds,c,tc,assets):
    eligible=assets['eligible_drugs']; groups=assets['drug_groups']
    allow=np.zeros_like(ds.associations,dtype=bool); allow[eligible]=True
    positive=np.argwhere((ds.associations==1)&allow); unknown=np.argwhere((ds.associations==0)&allow)
    rng=np.random.default_rng(42)
    negative=unknown[rng.choice(len(unknown),len(positive),replace=False)]
    pool=np.concatenate((positive,negative)); labels=np.r_[np.ones(len(positive)),np.zeros(len(negative))].astype(np.int64)
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261109)
    fold_rows=[]; all_frames=[]
    for fold,(itr,ite) in enumerate(splitter.split(pool,labels,groups[pool[:,0]]),1):
        test_groups=set(groups[pool[ite,0]])
        test_drugs=np.array([i for i in eligible if groups[i] in test_groups])
        train_drugs=np.array([i for i in eligible if groups[i] not in test_groups])
        assert not set(groups[train_drugs])&set(groups[test_drugs])
        assert set(pool[itr,0])<=set(train_drugs)
        queries=np.array([[i,j] for i in test_drugs for j in range(len(ds.diseases))],dtype=np.int64)
        truth=ds.associations[queries[:,0],queries[:,1]]
        balanced_keys={tuple(p) for p in pool[ite]}
        balanced=np.array([tuple(p) in balanced_keys for p in queries])
        fold_dir=OUT/'benchmark'/f'fold_{fold}'; fold_dir.mkdir(parents=True,exist_ok=True)
        split=dict(train_drugs=train_drugs.tolist(),test_drugs=test_drugs.tolist(),
                   train_groups=sorted(map(int,set(groups[train_drugs]))),test_groups=sorted(map(int,test_groups)))
        (fold_dir/'split.json').write_text(json.dumps(split,indent=2))
        for variant in VARIANTS:
            folder=fold_dir/variant
            seed=4200+fold
            print(f'BENCHMARK fold={fold}/5 variant={variant} train={len(itr)} test_all={len(queries)}',flush=True)
            model=get_model(ds,pool[itr],labels[itr],c,tc,assets,variant,seed,folder)
            score=predict(model,ds,queries,pool[itr],labels[itr],torch.device('cpu'))
            frame=pd.DataFrame(queries,columns=['drug_index','disease_index'])
            frame['label']=truth; frame['score']=score; frame['balanced_subset']=balanced; frame['fold']=fold; frame['variant']=variant
            frame.to_csv(folder/'predictions.csv',index=False)
            record=dict(fold=fold,variant=variant,
                all_pairs=metrics(truth,score),balanced=metrics(truth[balanced],score[balanced]),
                targets={t:metrics(truth[queries[:,1]==ds.diseases.index(t)],score[queries[:,1]==ds.diseases.index(t)]) for t in TARGETS})
            (folder/'metrics.json').write_text(json.dumps(record,indent=2))
            (folder/'complete.json').write_text(json.dumps(dict(checkpoint_sha256=sha256(folder/'model.pt'),finished=True)))
            fold_rows.append(record); all_frames.append(frame)
            print('FOLD_RESULT',json.dumps(record),flush=True)
    all_df=pd.concat(all_frames,ignore_index=True)
    all_df.to_csv(OUT/'benchmark_oof.csv',index=False)
    summary={}
    for variant in VARIANTS:
        f=all_df[all_df.variant.eq(variant)]; balanced=f.balanced_subset
        assert len(f)==len(eligible)*len(ds.diseases)
        assert not f.duplicated(['drug_index','disease_index']).any()
        vrows=[r for r in fold_rows if r['variant']==variant]
        summary[variant]=dict(pooled_all_pairs=metrics(f.label,f.score),pooled_balanced=metrics(f.loc[balanced,'label'],f.loc[balanced,'score']),
            mean_folds_all_pairs={k:float(np.mean([r['all_pairs'][k] for r in vrows])) for k in ['auroc','aupr']},
            sd_folds_all_pairs={k:float(np.std([r['all_pairs'][k] for r in vrows],ddof=1)) for k in ['auroc','aupr']},
            targets={t:metrics(f.loc[f.disease_index.eq(ds.diseases.index(t)),'label'],f.loc[f.disease_index.eq(ds.diseases.index(t)),'score']) for t in TARGETS})
    (OUT/'benchmark_summary.json').write_text(json.dumps(summary,indent=2))
    print('BENCHMARK_COMPLETE',json.dumps(summary),flush=True)

def screening(ds,c,tc,assets):
    candidates=json.loads((OLD/'candidates_encoded.json').read_text(encoding='utf-8'))
    eligible=assets['eligible_drugs']; allow=np.zeros_like(ds.associations,dtype=bool); allow[eligible]=True
    positives=np.argwhere((ds.associations==1)&allow); unknown=np.argwhere((ds.associations==0)&allow)
    count=len(ds.drugs); nparents=len(ds.drug_features)-count
    queries=np.array([[count+i,ds.diseases.index(t)] for i in range(nparents) for t in TARGETS],dtype=np.int64)
    # Matched original controls must use the SAME cleaned cohort and negative
    # samples. Historical v1 scores are also retained but are not a fair ablation.
    controls=[]
    for seed in [42,43,44]:
        rng=np.random.default_rng(seed); negative=unknown[rng.choice(len(unknown),len(positives),replace=False)]
        pairs=np.concatenate((positives,negative)); labels=np.r_[np.ones(len(positives)),np.zeros(len(negative))].astype(np.int64)
        folder=OUT/'screening'/f'original_seed_{seed}'
        model=get_model(ds,pairs,labels,c,tc,assets,'original',seed,folder)
        values=predict(model,ds,queries,pairs,labels,torch.device('cpu')).reshape(nparents,2)
        controls.append(values); np.save(folder/'scores.npy',values)
        (folder/'complete.json').write_text(json.dumps(dict(checkpoint_sha256=sha256(folder/'model.pt'),finished=True)))
    scores=[]
    for seed in [42,43,44]:
        rng=np.random.default_rng(seed); negative=unknown[rng.choice(len(unknown),len(positives),replace=False)]
        pairs=np.concatenate((positives,negative)); labels=np.r_[np.ones(len(positives)),np.zeros(len(negative))].astype(np.int64)
        control_pairs=pd.read_csv(OUT/'screening'/f'original_seed_{seed}'/'reference_pairs.csv')
        np.testing.assert_array_equal(pairs,control_pairs[['drug_index','disease_index']].to_numpy())
        np.testing.assert_array_equal(labels,control_pairs.label)
        folder=OUT/'screening'/f'seed_{seed}'
        model=get_model(ds,pairs,labels,c,tc,assets,'hybrid',seed,folder)
        values=predict(model,ds,queries,pairs,labels,torch.device('cpu')).reshape(nparents,2)
        scores.append(values)
        # Save auditable top evidence for both diseases; labels are training labels.
        evidence=[]
        model.eval()
        with torch.no_grad():
            dx=torch.as_tensor(ds.drug_features); px=torch.as_tensor(ds.disease_features)
            refs=torch.as_tensor(pairs); ys=torch.as_tensor(labels)
            for start in range(0,len(queries),64):
                batch=torch.as_tensor(queries[start:start+64]); output=model(batch,dx,px,refs,ys)
                for j,q in enumerate(batch):
                    neighbors=[]
                    for k in range(output['attention'].shape[1]):
                        if not output['reference_mask'][j,k]: continue
                        ri=int(output['reference_indices'][j,k]); drug,disease=map(int,pairs[ri])
                        assert assets['drug_groups'][int(q[0])]!=assets['drug_groups'][drug]
                        neighbors.append(dict(drug_index=drug,drug=ds.drugs[drug],disease=ds.diseases[disease],label=int(labels[ri]),
                            attention=float(output['attention'][j,k]),tanimoto=float(assets['chemical_similarity'][int(q[0]),drug])))
                    evidence.append(dict(candidate_parent=int(q[0])-count,disease=ds.diseases[int(q[1])],gate=float(output['reliability'][j]),neighbors=neighbors))
        (folder/'neighbors.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
        np.save(folder/'scores.npy',values)
        (folder/'complete.json').write_text(json.dumps(dict(checkpoint_sha256=sha256(folder/'model.pt'),finished=True)))
    array=np.stack(scores); control_array=np.stack(controls)
    np.savez_compressed(OUT/'candidate_scores.npz',scores=array,control_scores=control_array,queries=queries,seeds=[42,43,44])
    old=pd.read_csv(OLD/'predictions.csv')
    rows=[]
    for i,r in enumerate(candidates):
        item=dict(excel_row=r['excel_row'],catalog=r['catalog'],name=r['name'],status=r['prediction_status'],
                  parent_group=r.get('feature_index'),seen_in_original_training=r['seen_in_original_training'],
                  out_of_domain_peptide=r.get('amide_bonds',0)>=10,notes='; '.join(r['notes']))
        for j,t in enumerate(['CE','AE']):
            item[t+'_old_score']=float(old.iloc[i][t+'_score']) if r['prediction_status']=='ok' else None
            item[t+'_old_sd']=float(old.iloc[i][t+'_seed_sd']) if r['prediction_status']=='ok' else None
            if r['prediction_status']=='ok':
                s=array[:,r['feature_index'],j]
                item[t+'_score']=float(s.mean()); item[t+'_seed_sd']=float(s.std(ddof=1))
                cs=control_array[:,r['feature_index'],j]
                item[t+'_control_score']=float(cs.mean()); item[t+'_control_sd']=float(cs.std(ddof=1))
                item[t+'_history_change']=item[t+'_score']-item[t+'_old_score']
                item[t+'_change']=item[t+'_score']-item[t+'_control_score']
                for k,seed in enumerate([42,43,44]): item[t+f'_seed{seed}']=float(s[k])
            else:
                item[t+'_score']=item[t+'_seed_sd']=item[t+'_change']=None
                item[t+'_control_score']=item[t+'_control_sd']=item[t+'_history_change']=None
        rows.append(item)
    frame=pd.DataFrame(rows)
    unique=frame[frame.status.eq('ok')&~frame.out_of_domain_peptide].drop_duplicates('parent_group')
    comparison={}
    for t in ['CE','AE']:
        ranks=unique.set_index('parent_group')[t+'_score'].rank(ascending=False,method='min')
        frame[t+'_rank']=frame.parent_group.map(ranks)
        corr=unique[[t+f'_seed{s}' for s in [42,43,44]]].corr(method='spearman')
        comparison[t]=dict(parent_count=len(unique),higher_scores=int((unique[t+'_change']>0).sum()),lower_scores=int((unique[t+'_change']<0).sum()),
            mean_old_score=float(unique[t+'_old_score'].mean()),mean_new_score=float(unique[t+'_score'].mean()),
            mean_control_score=float(unique[t+'_control_score'].mean()),median_control_sd=float(unique[t+'_control_sd'].median()),
            median_old_sd=float(unique[t+'_old_sd'].median()),median_new_sd=float(unique[t+'_seed_sd'].median()),
            pairwise_spearman=[float(corr.iloc[0,1]),float(corr.iloc[0,2]),float(corr.iloc[1,2])])
    frame.to_csv(OUT/'candidate_comparison.csv',index=False,encoding='utf-8-sig')
    (OUT/'candidate_comparison.json').write_text(frame.astype(object).where(frame.notna(),None).to_json(orient='records',force_ascii=False,indent=2,double_precision=15),encoding='utf-8')
    (OUT/'candidate_summary.json').write_text(json.dumps(comparison,indent=2))
    print('SCREENING_COMPLETE',json.dumps(comparison),flush=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['benchmark','screen','all'],default='all'); args=parser.parse_args()
    torch.set_num_threads(4)
    assets=np.load(OUT/'retrieval_assets.npz')
    ds=load_dataset(ROOT/'data/processed/association.csv',ROOT/'data/processed/drug_embedding.csv',ROOT/'data/processed/disease_embedding_pubmed.csv')
    ds.drug_features=assets['drug_features']
    config=json.loads((ROOT/'configs/default.json').read_text()); c=ModelConfig(**config['model']); tc=config['training']
    protocol=dict(config=config,variants=VARIANTS,epochs_fixed_before_evaluation=60,folds=5,split_seed=20261109,
        pool_seed=42,final_seeds=[42,43,44],selection='hybrid preselected; no selection or tuning using test or candidate scores',
        primary_evaluation='all retained held-out drugs x 95 diseases; zeros are unlabeled',secondary_evaluation='fixed balanced pool subset',
        cold_groups='normalized parent name OR verified feature-derived parent connectivity',
        hybrid=dict(chemical_weight=0.25,chemical_recall=20,related_positive_threshold=0.35,context_dropout=0.1),
        assets_sha256=sha256(OUT/'retrieval_assets.npz'),dataset=ds.provenance)
    source_paths=[*list((ROOT/'rmdldpa').glob('*.py')),Path(__file__),ROOT/'scripts/prepare_coldstart_v2.py']
    protocol['source_hashes']={str(p.relative_to(ROOT)):sha256(p) for p in source_paths}
    if (OUT/'protocol.json').exists():
        previous=json.loads((OUT/'protocol.json').read_text())
        if previous!=protocol: raise ValueError('Protocol/source changed: use a new result directory, do not mix runs')
    else:
        (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
        snapshot=OUT/'source_snapshot'
        for p in source_paths:
            target=snapshot/p.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
    if args.stage in ['benchmark','all']: benchmark(ds,c,tc,assets)
    if args.stage in ['screen','all']: screening(ds,c,tc,assets)

if __name__=='__main__': main()
