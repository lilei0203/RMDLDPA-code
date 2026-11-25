> **2026-11-25 更新：已按带修订稿核对公式（1）–（32）及Table 2。本次用户范围仅为单次普通5折。推荐从上一级运行 python run_5cv.py。详见 [本次核对](docs/TRACKED_MANUSCRIPT_20261125.md)。现有数据与稿件数据仍不同。**

# npj_RMDLDPA：原文方程实现与普通 5-CV

本项目依据用户提供的 `reference/npj_RMDLDPA.pdf` 重建模型。包含**原文三个核心模块、三个损失、数据检查、预训练特征编码入口、普通 5-CV、内部参数搜索、重复运行、结果核验和历史实验归档**。

**必须区分两件事：模型方程已按原文实现并测试；原始数据和检查点尚未完全匹配，因此这里的运行结果不能称为原文数值的精确复现。** 现成数据是 390 药物、95 疾病、810 正关联、两侧 384 维；原稿写的是 383、51、742、两侧 768 维。完整差异见 [原文对照审计](docs/REPRODUCTION_AUDIT.md)。

按本次用户确认，只做普通 pair-wise 5-CV。冷启动、体外实验及案例分析不在本项目运行范围内。

**已验证（2026-11-08）：13 项测试全部通过；随附数据完成每折 60 epoch 的完整 5-CV。** AUC **0.8461 ± 0.0232**，AUPR/AP **0.8157 ± 0.0397**（五折间样本标准差）。新结果位于 `results/corrected_5cv_seed42/`；这不是原稿 383×51 数据集上的结果。

## 1. 项目结构

```text
npj_RMDLDPA/
├── README.md
├── run_5cv.py / run_5cv.ps1 / run_5cv.sh  # 主入口
├── encode_features.py                    # SMILES、疾病文本编码入口
├── audit_data.py                         # 数据、归档哈希、SMILES 覆盖检查
├── verify_results.py                     # 从预测文件重新计算指标
├── requirements.txt / requirements-encoders.txt
├── configs/
│   ├── default.json                      # Table 2 参数＋明确标注的实现选择
│   └── inner_grid_example.json           # 训练内部参数搜索示例
├── rmdldpa/
│   ├── config.py / data.py / splits.py
│   ├── features.py                      # 非 padding token 均值池化
│   ├── intrinsic.py                     # 双向门控与配对交互
│   ├── retrieval.py                     # 三视角检索、重排、证据聚合
│   ├── losses.py                        # 原文 (12)、(17)、(31)、(32)
│   ├── fusion.py / model.py             # 可靠性融合与完整模型
│   └── training.py / metrics.py / experiment.py
├── data/
│   ├── processed/                       # 开箱可跑的对齐数据
│   ├── raw/conversation_dataset/        # 对话中的 Dataset 原文件
│   ├── raw/additional_sources/          # 本地 SMILES 和其他来源，未擅自合并
│   ├── text/                            # 已提供的 PubMed 文本和可匹配 SMILES
│   ├── manifest.json                    # 来源、用途、文件大小与 SHA-256
│   └── audit.json                       # 实体计数、缺失和冲突清单
├── archive/
│   ├── pair_mlp_baseline/               # 旧简化基线代码与说明
│   └── previous_adaptation/             # 旧模型源码、配置、1/60 epoch 结果
├── reference/                           # 原稿 PDF 与提取文本
├── tests/                               # 方程、边界、划分、端到端测试
├── docs/REPRODUCTION_AUDIT.md
├── docs/EXPERIMENTS.md
└── results/                             # 新模型结果，和旧结果分开
```

## 2. 安装与快速运行

Python 3.10 或以上。仅运行随附特征上的训练不需要 Transformers，也不需要联网下载模型。

```bash
python -m pip install -r requirements.txt
python audit_data.py
python -m unittest discover -s tests -v
```

先检查完整五折流程（每折 1 epoch，只是 smoke test）：

```bash
python run_5cv.py --device cpu --epochs 1 --out-dir results/my_smoke
```

正式单次 5-CV（每折默认 60 epoch）：

```bash
python run_5cv.py --device cpu --threads 4 --out-dir results/my_5cv
python verify_results.py results/my_5cv
```

当前电脑已验证可用的 Python 位于 `D:\Mem-env\python.exe`，但它是 CPU 版 PyTorch。PowerShell 示例：

```powershell
Set-Location 'D:\kaust-postdoc\RMDLDPA_reproduction\npj_RMDLDPA'
& 'D:\Mem-env\python.exe' run_5cv.py --device cpu --threads 4 --out-dir results/my_5cv
& 'D:\Mem-env\python.exe' verify_results.py results/my_5cv
```

有 CUDA 版 PyTorch 的服务器上，可以限定可用 GPU：

```bash
CUDA_VISIBLE_DEVICES=1 python run_5cv.py --device cuda --out-dir results/gpu_5cv
```

脚本默认根据入口文件定位数据和配置，不依赖调用时的工作目录。显式传入的相对 `--out-dir` 等路径则相对于调用目录。**已有非空输出目录会被拒绝覆盖**；重跑请使用新目录。

## 3. 模型与训练流程

1. 严格检查关联矩阵、实体 ID、特征维度、重复项、NaN/Inf。
2. 每次独立运行从未知配对无放回抽取与正例等量的负例，固定本次的平衡数据池。
3. 将配对分成普通分层五折。不同折可以共享药物或疾病，但每折的训练配对与测试配对不重叠。
4. 仅用当前训练配对建立历史检索库，训练查询同时排除自身。
5. 计算双向门控的自身交互表示；从药物、疾病、配对三个视角检索参考配对。
6. 正负类别各取 top-25，经交互重排后选 top-30，聚合带参考标签的证据。
7. 使用相似度与标签一致性估计可靠性，再通过特征门控融合自身表示和历史证据。
8. 优化 `Lpred + 0.1*Lcon + 0.3*Lre`。其中 `Lcon` 严格使用公式（12）的逐正样本 log 概率平均。
9. 评估时不向模型提供测试标签，模型输出后才计算指标。
10. 保存划分、概率、损失、检查点和统计；用 `verify_results.py` 独立核验文件。

代码注释标出了原文公式号；逐式映射和所有未说明细节见 [REPRODUCTION_AUDIT.md](docs/REPRODUCTION_AUDIT.md)。没有把缺失步骤用 PairMLP 替代。

## 4. 重复运行与训练内部参数搜索

原稿报告 10 次独立实验的统计。入口默认运行 1 次以便检查；需要 10 次时显式指定：

```bash
python run_5cv.py --runs 10 --seed 42 --out-dir results/ten_runs
```

种子依次为 42–51；每次重新抽负例、划分、初始化。`run_XX/summary.json` 的标准差是该次的**五折间**样本标准差；总 `summary.json` 的 `across_run_means.std` 是**各次五折均值之间**的样本标准差。只有一次运行时，跨运行标准差为 `null`，不会伪造为 0。

原稿没有提供确切搜索空间。默认采用 Table 2 固定参数；如需训练内部搜索：

```bash
python run_5cv.py --grid configs/inner_grid_example.json --out-dir results/inner_search
```

该示例在每折训练集内部留出 20% 验证，按 AP 选择参数，再重新初始化模型、用外层完整训练集重训。**外层测试集不参与选择**。示例搜索空间不是作者原始网格；运行量也高于固定参数版本。选训练轮数可以使用 `training.epochs` 网格。

## 5. 指标及输出文件

同时输出 AUROC、Average Precision、梯形 PR 面积、ACC、Precision、Recall 和 F1，后三种分类指标及 ACC 的阈值为 0.5。

- `aupr`：Average Precision，与本次对话中历史结果的指标名称保持一致。
- `aupr_trapezoid`：对 PR 曲线做梯形积分。
- 原稿没有明确 AUPR 实现，两个值都保存，避免混用。

```text
results/my_5cv/
├── run_manifest.json               # 实际参数、版本、种子、输入/源码哈希、状态
├── source_snapshot/               # 本次运行时的源码与配置
├── fold_metrics.csv               # 每完成一折即更新
├── summary.json                   # 全部完成后的统计
├── verification.json              # verify_results.py 核验后生成
└── run_01/
    ├── summary.json
    └── fold_1/ ... fold_5/
        ├── train_pairs.csv / test_pairs.csv
        ├── predictions.csv        # 实体名、索引、真实标签、预测概率
        ├── epochs.jsonl / training.csv
        ├── model.pt               # 模型权重、配置、训练参考配对、实体顺序
        ├── metrics.json
        └── selection.json         # 仅内部搜索时生成
```

checkpoint 包含用于重放该折推理的训练参考配对。它是**该折模型**，不是在全部数据上训练的最终药物筛选模型。此处不运行额外的案例预测。

## 6. 从原始 SMILES / 文本生成特征

这是独立、可调用的编码阶段。必须先有完整、真实的原始输入和确认的 checkpoint。随附 384 维特征可直接训练；它们不会在程序内被伪装为原文 768 维。

```bash
python -m pip install -r requirements-encoders.txt
python encode_features.py --help
```

输入格式：

- 药物 CSV：`Drug,SMILES`；需要覆盖关联矩阵全部药物。
- 疾病 CSV：`Disease,Text`；需要覆盖全部疾病，不能用空字符串代替。
- 输出：`Drug,Embedding` 或 `Disease,Embedding`，另存 `.metadata.json`。

以已确认的本地检查点为例（路径需要替换为实际模型目录）：

```bash
python encode_features.py --kind drug --input data/text/drug_smiles_complete.csv --model /path/to/confirmed_chemberta_checkpoint --expected-dim 768 --output data/processed/drug_features_original.csv
python encode_features.py --kind disease --input data/text/disease_texts_original.csv --model /path/to/confirmed_bert_checkpoint --expected-dim 768 --output data/processed/disease_features_original.csv
```

默认 `local_files_only=True`。确实需要从 Hugging Face 下载时使用 `--allow-download`，并建议用 `--revision` 指定 commit。缓存位于项目 `.cache/huggingface`。实现基于 [Transformers AutoModel/AutoTokenizer](https://huggingface.co/docs/transformers/model_doc/auto)。

当前随附 `drug_smiles_available.csv` **并不完整，不能直接冒充 complete 文件**；`audit.json` 列出缺失/冲突。`disease_texts_pubmed.csv` 是可用替代文本，但不是原稿声明的 PubChem 语料。常见 ChemBERTa-2 checkpoint 可能实际输出 384 维；编码器会按 `--expected-dim` 校验并报错，不会自动补零或投影。

补齐原始输入后，可启用原稿形状检查：

```bash
python run_5cv.py --association /path/to/original_association.csv --drug-embedding /path/to/original_drug_features.csv --disease-embedding /path/to/original_disease_features.csv --input-mode paper-shape --runs 10 --out-dir results/original_inputs
```

形状检查不等于来源认证；仍需要核实原始语料、checkpoint 和训练设定。

## 7. 历史实验与可复核性

旧 PairMLP、旧适配模型与新模型分开放置；旧文件未修改。各结果的含义与指标见 [实验记录](docs/EXPERIMENTS.md)。不同数据特征、抽样和种子的结果不能视作严格消融比较。

`scripts/collect_context.py` 是本次把本地材料导入项目的一次性脚本，保留原始来源路径；在另一台机器上使用本项目不需要再次执行它。重导入将更新归档，因此正常使用只运行 `audit_data.py` 检查即可。

测试使用标准库 `unittest`，不需要 pytest。重点验证公式（12）的值和梯度、公式（17）的三元组权重、候选不足、空邻居、自检索排除、评估标签独立性、固定五折池、内部划分隔离，以及模型存取后的预测一致性。

## 8. 新名单的 CE / AE 药物冷启动预测（2026-11-09）

新增 `run_candidate_screen.py`、`verify_candidate_screen.py` 和 `scripts/*candidate*` / `scripts/*screen*`。运行数据、身份核对、训练权重及结果保存在 [本次预测目录](results/candidate_screen_20261109/README.md)。所有匹配到的名单药物均从训练及检索参考库剔除，疾病使用原矩阵对应列。

本次通过 772 条原始向量核验，确认所提供药物特征可用 `DeepChem/ChemBERTa-77M-MLM` 加 **slow tokenizer**、非 padding token 均值池化重现；389 个训练药物可追溯，剩余一个不纳入本次训练。此证据只确认所提供数据的编码方式，不代表论文的原始输入已全部取得。新输出是未经概率校准的关联排序分数，不能套用普通 5-CV 指标解释其冷启动性能。

## 9. 冷启动改进 V2

新增独立的冷启动改进模块、药物分组 5 折对照实验，以及 79 条名单的三种子重预测。原文实现保留；改进涉及同药物屏蔽、结构与表征混合检索、TabR 风格邻居聚合、相关正例约束和证据覆盖门控。详见 [方案与运行说明](docs/COLDSTART_V2.md)。
