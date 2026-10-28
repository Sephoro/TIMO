<div align="center">
  <img width=800 src="framework.png"/>
</div>

# TIMO ablation: a reduced-scale reproduction

This repository reproduces one claim from *Text and Image Are Mutually Beneficial: Enhancing Training-Free Few-Shot Classification with CLIP* (Li et al., AAAI 2025, [arXiv:2412.11375](https://arxiv.org/abs/2412.11375)).

**Claim tested.** The paper's ablation (Table 3) reports that adding Text-Guided-Image (TGI), Image-Guided-Text (IGT), or both to a base classifier (GDA-CLIP, and also Tip-Adapter) improves few-shot accuracy, and that both together improve it further.

**Scale.** CLIP RN50, three of the paper's eleven datasets (Caltech101, DTD, Oxford-IIIT Pets), 1, 4 and 16 shots per class, seeds 1, 2 and 3. All results were produced on CPU. TIMO-S and the other backbones are not evaluated.

**What is ours.** The code builds on the authors' implementation (<https://github.com/lyymuwu/TIMO>). We patched it to run on CPU, added a `--timo_mode` flag (`base`, `tgi`, `igt`, `both`) that applies each component to Tip-Adapter and GDA-CLIP, and added a wrong-class prompt-injection experiment (`--junk`) that the paper does not report. The authors' `TIMO` function is left unmodified and run as a reference row. The extension of TGI to Tip-Adapter is our own construction and has not been validated against the authors' code.

---

<a id="quick-start"></a>
## Quick start

```bash
mkdir timo-workspace && cd timo-workspace
git clone https://github.com/Sephoro/TIMO.git && cd TIMO
conda create -n TIMO python=3.8 -y && conda activate TIMO
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
bash run-all.sh      # data, features, all experiments, analysis (several hours)
```

Run everything from the repository root. The code expects the datasets in `../DATA`, one level above the repository, which is why the clone goes inside a workspace folder. See [Run the Complete Pipeline](#complete) for details.

If the quick start worked, skim [Environment](#env-setting) and [Data](#data) to confirm your setup matches, then jump to [Model Inference](#inference) and [Results and Analysis](#results). You can re-run individual ablation settings later via [Running the Full Experiment Suite](#full-suite).

<a id="requirements"></a>
## Requirements

| | |
|---|---|
| System | Linux, conda. No GPU needed. |
| Disk | Datasets 7.1 GB extracted (6.9 GB archive, deleted after extraction). Feature cache 0.8 GB archive, ~2 GB extracted. |
| Time | Feature extraction ~4 hours on an 8-thread laptop CPU. Inference and analysis ~6 hours. |

---

<a id="setup"></a>
## Setup: step by step

Use this section if the quick start failed, timed out, or you prefer to run and inspect each stage individually. Most steps are covered by [Quick start](#quick-start).

<a id="getting-the-code"></a>
### Getting the code

```bash
mkdir timo-workspace && cd timo-workspace
git clone https://github.com/Sephoro/TIMO.git
cd TIMO
```

<a id="env-setting"></a>
### Environment

```bash
conda create -n TIMO python=3.8
conda activate TIMO
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
```

The results in the report were produced on CPU (no GPU is needed). `requirements.txt` includes `gdown`, which the data script uses.

<a id="data"></a>
### Data

The experiments use Caltech101, DTD and Oxford-IIIT Pets, with the CoOp train/val/test split files. We host one archive [`data1.tar.gz`](https://drive.google.com/file/d/1eQozsqRSj8h8gPKtFsOc05shFKSTHakk/view?usp=drive_link) containing the three datasets and their splits, already arranged as the code expects.

**Disk space.** About 6.9 GB for the archive and 7.1 GB once extracted, plus space for the environment and feature caches. Free up space before you start.

#### Option 1: script (recommended)

```bash
bash setup_data.sh
```

This downloads the archive, extracts it into `../DATA`, and removes the archive. If `../DATA` already holds the three datasets it does nothing. If this fails, follow Option 2 below.

#### Option 2: manual (skip if Option 1 worked)

Use this if the script fails (for example, a Google Drive download quota or lack of space).

```bash
mkdir -p ../DATA
gdown --fuzzy "https://drive.google.com/file/d/1eQozsqRSj8h8gPKtFsOc05shFKSTHakk/view?usp=drive_link" -O data1.tar.gz
tar -xzf data1.tar.gz --strip-components=1 -C ../DATA
rm data1.tar.gz
```

You can also download `data1.tar.gz` through a browser from the same link, then run the `tar` and `rm` lines. If extraction creates `../DATA/DATA/...`, redo it with `tar -xzf data1.tar.gz -C ../DATA --strip-components=1`.

#### Check the layout

```bash
ls ../DATA
```

Expected contents: `caltech-101`, `dtd`, `oxford_pets`, each containing its split JSON (for example `caltech-101/split_zhou_Caltech101.json`).

#### Using the original sources

The archive is a convenience copy. The datasets belong to their original authors and keep their own licences. To build from the originals, follow the Tip-Adapter / CoOp instructions: <https://github.com/gaopengcuhk/Tip-Adapter/blob/main/DATASET.md>

#### Configs

The running configurations live in `configs/<dataset>.yaml`. Edit the `root_path` field to point to where the dataset resides on your machine.

**Note:** `data_path` in `extract_features_all.py` must be updated to match as well.

<a id="features"></a>
### Extracting features

Feature extraction and model inference are separate steps. Features are extracted once and cached; every later experiment reads from the cache.

```bash
python extract_features_all.py
```

This extracts CLIP RN50 features for Caltech101, DTD and Oxford-IIIT Pets, with shots 1, 4 and 16 and seeds 1, 2 and 3, writing them to `caches/RN50/<seed>/<dataset>/`. The seed determines which support images are sampled, so few-shot features differ per seed. Each support image is encoded under ten random-crop and flip augmentations.

**Runtime and device.** This is the slowest step by far: about 4 hours on an 8-thread laptop CPU. It runs on CPU by default, which is the setting the reported results come from. To use a GPU: `CUDA_VISIBLE_DEVICES=0 python extract_features_all.py` (feature values may differ slightly). To run a smaller subset, edit `seeds`, `datasets` and `k_shot` at the bottom of `extract_features_all.py`.

#### Skipping extraction with the pre-computed cache

If extraction takes too long, download the cache we generated for all three datasets, seeds 1/2/3, k-shots 1/4/16: <https://drive.google.com/file/d/10HY1IybNdYOUj7xv9O09DXmefMjAEJ-7/view>

```bash
gdown --fuzzy "https://drive.google.com/file/d/10HY1IybNdYOUj7xv9O09DXmefMjAEJ-7/view" -O caches.tar.gz
tar -xzf caches.tar.gz            # run from the repository root
rm caches.tar.gz
ls caches/RN50                    # expect: 1  2  3
```

The archive must unpack to a folder named `caches/` in the repository root, because the code reads `./caches/RN50/<seed>/<dataset>/`. Size: ~0.8 GB archive, ~2 GB extracted. `run-all.sh` skips extraction when the cache is present.

---

<a id="inference"></a>
## Model inference

All experiments should be run from the repository root after preparing the datasets and extracting the required features.

### Base configuration (original authors' settings)

```bash
python main.py --config configs/<dataset_name>.yaml --shot <shot_number> --seed <seed_number>
```

- `dataset_name`: one of `caltech101`, `dtd`, `oxford_pets`
- `shot_number`: training examples per class (`1`, `4`, or `16`)
- `seed_number`: random seed (any integer, as long as the corresponding features have been extracted)

**Note:** Without additional arguments, `--timo_mode` defaults to `base` and `--junk` defaults to `0`. This matches the original authors' setup: it runs the baseline implementations (Tip-Adapter, APE, GDA-CLIP) without TGI/IGT modifications. The authors' original `TIMO` function is retained as a reference row.

### Single experiment with custom settings

```bash
python main.py \
    --config configs/caltech101.yaml \
    --shot 1 \
    --seed 1 \
    --timo_mode base \
    --junk 0
```

**Arguments:**

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Dataset config (`caltech101`, `dtd`, `oxford_pets`) | — |
| `--shot` | Training examples per class (`1`, `4`, `16`) | `1` |
| `--seed` | Random seed (`1`, `2`, `3`) | `1` |
| `--timo_mode` | Ablation mode (see table below) | `base` |
| `--junk` | Number of junk prompts to inject | `0` |

### Ablation modes

| Mode | Description |
|------|-------------|
| `base` | Baseline implementations (default) |
| `tgi` | Text-Guided-Image component only |
| `igt` | Image-Guided-Text component only |
| `both` | Combined TGI and IGT components |

When `--timo_mode` is `base`, `tgi`, or `igt`, the ablation affects Tip-Adapter and GDA-CLIP only. The `base` run establishes the reference numbers; the other modes layer one or both components on top. See [Running the Full Experiment Suite](#full-suite) for end-to-end commands.

### Junk-prompt injection

`--junk` injects irrelevant prompts drawn from incorrect classes to test robustness. A value of `0` disables injection (default). Our experiments evaluate 2, 4, 8, 16, and 32 junk prompts.

```bash
python main.py \
    --config configs/caltech101.yaml \
    --shot 1 \
    --seed 1 \
    --junk 8
```

---

<a id="full-suite"></a>
## Running the full experiment suite

Each command below evaluates all three datasets, shot settings (1, 4, 16), and seeds (1, 2, 3).

### Baseline

```bash
bash run.sh
```

### TGI ablation

```bash
bash run.sh tgi
```

### IGT ablation

```bash
bash run.sh igt
```

### Combined TGI + IGT

```bash
bash run.sh both
```

### Junk-prompt robustness

All five noise levels (2, 4, 8, 16, 32) in one go:

```bash
bash run.sh --all-noise
```

With a specific ablation mode:

```bash
bash run.sh both --all-noise
```

A single noise level:

```bash
bash run.sh base --noise 8
```

**Note:** The junk-prompt experiments extend the original paper and are not part of its reported results.

---

<a id="results"></a>
## Results and analysis

All runs write raw results to `outputs/` in the repository root. Each mode gets its own subfolder with one `.txt` file per metric.

```text
outputs/
├── original/                 # clean baseline (no junk, base mode)
│   ├── APE.txt
│   ├── GDA_CLIP.txt
│   ├── TIMO.txt
│   └── Tip_Adapter.txt
├── tgi/                      # TGI-only ablation
├── igt/                      # IGT-only ablation
├── both/                     # combined TGI + IGT
├── original_junk2/           # baseline + 2 junk prompts
├── original_junk4/           # baseline + 4 junk prompts
└── ...                       # original_junk8, original_junk16, original_junk32
```

Each `.txt` file stores one line per experiment:

```text
dataset_shots_seed: score
```

### Generating CSV tables

Use `analyse.py` to convert the raw logs into organised CSVs:

```bash
python analyse.py --mode base
```

This reads `outputs/original/`, aggregates across seeds, and writes CSVs to `outputs/analysis/base/`. Specify a different root with `--root`:

```bash
python analyse.py --mode base --root outputs
```

### Ablation analysis (`--mode base|tgi|igt|both`)

Produces the following files inside `outputs/analysis/<mode>/`:

| File | Contents |
|------|----------|
| `all_results.csv` | One row per experiment, one column per method. |
| `summary_by_dataset_shots.csv` | Mean ± std across seeds, per dataset and shot count. |
| `method_comparison.csv` | Mean score per method, best method flagged. |
| `overall_method_ranking.csv` | Overall mean accuracy and spread per method. |

```bash
python analyse.py --mode both
```

### Junk-prompt analysis (`--mode junk`)

Compares every junk run against the clean baseline and computes accuracy drop per method.

```bash
python analyse.py --mode junk
```

Limit to specific noise levels with `--noise-levels` (accepts one or more integers):

```bash
python analyse.py --mode junk --noise-levels 2 16
```

Writes to `outputs/analysis/junk_prompt/`:

| File | Contents |
|------|----------|
| `noise_degradation.csv` | Every noisy run matched to its clean baseline, with the drop. |
| `method_robustness_summary.csv` | Mean and relative drop per method and noise level. |
| `degradation_by_dataset_shots.csv` | Breakdown by dataset, shot count, and noise level. |
| `timo_shot_sensitivity.csv` | TIMO accuracy drop by noise level and shot count. |
| `timo_most_affected_shots.csv` | Shot setting with the largest drop at each noise level. |
| `junk_weight_analysis.csv` | Raw junk-weight logs for inspection. |

**Note:** Junk analysis assumes the clean baseline lives in `outputs/original/`. Run `bash run.sh` (base mode) first.

### Numerical results

Pre-computed CSVs for base, ablation, and junk-prompt runs are provided under `outputs/archive/` for reference.

---

<a id="complete"></a>
## Run the complete pipeline

```bash
bash run-all.sh
```

Stages, in order:

1. **Dataset setup** — downloads and extracts the three datasets.
2. **Feature extraction** — caches CLIP RN50 image and text features (skipped if `caches/` is present).
3. **Baseline experiments** — `run.sh base`.
4. **TGI experiments** — `run.sh tgi`.
5. **IGT experiments** — `run.sh igt`.
6. **Combined experiments** — `run.sh both`.
7. **Junk-prompt experiments** — `run.sh base --all-noise`.
8. **Analysis** — `analyse.py` for every mode and the junk comparison.

Experiments cover all three datasets, three shot counts, and three seeds. Feature extraction is a separate step so cached features are reused across modes.

> **Note:** The full pipeline may take several hours depending on hardware and whether features are already cached.

### Resuming an interrupted run

The pipeline records completed stages in `.pipeline_state/`. If a stage fails, fix the problem and re-run:

```bash
bash run-all.sh
```

Stages with a completion marker are skipped. To force a re-run, delete the corresponding `.done` file from `.pipeline_state/` first. Be aware that re-running experiments appends to existing logs, which may produce duplicate entries.

---

## Acknowledgements

This repo is built upon previous work including [TIP-Adapter](https://github.com/gaopengcuhk/Tip-Adapter/), [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/CoOp/), [SuS-X](https://github.com/vishaal27/SuS-X), [CuPL](https://github.com/sarahpratt/CuPL), [APE](https://github.com/yangyangyang127/APE), and [GDA-CLIP](https://github.com/mrflogs/ICLR24). Thanks for their contributions to the field.
