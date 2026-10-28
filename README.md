<div align="center">
  <img width=800 src="framework.png"/>
</div>

# TIMO ablation: a reduced-scale reproduction

This repository reproduces one claim from *Text and Image Are Mutually Beneficial:
Enhancing Training-Free Few-Shot Classification with CLIP* (Li et al., AAAI 2025,
[arXiv:2412.11375](https://arxiv.org/abs/2412.11375)).

**Claim tested.** The paper's ablation (Table 3) reports that adding Text-Guided-Image
(TGI), Image-Guided-Text (IGT), or both to a base classifier (GDA-CLIP, and also
Tip-Adapter) improves few-shot accuracy, and that both together improve it further.

**Scale.** CLIP RN50, three of the paper's eleven datasets (Caltech101, DTD, Oxford-IIIT
Pets), 1, 4 and 16 shots per class, seeds 1, 2 and 3. All results were produced on CPU.
TIMO-S and the other backbones are not evaluated.

**What is ours.** The code builds on the authors' implementation
(<https://github.com/lyymuwu/TIMO>). We patched it to run on CPU, added a `--timo_mode`
flag (`base`, `tgi`, `igt`, `both`) that applies each component to Tip-Adapter and
GDA-CLIP, and added a wrong-class prompt-injection experiment (`--junk`) that the paper
does not report. The authors' `TIMO` function is left unmodified and run as a reference row.
The extension of TGI to Tip-Adapter is our own construction and has not been validated
against the authors' code.

## Getting the code

Run everything from the repository root. The code expects the datasets in `../DATA` (Setup in the next section), one
level above the repository, so create a workspace folder first:

```bash
mkdir timo-workspace && cd timo-workspace
git clone https://github.com/Sephoro/TIMO.git
cd TIMO
```

## Environment

```bash
conda create -n TIMO python=3.8
conda activate TIMO
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
```

The results in the report were produced on CPU (no GPU is needed). `requirements.txt`
includes `gdown`, which the data script uses.

## Data

The experiments use Caltech101, DTD and Oxford-IIIT Pets, with the CoOp train/val/test
split files. We host one archive [`data1.tar.gz`](https://drive.google.com/file/d/1eQozsqRSj8h8gPKtFsOc05shFKSTHakk/view?usp=drive_link) containing the three datasets and their
splits, already arranged as the code expects.

**Disk space.** About 6.9 GB for the archive and 7.1 GB once extracted, plus space for the
environment and feature caches. Free up space before you start.

### Option 1: script (recommended)

```bash
bash setup_data.sh
```

This downloads the archive, extracts it into `../DATA`, and removes the archive. If
`../DATA` already holds the three datasets it does nothing. If this process fails, follow option 2 instruction to download the dataset.

### Option 2: manual (Incase of failure in option 1, Skip if option 1 worked)

Use this if the script fails (for example a Google Drive download quota or lack of space).

```bash
mkdir -p ../DATA
gdown --fuzzy "https://drive.google.com/file/d/1eQozsqRSj8h8gPKtFsOc05shFKSTHakk/view?usp=drive_link" -O data1.tar.gz
tar -xzf data1.tar.gz --strip-components=1 -C ../DATA
rm data1.tar.gz
```

You can also download `data.tar.gz` through a browser from the same link, and run the
`tar` and `rm` lines. If extraction creates `../DATA/DATA/...`, redo it with
`tar -xzf data.tar.gz -C ../DATA --strip-components=1`.

### Check the layout

```bash
ls ../DATA
```

The expected contents are `caltech-101`, `dtd` and `oxford_pets`, and each contains its
split JSON (for example `caltech-101/split_zhou_Caltech101.json`).

### Using the original sources

The archive is a convenience copy. The datasets belong to their original authors and keep
their own licences. To build the data from the originals, follow the instructions used by
Tip-Adapter and CoOp:
<https://github.com/gaopengcuhk/Tip-Adapter/blob/main/DATASET.md>

### Configs

The running configurations can be modified in `configs/dataset.yaml`, user need to modify the root_path in configurations to the folder where dataset resides.

Notes, `data_path` in `extract_features_all.py` need to be modified as well.

### Extracting Features

Feature extraction and model inference are separate steps. Features are extracted once
and cached, and every later experiment reads from the cache.

Extract all image and text features with:

```bash
python extract_features_all.py
```

This extracts CLIP RN50 features for Caltech101, DTD and Oxford-IIIT Pets, with shots
1, 4 and 16 and seeds 1, 2 and 3, and writes them to `caches/RN50/<seed>/<dataset>/`.
The seed decides which support images are sampled, so the few-shot features differ per
seed. Each support image is encoded under ten random-crop and flip augmentations.

**Runtime and device.** This step is by far the slowest. On an 8-thread laptop CPU it took
about 4 hours. It runs on CPU by default, which is the setting the reported results come
from. To use a GPU, run `CUDA_VISIBLE_DEVICES=0 python extract_features_all.py`; feature
values may differ slightly from the CPU ones. To run a smaller subset, edit `seeds`,
`datasets` and `k_shot` at the bottom of `extract_features_all.py`.

### Skipping extraction with the pre-computed cache

If extraction takes too long, download the cache we generated from the three datasources being Caltech101, DTD and Oxford-IIIT Pets using seeds 1, 2 and 3 under k-shots 1, 4 and 16:
<https://drive.google.com/file/d/10HY1IybNdYOUj7xv9O09DXmefMjAEJ-7/view>

```bash
gdown --fuzzy "https://drive.google.com/file/d/10HY1IybNdYOUj7xv9O09DXmefMjAEJ-7/view" -O caches.tar.gz
tar -xzf caches.tar.gz            # run from the repository root
rm caches.tar.gz
ls caches/RN50  # expect: 1  2  3
rm caches.tar.gz
                   
```

The archive must unpack to a folder named `caches/` in the repository root, because the
code reads `./caches/RN50/<seed>/<dataset>/`. Size: about 0.8GB GB archive, X GB extracted.
`run_all.sh` skips extraction when the cache is present.

### Model Inference

All experiments should be run from the repository root after preparing the datasets and extracting the required features.

#### Base Configuration (Original Authors' Settings)

To reproduce the baseline results under the original authors' settings:

```bash
python main.py --config configs/[dataset_name].yaml --shot [shot_number] --seed [seed_number]
```

- `dataset_name`: one of `caltech101`, `dtd`, or `oxford_pets`
- `shot_number`: number of training examples per class (`1`, `4`, or `16`)
- `seed_number`: random seed (any integer, as long as corresponding features have been extracted)

**Note:** When run without additional arguments, `--timo_mode` defaults to `base` and `--junk` defaults to `0`. This configuration matches the original authors' experimental setup, running the baseline implementations (Tip-Adapter, APE, GDA-CLIP) without our TGI/IGT ablation modifications. The authors' original TIMO implementation is retained as a reference.

#### Single Experiment with Custom Settings

To evaluate a specific configuration with custom ablation modes or junk injection:

```bash
python main.py \
    --config configs/caltech101.yaml \
    --shot 1 \
    --seed 1 \
    --timo_mode base \
    --junk 0
```

**Arguments:**

- `--config`: Dataset configuration (`caltech101`, `dtd`, or `oxford_pets`)
- `--shot`: Training examples per class (`1`, `4`, or `16`)
- `--seed`: Random seed (`1`, `2`, or `3`)
- `--timo_mode`: Ablation mode (see table below, default `base`)
- `--junk`: Number of junk prompts to inject (default `0`)

#### Ablation Modes

The `--timo_mode` argument controls which components are enabled:

| Mode | Description |
| ------ | ------------- |
| `base` | Baseline implementations (default) |
| `tgi` | Text-Guided-Image component only |
| `igt` | Image-Guided-Text component only |
| `both` | Combined TGI and IGT components |

When timo_mode in `base`, `igt` or `tgi`, the ablation only affects Tip-Adapter and GDA-CLIP, this assuming that the model is already under base conditions configuration for base results. More on how to run end-to-end in the coming sections

#### Junk-Prompt Injection

The `--junk` argument injects irrelevant prompts from incorrect classes to test robustness. A value of `0` disables injection (default). Our experiments evaluate `2`, `4`, `8`, `16`, and `32` junk prompts.

Example with junk injection:

```bash
python main.py \
    --config configs/caltech101.yaml \
    --shot 1 \
    --seed 1 \
    --junk 8
```

### Running the Full Experiment Suite

To reproduce the experiments, run the commands below from the repository root. Each command evaluates all three datasets, shot settings (1, 4, and 16), and random seeds (1, 2, and 3).

#### Baseline

Run the baseline implementations without the additional TGI/IGT ablations or junk prompts:

```bash
bash run.sh
```

##### TGI Ablation

Evaluate the Text-Guided-Image (TGI) component for Tip-Adapter and GDA-CLIP :

```bash
bash run.sh tgi
```

##### IGT Ablation

Evaluate the Image-Guided-Text (IGT) component for Tip-Adapter and GDA-CLIP:

```bash
bash run.sh igt
```

##### Combined TGI and IGT

Evaluate both components together, TIMO used in Tip-Adapter and GDA-CLIP:

```bash
bash run.sh both
```

#### Junk-Prompt Robustness Experiments

Evaluate robustness to irrelevant prompts using 2, 4, 8, 16, and 32 additional junk prompts:

```bash
bash run.sh --all-noise
```

To run the junk-prompt experiments with a specific ablation, replace `base` with `tgi`, `igt`, or `both`. For example:

```bash
bash run.sh both --all-noise
```

To test a single junk-prompt setting, use `--noise` followed by the desired number:

```bash
bash run.sh base --noise 8
```

**Note:** The junk-prompt experiments extend the original paper and are not part of its reported experiments.

### Results and Analysis

All experiment runs write their raw results to the `outputs/` folder in the repository root. Each run mode gets its own subfolder, and within each subfolder there is one `.txt` file per logged metric.

```
outputs/
├── original/                 # clean baseline run (no junk, base mode)
│   ├── APE.txt
│   ├── GDA_CLIP.txt
│   ├── TIMO.txt
│   └── Tip_Adapter.txt
├── tgi/                      # TGI-only ablation
├── igt/                      # IGT-only ablation
├── both/                     # combined TGI + IGT ablation
├── junk_2/                   # baseline + 2 junk prompts
├── junk_4/                   # baseline + 4 junk prompts
└── ...                       # junk_8, junk_16, junk_32
```

Each `.txt` file stores one line per experiment in the format:

```
dataset_shots_seed: score
```

#### Generating CSV tables

The raw logs are hard to read, so use `analyse.py` to convert them into organised CSV tables. Run it from the repository root:

```bash
python analyse.py --mode base
```

This reads `outputs/base/`, aggregates the results across seeds, and writes a set of CSV files to `analysis/base/`. You can also specify a different results root if needed:

```bash
python analyse.py --mode base --root outputs
```

#### Ablation analysis (`--mode base|tgi|igt|both`)

For each of the four ablation modes, `analyse.py` produces the following files inside `analysis/<mode>/`:

| File | Contents |
| ------ | ---------- |
| `all_results.csv` | One row per experiment, one column per method. |
| `summary_by_dataset_shots.csv` | Mean and standard deviation across seeds, per dataset and shot count. |
| `method_comparison.csv` | Mean score per method, with the best method flagged. |
| `overall_method_ranking.csv` | Overall mean accuracy and spread per method. |

Example:

```bash
python analyse.py --mode both
```

#### Junk-prompt analysis (`--mode junk`)

This mode compares every junk-prompt run against the clean baseline and computes how much accuracy each method loses as noise increases.

```bash
python analyse.py --mode junk
```

To run for a specific noise level you add the `--noise-levels` flag. It can take more than one noise level, example you can have it as `--noise-levels 2` for 2 or `--noise-levels 2 16` for 2 and 16

```bash
python analyse.py --mode junk --noise-levels 2 16
```

It writes the following files to `analysis/junk_prompt/`:

| File | Contents |
| ------ | ---------- |
| `noise_degradation.csv` | Every noisy run matched to its clean baseline, with the accuracy drop. |
| `method_robustness_summary.csv` | Mean drop and relative drop per method and noise level. |
| `degradation_by_dataset_shots.csv` | Breakdown of the drop by dataset, shot count, and noise level. |
| `timo_shot_sensitivity.csv` | TIMO accuracy drop by noise level and shot count. |
| `timo_most_affected_shots.csv` | The shot setting with the largest drop at each noise level. |
| `junk_weight_analysis.csv` | Raw junk-weight logs exported for inspection. |

**Note:** The junk-prompt analysis assumes the clean baseline lives in `outputs/original/`. Make sure you have run the clean baseline (`bash run.sh base`) before running `--mode junk`.

### Numerical Results

We provide the numerical results for base, ablation and junk prompt under output/archive for reference

### Run the Complete Pipeline

Execute:

```bash
bash run-all.sh
```

The script performs the following stages in order:

1. **Dataset setup:** Prepares the datasets required for the experiments.
2. **Feature extraction:** Extracts and caches the CLIP image and text features.
3. **Baseline experiments:** Runs the baseline configuration.
4. **TGI experiments:** Evaluates Text-Guided-Image.
5. **IGT experiments:** Evaluates Image-Guided-Text.
6. **Combined experiments:** Evaluates the combination of TGI and IGT.
7. **Junk-prompt experiments:** Evaluates robustness to additional incorrect-class prompts using 2, 4, 8, 16, and 32 junk prompts.
8. **Analysis:** Processes the experiment logs and generates summary tables for the baseline, ablation, and junk-prompt experiments.

The experiments cover all three datasets, three shot counts, and three random seeds. Feature extraction is performed separately so cached features can be reused across experiment modes.

> **Note:** The full pipeline may take several hours, depending on the hardware and whether feature extraction has already been completed.

### Resuming an Interrupted Run

The pipeline records completed stages in `.pipeline_state/`. If a stage fails, fix the underlying problem and run the script again:

```bash
bash run-all.sh
```

Stages with completion markers are skipped, allowing the pipeline to resume without repeating every completed stage.

> **Note:** The completion markers are local pipeline state, not experiment results. To rerun a completed stage, remove its corresponding marker from `.pipeline_state/` before running the script again. Be aware that rerunning experiments may append duplicate entries to existing logs, depending on the logging implementation.

## Acknowledgements

This repo is build upon previous amazing repos include [TIP-Adapter](https://github.com/gaopengcuhk/Tip-Adapter/), [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/CoOp/), [SuS-X](https://github.com/vishaal27/SuS-X), [CuPL](https://github.com/sarahpratt/CuPL) [APE](https://github.com/yangyangyang127/APE) and [GDA-CLIP](https://github.com/mrflogs/ICLR24). Thanks for their contributions to the field.
