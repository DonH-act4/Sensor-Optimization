# Sensor Optimization Project Handoff

## 1. Project objective

This project studies **fixed sensor selection for leak localization in the
Anderson Road water-pipe topology**. There are 12 pressure-sensor channels and
13 candidate leak pipes.

The current prediction task is intentionally simple:

- Input: all 12 pressure channels, or a selected subset of the 12 channels.
- Signal length: 3334 time samples per channel.
- Target: `PipeID` only, a 13-class classification problem.
- Loss: cross entropy.
- `xL` and `sL` are not prediction targets in the current baseline.

Label meanings:

- `PipeID`: integer 1--13 identifying the leaking pipe.
- `xL`: relative leak position along the current pipe, measured from that
  pipe's inlet node. Physical distance from inlet is `xL * pipe_length`.
- `sL`: leak size.

Do **not** revive or extend the old TAS code. The old `TAS.py`, `act1/`, and
`act2/` directories are historical experiments and are not the method for this
project.

## 2. Topology and channel context

The 13 pipes are:

| PipeID | Pipe | Length (m) | DN (mm) |
|---:|:---:|---:|---:|
| 1 | AB | 200 | 300 |
| 2 | BC | 100 | 300 |
| 3 | BD | 300 | 300 |
| 4 | DE | 150 | 300 |
| 5 | EF | 100 | 300 |
| 6 | EG | 300 | 300 |
| 7 | DH | 250 | 450 |
| 8 | HJ | 150 | 450 |
| 9 | KJ | 120 | 300 |
| 10 | KI | 200 | 300 |
| 11 | IH | 120 | 300 |
| 12 | KL | 150 | 150 |
| 13 | JM | 200 | 600 |

The data have 12 measurement channels. Preserve the original MATLAB channel
order unless a verified channel-to-node map is added to this file. Never infer
or silently reorder channels from topology labels.

When reporting selected sensors, use 1-based channel IDs `1..12` for the user
and W&B artifacts. Internally, PyTorch indexing is 0-based.

## 3. Dataset structure

The four prepared MATLAB files are placed directly in the repository root:

```text
SensorOptimization/
  AGENTS.md
  sensor_dataloader.py
  train_all_sensors.py
  analyze_channel_attribution.py
  select_sensors_logdet.py
  train_selected_sensors.py
  Dataset_4000_Clean.mat
  Dataset_4000_SNR_0.mat
  Dataset_4000_SNR_1.mat
  Dataset_4000_SNR_5.mat
```

Do not expect a `Data for DON/` subdirectory on the server. Pass
`--data-dir .` to scripts.

Each file contains 4000 paired samples. An older topology figure says 5000
samples; trust the files actually present and print the sample count before
training.

Expected struct layout:

```text
Dataset: 1 x 4000 struct
Dataset(i).Signals:             [3334, 12]
Dataset(i).LeakInfo_ID_xL_sL:   [PipeID, xL, sL]
```

Expected signal fields:

```text
clean: Signals
snr0:  NoisySignals_SNR0dB
snr1:  NoisySignals_SNR1dB
snr5:  NoisySignals_SNR5dB
```

The noisy files are paired versions of the same underlying clean samples and
must retain the same sample order and labels.

Before any long server run:

```bash
find . -maxdepth 1 -type f -name 'Dataset_4000_*.mat' -print
python train_all_sensors.py --help
```

If the filenames differ, update `DATASET_FILES` / dataset resolution in
`sensor_dataloader.py`. Do not duplicate or manually reshuffle the data.

## 4. Non-negotiable experimental protocol

The user explicitly chose a simple 80/20 protocol with no validation set:

- Stratified 80% training / 20% testing by `PipeID`.
- Fixed random seed and fixed number of epochs.
- Use the exact same split indices for clean, SNR 0, SNR 1, and SNR 5.
- Train a new model from scratch for every noise level unless explicitly
  resuming the same run/checkpoint.
- Use the same architecture, initialization seed, optimizer, and
  hyperparameters when comparing noise levels.
- Any sensor ranking used to create a reduced-sensor model must be computed
  from training samples only. The held-out 20% must not influence selection.

Strict reporting rule:

- Do not select the best epoch using test performance.
- Save and evaluate the final pre-specified epoch.
- If test metrics are logged during training for diagnostics, mark the run as
  exploratory and do not present that test curve as an independent model
  selection procedure.
- If test results are used to modify architecture or hyperparameters, they
  cease to be an independent test. Do not silently tune on test results.

Normalization:

- Compute one mean and standard deviation per sensor channel.
- Compute statistics using the 80% training split only.
- Apply those training statistics to both train and test samples.
- Default preprocessing is channel-wise Z-score normalization on raw pressure
  signals.
- Do not normalize each sample independently; that may remove physically useful
  amplitude information related to leak behavior.
- Optional first-time-point subtraction exists but is disabled by default. Do
  not change this preprocessing between methods being compared.

## 5. Mandatory Weights & Biases logging

Every experiment and analysis result must be recorded in Weights & Biases
(W&B), in addition to being saved locally. Do not silently disable W&B and do
not rely only on terminal output or local CSV/JSON files.

Use one W&B project for the complete study:

```text
project: SensorOptimization
```

Configure the user/team entity through an environment variable or command-line
argument. Never hardcode an API key or commit credentials.

On a new server:

```bash
wandb login
# or export WANDB_API_KEY through the server's secret manager
```

Required run organization:

```text
group:    baseline | attribution | logdet | gating | random_topk
job_type: train | analysis | sensor_selection
name examples:
  baseline-v6-clean-seed42-e0300
  baseline-v7-clean-seed42-e0300
  baseline-v6-snr0-seed42-e0300
  attribution-clean-seed42
  logdet-snr1-seed42
  selected-attribution-snr5-K3-v6-seed42
  gating-snr5-K3-seed42
```

Every training run must store the following W&B config fields:

- method and dataset/noise level;
- seed and path/name of the reused split-index file;
- train/test ratio;
- selected channel IDs and K, or all 12 for baseline;
- epochs, batch size, optimizer, learning rate, weight decay, and scheduler;
- normalization and optional baseline-subtraction settings;
- model architecture/version;
- Git commit hash when the project is under Git.

Per-epoch training logs must include at least:

```text
epoch
train/loss
train/accuracy
learning_rate
```

Default formal protocol has no validation set and final-only test evaluation.
`train_all_sensors.py` also has `--eval-test-every` for debugging curves:

```text
test_epoch/loss
test_epoch/accuracy
test_epoch/macro_f1
```

Use that option only for exploratory diagnosis. For strict final comparison,
keep `--eval-test-every 0` and log final:

```text
test/loss
test/accuracy
test/macro_f1
test/per_pipe_precision
test/per_pipe_recall
test/per_pipe_f1
```

Upload these as W&B artifacts or tables where applicable:

- final model checkpoint;
- exact `split_indices_80_20.npz`;
- classification report and confusion matrix;
- test predictions;
- sensor rankings, selected-channel lists, and Top-K performance tables;
- attribution tables and figures;
- gating logits and selection-stability summaries;
- the final consolidated CSV and paper figures.

Continue saving local files because they are useful for reproducibility and
artifact recovery. If the server has no outbound connection, use
`WANDB_MODE=offline`, keep the generated `wandb/` directory, and run
`wandb sync` later. An offline run is acceptable; a missing run is not.

## 6. Current primary training code

Primary script:

```text
train_all_sensors.py
```

Shared dataloader:

```text
sensor_dataloader.py
```

The historical `cnn_baseline_13class.py` / `cnn_baseline_26class.py` scripts are
not the active implementation in this workspace. Do not create new work on top
of them unless the user explicitly asks for backward compatibility.

### Architecture registry

The active registry in `train_all_sensors.py` is:

```text
v1: all_sensor_cnn_v1_gap
v2: all_sensor_cnn_v2_temporal8
v3: all_sensor_resnet_v3_gapmax
v4: all_sensor_cnn_v4_wide_gap
v5: all_sensor_patch_transformer_v5
v6: all_sensor_tcn_v6_dilated
v7: all_sensor_tcn_v7_dilated6
```

Default architecture remains:

```text
v1
```

This is deliberate: `v1` is the conservative baseline. `v6` is the stronger
8-block TCN model and should be selected explicitly with `--architecture v6`.
`v7` is a smaller 6-block TCN intended to test whether `v6` is over-capacity.

### v1 baseline CNN

```text
Input [B, C, 3334]
Conv1d C->64,   kernel 9 + BN + ReLU + MaxPool
Conv1d 64->128, kernel 7 + BN + ReLU + MaxPool
Conv1d 128->256,kernel 5 + BN + ReLU + MaxPool
Conv1d 256->384,kernel 3 + BN + ReLU + MaxPool
AdaptiveAvgPool1d(1)
Linear 384->192 + ReLU + Dropout(0.4)
Linear 192->13
```

`C` is 12 for all-sensor training and `K` for selected-sensor retraining.

Observed pilot result on clean data: `v1` can reach roughly macro-F1 `0.71`
after a long 300-epoch run, but it learns much more slowly than `v6`.

### v6 attribution-friendly TCN

`v6` is a non-causal residual TCN for full-transient classification, not
forecasting.

```text
Input [B, C, 3334]
Stem:
  Conv1d C->64, kernel 7, padding 3, no bias
  BatchNorm1d
  GELU

Residual dilated blocks, no temporal downsampling:
  channels: 64 -> 96 -> 128 -> 160 -> 192 -> 192 -> 192 -> 192 -> 192
  dilations: 1, 2, 4, 8, 16, 32, 64, 128
  each block:
    dilated Conv1d + BN + GELU + Dropout(0.15)
    dilated Conv1d + BN + GELU + Dropout(0.15)
    residual shortcut, with 1x1 projection if channel count changes

Classifier:
  AdaptiveAvgPool1d(1)
  Linear 192->256
  GELU
  Dropout(0.25)
  Linear 256->13
```

Important v6 design choices:

- No max pooling in the classifier path.
- No temporal downsampling through the TCN stack.
- Non-causal same-length padding, because the full transient is available at
  inference.
- Pure differentiable PyTorch layers: Conv1d, BatchNorm, GELU, Dropout,
  residual addition, global average pooling, Linear.
- Use `model.eval()` during attribution so dropout is deterministic.

Observed pilot result: clean `v6` reached about macro-F1 `0.97` after 300
epochs, substantially above the `v1` pilot. This is strong evidence that the
pressure transient contains learnable long-range temporal structure. It does
not by itself prove that the resulting sensor ranking is stable; Top-K
retraining is still required.

Memory note:

- `v6` keeps temporal length 3334 through the residual stack, so activation
  memory is high.
- On the RTX 5080 16 GB server, `--batch-size 128` caused CUDA OOM.
- Use `--batch-size 64`; reduce to 32 or 16 if necessary.

### v7 smaller TCN

`v7` keeps the same attribution-friendly design as `v6`, but removes the last
two high-dilation residual blocks.

```text
Input [B, C, 3334]
Stem:
  Conv1d C->64, kernel 7, padding 3, no bias
  BatchNorm1d
  GELU

Residual dilated blocks, no temporal downsampling:
  channels: 64 -> 96 -> 128 -> 160 -> 192 -> 192 -> 192
  dilations: 1, 2, 4, 8, 16, 32

Classifier:
  AdaptiveAvgPool1d(1)
  Linear 192->256
  GELU
  Dropout(0.25)
  Linear 256->13
```

Use `v7` when testing whether `v6` is unnecessarily large for sensor
selection. `v7` keeps smooth channel attribution behavior because it still uses
global average pooling and differentiable TCN blocks.

## 7. Standard training commands

Short clean sanity run:

```bash
python train_all_sensors.py \
  --data-dir . \
  --output-dir results_tcn_v6 \
  --datasets clean \
  --epochs 50 \
  --scheduler-t-max 300 \
  --batch-size 64 \
  --architecture v6 \
  --checkpoint-every 50
```

Full clean v6 run:

```bash
python train_all_sensors.py \
  --data-dir . \
  --output-dir results_tcn_v6 \
  --datasets clean \
  --epochs 300 \
  --scheduler-t-max 300 \
  --batch-size 64 \
  --architecture v6 \
  --checkpoint-every 50
```

Smaller clean v7 comparison run:

```bash
python train_all_sensors.py \
  --data-dir . \
  --output-dir results_tcn_v7 \
  --datasets clean \
  --epochs 300 \
  --scheduler-t-max 300 \
  --batch-size 64 \
  --architecture v7 \
  --checkpoint-every 50
```

Run noisy conditions after clean is complete:

```bash
python train_all_sensors.py \
  --data-dir . \
  --output-dir results_tcn_v6 \
  --datasets snr0 snr1 snr5 \
  --epochs 300 \
  --scheduler-t-max 300 \
  --batch-size 64 \
  --architecture v6 \
  --checkpoint-every 50
```

For diagnostic test curves only:

```bash
python train_all_sensors.py \
  --data-dir . \
  --output-dir results_tcn_v6 \
  --datasets snr0 \
  --epochs 300 \
  --scheduler-t-max 300 \
  --batch-size 64 \
  --architecture v6 \
  --checkpoint-every 50 \
  --eval-test-every 10
```

Do not use `--eval-test-every` to select the best epoch for formal results.

## 8. Checkpointing and resume behavior

`train_all_sensors.py` saves numbered checkpoints, `latest_checkpoint.pt`, and
`final_model.pt`. Checkpoints include model state, optimizer/scheduler state
when available, epoch, dataset, architecture, normalization metadata, split
metadata, and W&B run ID.

Resume semantics:

- `--resume-checkpoint PATH` continues from an existing checkpoint.
- In resume mode, `--epochs` means additional epochs.
- Accumulated epoch numbers are used in output names and W&B logs.
- The checkpoint's W&B run ID is reused by default when available.
- Use `--new-wandb-run` to force a separate run.
- Use `--resume-training-state` only when you intentionally want the old
  optimizer/scheduler state. If the old cosine schedule already annealed the
  learning rate close to zero, usually do not restore scheduler state.

Example: continue a v6 run from epoch 50 to epoch 300 with the learning-rate
schedule continuing by accumulated epoch:

```bash
python train_all_sensors.py \
  --data-dir . \
  --output-dir results_tcn_v6 \
  --datasets clean \
  --epochs 250 \
  --scheduler-t-max 300 \
  --batch-size 64 \
  --architecture v6 \
  --resume-checkpoint results_tcn_v6/clean_v6_e0050/checkpoint_clean_v6_epoch0050.pt \
  --checkpoint-every 50
```

If running over SSH, use `tmux`, `screen`, or `nohup` so training continues
after disconnect.

Example with `tmux`:

```bash
tmux new -s sensor
python train_all_sensors.py --data-dir . --output-dir results_tcn_v6 --datasets clean --epochs 300 --scheduler-t-max 300 --batch-size 64 --architecture v6 --checkpoint-every 50
# detach: Ctrl-b then d
# reattach later:
tmux attach -t sensor
```

## 9. Learning-rate scheduling notes

Default schedule is cosine annealing. `--scheduler-t-max` controls how many
epochs the cosine curve uses.

Implemented option:

```text
--constant-after-epoch N
```

This means:

- use cosine schedule through accumulated epoch `N`;
- then keep the learning rate fixed at the epoch-`N` learning-rate value.

Example:

```bash
python train_all_sensors.py \
  --data-dir . \
  --output-dir results_v1_long \
  --datasets clean \
  --epochs 600 \
  --scheduler-t-max 600 \
  --constant-after-epoch 300 \
  --batch-size 128 \
  --architecture v1 \
  --checkpoint-every 50
```

Use this for controlled long runs. Do not change the schedule for only one SNR
condition if the goal is a fair across-noise comparison.

## 10. Phase B -- neural-network channel attribution

Implemented script:

```text
analyze_channel_attribution.py
```

Purpose: analyze a trained full-channel model and produce one scalar importance
score per sensor channel. This is an analysis of the trained model, not a
standalone selection algorithm and not an unsupervised method.

Current method:

- Captum Integrated Gradients.
- Work on normalized inputs.
- Zero baseline, because zero is the train-set channel mean after Z-score
  normalization.
- Target is the true `PipeID` logit, not softmax probability.
- Attribution tensor shape: `[N, C, 3334]`.
- Channel aggregation:

```text
sample_channel_score = mean_time(abs(attribution))
global_channel_score = mean_samples(sample_channel_score)
```

Outputs are JSON files containing channel scores and rankings. Per-sample
scores are saved only when explicitly requested.

Run attribution for each condition using training samples only:

```bash
python analyze_channel_attribution.py --data-dir . --output-dir results_attribution --dataset clean --checkpoint results_tcn_v6/clean_v6_e0300/final_model.pt --architecture v6 --split train --batch-size 2 --n-steps 32
python analyze_channel_attribution.py --data-dir . --output-dir results_attribution --dataset snr0  --checkpoint results_tcn_v6/snr0_v6_e0300/final_model.pt  --architecture v6 --split train --batch-size 2 --n-steps 32
python analyze_channel_attribution.py --data-dir . --output-dir results_attribution --dataset snr1  --checkpoint results_tcn_v6/snr1_v6_e0300/final_model.pt  --architecture v6 --split train --batch-size 2 --n-steps 32
python analyze_channel_attribution.py --data-dir . --output-dir results_attribution --dataset snr5  --checkpoint results_tcn_v6/snr5_v6_e0300/final_model.pt  --architecture v6 --split train --batch-size 2 --n-steps 32
```

Expected output pattern:

```text
results_attribution/{clean,snr0,snr1,snr5}/
  {dataset}_{architecture}_ig_train.json
```

Important distinction:

- Attribution on the held-out test set may be reported as explanation of final
  model behavior.
- If an attribution ranking is later used to choose Top-K sensors and retrain a
  model, generate that ranking using the training split only.

## 11. Phase C -- unsupervised log-determinant sensor selection

Implemented script:

```text
select_sensors_logdet.py
```

This is a separate unsupervised selection method. It must not use `PipeID`,
`xL`, or `sL`.

The intended criterion is the Gaussian joint-entropy/log-determinant objective:

```text
H(S) = 0.5 * log det(Cov(X_S) + epsilon * I)
gain(j | S) = H(S union {j}) - H(S)
```

Strict terminology: this is a log-determinant Gaussian entropy or D-optimal
information criterion. Do not claim it is label mutual information `I(X;Y)`,
because labels are intentionally unused.

Implementation procedure:

1. Load only the 80% training samples for that condition.
2. Apply exactly the same train-only channel normalization used by the neural
   network.
3. Reshape normalized training signals from `[N, 12, 3334]` into observations
   `[N * 3334, 12]`.
4. Compute the regularized 12 x 12 sensor covariance matrix.
5. Start with an empty set and greedily add the channel with the largest
   marginal log-det gain.
6. Save the complete ranking of all 12 channels and every marginal gain.

Run log-det rankings:

```bash
python select_sensors_logdet.py --data-dir . --output-dir results_logdet --dataset clean --batch-size 16
python select_sensors_logdet.py --data-dir . --output-dir results_logdet --dataset snr0  --batch-size 16
python select_sensors_logdet.py --data-dir . --output-dir results_logdet --dataset snr1  --batch-size 16
python select_sensors_logdet.py --data-dir . --output-dir results_logdet --dataset snr5  --batch-size 16
```

Expected output pattern:

```text
results_logdet/{clean,snr0,snr1,snr5}/
  {dataset}_logdet_train.json
```

Because time samples within one transient are correlated, the log-det criterion
is an engineering information criterion rather than an exact independent-sample
entropy estimate. State this limitation in reports.

## 12. Phase C evaluation -- selected-sensor retraining

Implemented script:

```text
train_selected_sensors.py
```

Purpose: read a ranking JSON, choose Top-K sensor channels, rebuild the same
architecture with `in_channels=K`, retrain from scratch, and evaluate once on
the same held-out test split.

Example: train v6 with Top-3 attribution sensors on clean data:

```bash
python train_selected_sensors.py \
  --data-dir . \
  --output-dir results_selected_sensors \
  --dataset clean \
  --selection-json results_attribution/clean/clean_v6_ig_train.json \
  --top-k 3 \
  --architecture v6 \
  --epochs 300 \
  --scheduler-t-max 300 \
  --batch-size 64
```

Example: train v6 with Top-5 log-det sensors on SNR 0:

```bash
python train_selected_sensors.py \
  --data-dir . \
  --output-dir results_selected_sensors \
  --dataset snr0 \
  --selection-json results_logdet/snr0/snr0_logdet_train.json \
  --top-k 5 \
  --architecture v6 \
  --epochs 300 \
  --scheduler-t-max 300 \
  --batch-size 64
```

Suggested K values for systematic comparison:

```text
K = 1, 2, 3, 4, 5, 6, 8, 10, 12
```

Do not pick one K by maximizing test performance. Report the complete
performance-versus-budget curve.

## 13. Phase D -- global gating sensor selection

Not implemented yet.

This is a supervised, task-driven method independent of log-det and attribution.
The gate must learn a **fixed physical sensor subset**.

Do not use an input-dependent gate. A selector that first reads all 12 channels
and then chooses channels per sample is invalid for physical sensor placement.

Required global Top-K design:

1. Create one learnable global logit per channel:

   ```text
   alpha: learnable vector [12]
   ```

2. Place the gate at the model input, before the classifier consumes sensor
   signals.
3. For a requested budget K, use a differentiable hard Top-K mechanism:
   - training: add Gumbel noise to global logits and use a soft relaxation;
   - forward pass: hard K-hot mask;
   - backward pass: straight-through soft gradient;
   - inference: deterministic Top-K of `alpha`, with no Gumbel noise.
4. The same mask is shared by every sample in the dataset.
5. Mask unselected channels before the first convolution, or physically gather
   only selected channels and construct the classifier with `in_channels=K`.
6. Optimize only the 13-class cross-entropy task loss, plus any explicitly
   documented gate regularization.
7. Use a pre-specified temperature schedule from high/soft to low/hard.
8. Train a separate gate+model from scratch for each SNR condition and K.
9. At the end, save global logits, deterministic selected indices, selection
   order, final model, and test metrics.

Use at least 5 random seeds for gating because discrete selection can be
unstable. Report selection frequency and pairwise Jaccard stability across
seeds.

## 14. Final comparison

The three sensor-importance approaches are intentionally different:

1. **Attribution:** what the trained all-sensor model relies on.
2. **Unsupervised log-det:** which sensors preserve the most nonredundant
   signal information without labels.
3. **Global gating:** which fixed sensors maximize supervised PipeID
   classification performance.

They should not be merged into one method in the first study.

For each SNR and K, compare:

```text
All 12 sensors (upper/reference baseline)
Random Top-K (repeat across seeds)
Attribution Top-K + retrained model
Unsupervised log-det Top-K + retrained model
Supervised global-gating Top-K
```

Use the same split, preprocessing, architecture family, training epochs,
optimizer, and evaluation metrics as closely as possible. In addition to
predictive metrics, compare:

- selected channel IDs;
- ranking agreement, such as Spearman correlation where appropriate;
- Top-K overlap/Jaccard similarity;
- ranking changes across SNR;
- gating stability across seeds;
- accuracy/Macro-F1 versus number of sensors.

The final scientific question is not merely "which method has the highest
accuracy." It is:

> Do signal-information, neural-network attribution, and task-driven global
> gating identify the same physically important pressure sensors, and how does
> that agreement and localization performance change with SNR and sensor
> budget?

## 15. Git and server workflow

The intended workflow is:

1. Modify and test code locally.
2. Commit and push source code to GitHub.
3. Pull the repository on the server.
4. Keep the four large `.mat` files available locally in the server project
   root, but do not commit them to ordinary Git history.
5. Run experiments on the server and record all results in W&B.

`.gitignore` should exclude at least:

```text
*.mat
wandb/
results_*/
*.pt
*.pth
__pycache__/
.DS_Store
._*
```

Commit source code, `AGENTS.md`, lightweight configuration files, and plotting
scripts. Do not commit:

- MATLAB datasets unless Git LFS is deliberately configured;
- W&B API keys or shell environment files containing credentials;
- model checkpoints and generated result directories;
- machine-specific absolute paths.

Record the Git commit hash in every W&B run so that a server result can be
traced back to the exact code. Do not run destructive Git commands on the
server to discard uncommitted experiment changes.

On the server, update source code with:

```bash
git pull
```

If generated results exist locally, do not run destructive Git commands such as
`git reset --hard` unless the user explicitly asks for that operation.

## 16. Current implementation status

Implemented in this workspace:

- `sensor_dataloader.py`: shared MATLAB loading, fixed stratified 80/20 split,
  train-only channel normalization, optional channel subset support.
- `train_all_sensors.py`: all-sensor 13-class training with architecture
  registry `v1`--`v7`, W&B logging, checkpoints, resume support, optional
  diagnostic test curves.
- `analyze_channel_attribution.py`: Integrated Gradients channel attribution
  producing per-channel JSON rankings for each noise condition.
- `select_sensors_logdet.py`: unsupervised train-only Gaussian log-det channel
  ranking.
- `train_selected_sensors.py`: Top-K selected-sensor retraining from attribution
  or log-det JSON rankings.

Not implemented yet:

- global exact Top-K gating model;
- random Top-K repeated baseline;
- consolidated comparison table/plotting script;
- multi-seed stability analysis for rankings and gating.

Current empirical notes from pilot runs:

- `v2` and `v3` were weaker/unstable in early tests and should not be treated
  as the main result.
- `v1` is a stable conservative CNN baseline but learns slowly.
- `v6` TCN is currently the strongest all-sensor model and is suitable for
  attribution because it uses differentiable layers and global average pooling.
- `v7` is the smaller 6-block TCN comparison model. Use it to test whether
  `v6` has more capacity than needed before committing to sensor-selection
  runs.
- Strong all-sensor `v6` performance does not prove that a small sensor subset
  is sufficient. The scientific evidence for sensor selection must come from
  Top-K retraining and, later, gating/multi-seed stability.
