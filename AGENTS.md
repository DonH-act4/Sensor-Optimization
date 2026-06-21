# Sensor Optimization Project Handoff

## 1. Project objective

This project studies **fixed sensor selection for leak localization in the
Anderson Road water-pipe topology**. There are 12 pressure-sensor channels and
13 candidate leak pipes.

The immediate prediction task is deliberately simple:

- Input: all or a selected subset of 12 pressure channels.
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

## 3. Dataset structure

In the new server environment, the four prepared MATLAB files are placed
**directly in the repository/project root**, beside the Python scripts:

```text
SensorOptimization/
  AGENTS.md
  cnn_baseline_13class.py
  Dataset_4000_Clean.mat
  Dataset_4000_SNR_0.mat
  Dataset_4000_SNR_1.mat
  Dataset_4000_SNR_5.mat
```

Do not expect a `Data for DON/` subdirectory on the server. Pass
`--data-dir .` to scripts, or change their default data directory to the project
root without changing the filenames.

Each file contains 4000 paired samples in MATLAB v7.3 format. An older topology
figure says 5000 samples; trust the files actually present and print the sample
count before training.

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
must retain the same sample order and labels. On a new server, verify all files
and fields before a long run:

```bash
find . -maxdepth 1 -type f -name 'Dataset_4000_*.mat' -print
python cnn_baseline_13class.py --help
```

If the filenames differ, update `resolve_dataset()` in
`cnn_baseline_26class.py`. Do not duplicate or manually reshuffle the data.

## 4. Non-negotiable experimental protocol

The user explicitly chose a simple 80/20 protocol with no validation set:

- Stratified 80% training / 20% testing by `PipeID`.
- Fixed random seed and fixed number of epochs.
- Do not inspect the test set at every epoch.
- Do not select the best epoch using test performance.
- Save and evaluate the final pre-specified epoch only.
- Use the exact same sample indices for clean, SNR 0, SNR 1, and SNR 5.
- Train a new model from scratch for every noise level.
- Use the same architecture, initialization seed, optimizer, and hyperparameters
  for every noise level.
- Any sensor ranking used to create a reduced-sensor model must be computed
  from training samples only. The held-out 20% must not influence selection.

If test results are used to modify architecture or hyperparameters, they cease
to be an independent test. Do not silently tune on test results.

Normalization:

- Compute one mean and standard deviation per sensor channel.
- Compute statistics using the 80% training split only.
- Apply those training statistics to both train and test samples.
- Default baseline currently performs channel-wise Z-score normalization on
  raw pressure signals.
- Do not normalize each sample independently; that may remove physically useful
  amplitude information related to leak behavior.
- Optional first-time-point subtraction exists but is disabled by default. Do
  not change this preprocessing between methods being compared.

## 5. Mandatory Weights & Biases logging

**Every experiment and analysis result must be recorded in Weights & Biases
(W&B), in addition to being saved locally.** Do not silently disable W&B and do
not rely only on terminal output or local CSV files.

Use one W&B project for the complete study:

```text
project: SensorOptimization
```

Configure the user/team entity through an environment variable or command-line
argument. Never hardcode an API key or commit credentials. On a new server:

```bash
wandb login
# or export WANDB_API_KEY through the server's secret manager
```

Required run organization:

```text
group:    baseline | attribution | logdet | gating | random_topk
job_type: train | analysis | sensor_selection
name examples:
  baseline-clean-seed42
  baseline-snr0-seed42
  attribution-clean-seed42
  logdet-snr1-K5-seed42
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

There is no validation set in the current protocol. Do not log test metrics per
epoch. Evaluate the test set once after the fixed final epoch, then log:

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
artifact recovery, but upload/link them to the corresponding W&B run. If the
server has no outbound connection, use `WANDB_MODE=offline`, keep the generated
`wandb/` directory, and run `wandb sync` later. An offline run is acceptable; a
missing run is not.

## 6. Phase A -- all-sensor CNN baseline (implemented)

Primary script:

```text
cnn_baseline_13class.py
```

Current model:

```text
Input [B, 12, 3334]
Conv1d 12->64,  kernel 9 + BN + ReLU + MaxPool
Conv1d 64->128, kernel 7 + BN + ReLU + MaxPool
Conv1d 128->256,kernel 5 + BN + ReLU + MaxPool
Conv1d 256->384,kernel 3 + BN + ReLU + MaxPool
AdaptiveAvgPool1d(1)
Linear 384->192 + ReLU + Dropout(0.4)
Linear 192->13
```

Training uses `CrossEntropyLoss`, AdamW, and a pre-specified cosine learning-rate
schedule. It does not use contrastive learning. Contrastive learning is not part
of the baseline and is not currently planned because the goal is to measure
each SNR condition independently.

Run all four independent experiments:

```bash
python cnn_baseline_13class.py \
  --data-dir . \
  --datasets clean snr0 snr1 snr5 \
  --epochs 100 \
  --batch-size 32
```

For a server GPU, increase batch size only after confirming memory use. Do not
change batch size for just one SNR condition.

Outputs:

```text
results_cnn_13class/
  split_indices_80_20.npz
  summary.csv
  clean/
  snr0/
  snr1/
  snr5/
```

Each dataset directory contains:

```text
final_model.pt
train_history.csv
test_metrics.json
classification_report.csv
confusion_matrix.csv
test_predictions.csv
```

Required baseline reporting:

- Accuracy.
- Macro-F1.
- Per-pipe precision, recall, and F1.
- 13 x 13 confusion matrix.
- Training time and inference time if later comparing deployment cost.

Before starting later methods, confirm that all four baseline runs completed
and that `summary.csv` contains clean, snr0, snr1, and snr5. Also confirm that
four corresponding W&B runs finished successfully and contain their model and
evaluation artifacts.

## 7. Phase B -- neural-network channel attribution (SHAP-style analysis)

This is an **analysis of the full 12-channel CNN**, not a gating method and not
an unsupervised method.

Recommended implementation:

1. Load each trained `final_model.pt` from Phase A.
2. Use Captum `IntegratedGradients` or `GradientShap`. These are practical
   gradient-based SHAP-style attribution methods for a long time series.
3. Work on normalized inputs. A zero baseline then corresponds to the
   training-set mean of each normalized channel.
4. Attribute the logit of the true `PipeID` class for each sample.
5. Produce an attribution tensor `[samples, 12, 3334]`.
6. Convert it to a channel score without allowing positive and negative values
   to cancel:

   ```text
   sample_channel_score = mean_time(abs(attribution))
   global_channel_score = mean_samples(sample_channel_score)
   ```

7. Report both:
   - global importance over all samples;
   - class-conditional importance for each of the 13 pipes.
8. Repeat independently for clean, SNR 0, SNR 1, and SNR 5.
9. Save raw per-sample channel scores, rankings, plots, and the attribution
   configuration (method, baseline, integration steps, seed).

Important distinction:

- Attribution on the held-out test set may be reported as explanation of final
  model behavior.
- If a SHAP ranking is later used to choose Top-K sensors and retrain a model,
  generate that ranking using the training split only. Otherwise the selected
  sensor model leaks test information.

Recommended future script and outputs:

```text
analyze_channel_attribution.py
results_attribution/{clean,snr0,snr1,snr5}/
  global_channel_ranking.csv
  per_pipe_channel_importance.csv
  per_sample_channel_importance.csv
  channel_importance.png
```

For a fair sensor-selection comparison, take Top-K channels from the
training-only attribution ranking, retrain the same CNN with only those
channels, and evaluate once on the same held-out test indices.

## 8. Phase C -- unsupervised MI/log-determinant sensor selection

This is a separate unsupervised selection method. It must not use `PipeID`,
`xL`, or `sL`.

The intended criterion is the Gaussian joint-entropy/log-determinant objective:

```text
H(S) = 0.5 * log det(Cov(X_S) + epsilon * I)
gain(j | S) = H(S union {j}) - H(S)
```

Strict terminology: this is a log-determinant Gaussian entropy or D-optimal
information criterion. Do not claim it is label mutual information
`I(X;Y)`, because labels are intentionally unused.

Implementation procedure for each noise condition:

1. Load only the 80% training samples for that condition.
2. Apply exactly the same train-only channel normalization used by the CNN.
3. Reshape normalized training signals from `[N, 12, 3334]` into observations
   `[N * 3334, 12]`.
4. Compute the regularized 12 x 12 sensor covariance matrix.
5. Start with an empty set and greedily add the channel with the largest
   marginal log-det gain.
6. Save the complete ranking of all 12 channels and every marginal gain.
7. Evaluate Top-K subsets, suggested `K = 1, 2, 3, 4, 5, 6, 8, 10, 12`.
8. For every K, train the same downstream CNN from scratch using only those K
   input channels, the same train/test indices, fixed epochs, and otherwise
   unchanged hyperparameters.

Because time samples within one transient are correlated, the log-det criterion
is an engineering information criterion rather than an exact independent-sample
entropy estimate. State this limitation. Later extensions may compute separate
early-transient, mid-transient, and late-transient scores, but the first version
should remain the simple full-signal criterion.

Recommended future scripts and outputs:

```text
select_sensors_logdet.py
train_selected_sensor_cnn.py
results_logdet/{clean,snr0,snr1,snr5}/
  sensor_ranking.csv
  topk_performance.csv
```

## 9. Phase D -- global gating sensor selection

This is a supervised, task-driven method independent of MI and attribution.
The gate must learn a **fixed physical sensor subset**.

Do not use an input-dependent gate. A selector that first reads all 12 channels
and then chooses channels per sample is invalid for physical sensor placement.

Required global Top-K design:

1. Create one learnable global logit per channel:

   ```text
   alpha: learnable vector [12]
   ```

2. Place the gate at the model input, before the CNN consumes sensor signals.
3. For a requested budget K, use a differentiable hard Top-K mechanism:
   - training: add Gumbel noise to global logits and use a soft relaxation;
   - forward pass: hard K-hot mask;
   - backward pass: straight-through soft gradient;
   - inference: deterministic Top-K of `alpha`, with no Gumbel noise.
4. The same mask is shared by every sample in the dataset.
5. Mask unselected channels before the first convolution, or physically gather
   only selected channels and construct the CNN with `in_channels=K`.
6. Optimize only the 13-class cross-entropy task loss, plus any explicitly
   documented gate regularization. Exact Top-K already enforces the budget, so
   do not add an uncontrolled sparsity penalty unless needed.
7. Use a pre-specified temperature schedule from high/soft to low/hard.
8. Train a separate gate+CNN model from scratch for each SNR condition and K.
9. At the end, save global logits, deterministic selected indices, selection
   order, final model, and test metrics.

Suggested K values:

```text
K = 1, 2, 3, 4, 5, 6, 8, 10
```

Do not pick one K by maximizing test performance. Report the complete
performance-versus-budget curve. Use at least 5 random seeds for gating because
discrete selection can be unstable. Report selection frequency and pairwise
Jaccard stability across seeds.

Recommended future script and outputs:

```text
train_global_gating.py
results_gating/{clean,snr0,snr1,snr5}/K*/seed*/
  final_model.pt
  selected_channels.json
  gate_logits.csv
  test_metrics.json
```

## 10. Final comparison

The three sensor-importance approaches are intentionally different:

1. **Attribution:** what the trained all-sensor CNN relies on.
2. **Unsupervised log-det:** which sensors preserve the most nonredundant signal
   information without labels.
3. **Global gating:** which fixed sensors maximize supervised PipeID
   classification performance.

They should not be merged into one method in the first study.

For each SNR and K, compare:

```text
All 12 sensors (upper/reference baseline)
Random Top-K (repeat across seeds)
Attribution Top-K + retrained CNN
Unsupervised log-det Top-K + retrained CNN
Supervised global-gating Top-K
```

Use the same split, preprocessing, CNN capacity as closely as possible, training
epochs, optimizer, and evaluation metrics. In addition to predictive metrics,
compare:

- selected channel IDs;
- ranking agreement (Spearman correlation where appropriate);
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

## 11. Git and server workflow

The intended future workflow is:

1. Modify and test code locally.
2. Commit and push source code to GitHub.
3. Clone or pull the repository on the server.
4. Keep the four large `.mat` files available locally on the server project
   root, but do not commit them to ordinary Git history.
5. Run experiments on the server and record all results in W&B.

Before creating the repository, add a `.gitignore` that excludes at least:

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

## 12. Implementation order after migration

Follow this order; do not jump directly to gating:

1. Clone/pull the repository and read this entire file.
2. Verify Python environment: PyTorch, NumPy, SciPy, h5py, pandas,
   scikit-learn, and W&B. Captum is additionally needed for attribution.
3. Configure W&B login and verify the `SensorOptimization` project is writable.
4. Verify the four root-level MATLAB filenames, fields, shapes, and sample
   order.
5. Add mandatory W&B logging to any script that does not yet implement it.
6. Run the four all-sensor CNN baselines to completion.
7. Inspect `results_cnn_13class/summary.csv`, W&B runs, and confusion matrices for obvious
   loader or label failures.
8. Implement and run channel attribution on the fixed baseline models.
9. Implement training-only unsupervised log-det rankings and reduced-channel
   CNN evaluation.
10. Implement global exact Top-K gating and multi-seed experiments.
11. Generate one consolidated table and performance-versus-K/SNR figures, and
    upload the final comparison as a W&B artifact/report.

## 13. Current code status

Implemented and smoke-tested locally:

- `cnn_baseline_13class.py`: current 13-class, all-sensor, 80/20 baseline.
- `cnn_baseline_26class.py`: older 26-class script; its robust MATLAB v5/v7.3
  loading functions are currently imported by the 13-class script.
- Clean MATLAB v5 loading, channel normalization, and CNN forward pass have
  been smoke-tested.
- The current baseline saves local result files but still needs mandatory W&B
  integration before the definitive server runs.

Not implemented yet:

- channel attribution script;
- log-det selector for this 12-channel dataset;
- reduced-channel CNN evaluation script;
- global Top-K gating model;
- consolidated comparison/plotting script.

The target server layout has all four prepared files directly in the project
root: `Dataset_4000_Clean.mat`, `Dataset_4000_SNR_0.mat`,
`Dataset_4000_SNR_1.mat`, and `Dataset_4000_SNR_5.mat`.
