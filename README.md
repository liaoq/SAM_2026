# PNAS 2026 Code Release

Code for SAM vs backpropagation comparison experiments (`run_netv_compare_with_bp.py`) and plotting (`plot_stat_timeseries.py`).

## Errata and clarifications

Minor corrections and explanations to the manuscript text that could not be updated in the submitted revision PDF. We document them here for reviewers and readers of this repository.

### 1. Figure 4 caption (panel B)

**Submitted text:** “**B:** cosine similarity between SAL and backprop gradients on $W^\ell$ stays near one on all layers”

**Corrected text:** “**B:** cosine similarities between SAL and backprop gradients on $W^\ell$ increase during learning and reach high level”

**Explanation:** This wording was introduced incorrectly during AI-assisted polishing. In the experiments, cosine similarities can rise to a high level (roughly 0.8 on typical layers) but do not stay near 1 on all layers.

## Setup

```bash
pip install -r requirements.txt
```

Requires: `torch`, `torchvision`, `numpy`, `tqdm`. Optional: `medmnist` (for `pathmnist` only), `matplotlib` (for plotting).

## Example: CIFAR-10 SAM vs BP comparison

Run from the repository root (paths below use `./data` for datasets and `./runs` for experiment outputs).

### 3 hidden layers

```bash
python PNAS_2026_release/run_netv_compare_with_bp.py \
  --dataset cifar10 \
  --num_classes 10 \
  --hidden 200 \
  --data_root ./data \
  --exp_root ./runs \
  --run_name cifar10_SAM_compare_with_bp_vv0.8_wb0.8_e50_lr5e-3_layer3_PNAS \
  --batch_size 256 \
  --epochs 50 \
  --seed 0 \
  --device cuda \
  --record_every 5 \
  --log_every 50 \
  --lr_v_prime 0.8 \
  --lr_v 0.8 \
  --lr_w_prime 0.8 \
  --symmetric_vv \
  --weight_decay 0.001 \
  --momentum 0.9 \
  --weight_decay_v 10e-3 \
  --weight_decay_v_bar 10e-3 \
  --lr 5e-3 \
  --forward_norm_type none \
  --backward_norm_type none \
  --sal_update combined \
  --trace_v_updates \
  --optimizer sgd \
  --forward_nonlin relu \
  --num_hidden_layers 3
```

### 4 hidden layers

```bash
python PNAS_2026_release/run_netv_compare_with_bp.py \
  --dataset cifar10 \
  --num_classes 10 \
  --hidden 200 \
  --data_root ./data \
  --exp_root ./runs \
  --run_name cifar10_SAM_compare_with_bp_vv0.8_wb0.8_e50_lr5e-3_layer4_PNAS \
  --batch_size 256 \
  --epochs 50 \
  --seed 0 \
  --device cuda \
  --record_every 5 \
  --log_every 50 \
  --lr_v_prime 0.8 \
  --lr_v 0.8 \
  --lr_w_prime 0.8 \
  --weight_decay 0.001 \
  --momentum 0.9 \
  --weight_decay_v 10e-3 \
  --weight_decay_v_bar 10e-3 \
  --lr 5e-3 \
  --forward_norm_type none \
  --backward_norm_type none \
  --sal_update combined \
  --trace_v_updates \
  --optimizer sgd \
  --forward_nonlin relu \
  --num_hidden_layers 4
```

Outputs are written under `./runs/<run_name>/` (including `stat_timeseries.json` when `--record_every` is set).

## Plot results

After a run finishes:

```bash
python PNAS_2026_release/plot_stat_timeseries.py \
  --exp_root ./runs \
  --run_name cifar10_SAM_compare_with_bp_vv0.8_wb0.8_e50_lr5e-3_layer3_PNAS
```

For the 4-layer run, use `--run_name cifar10_SAM_compare_with_bp_vv0.8_wb0.8_e50_lr5e-3_layer4_PNAS`.

## Reproducibility

We do not guarantee bitwise equivalence between runs on different hardware or software stacks. Results should be qualitatively similar when reproducing the commands above, but exact curves and divergence timing may differ.

The **4-layer** result reported in the paper was obtained on an **NVIDIA L40S**. If you run the same command on an **A100**, training may diverge a few epochs earlier than on L40S. The reason is unclear.

## Additional code

The `code_2025/` subdirectory contains the earlier NeuroSGD (our internal project name for SAL/SAM) training codebase. See `code_2025/README.md` for usage.
