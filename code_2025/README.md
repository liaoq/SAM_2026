# NeuroSGD

A feedforward net trained via a custom update rule (`update_grad_v_no_mask2`), where learnable **V matrices** modulate the backward pass instead of using standard backpropagation.


## Architecture

- **Net_V** (`examples/cell_model.py`): Non-quantized version using `Linear_V` layers, each containing a weight matrix **W** and a learnable backward matrix **V**.
- **QuantizedNet_V** (`examples/cell_model.py`): Quantized version using `QuantizedLinear_V` layers with STE-based weight/activation quantization controlled by `--w_bits` and `--a_bits`.
- Both models use a 4-layer MLP: `[input] → hidden → hidden → hidden → [output]`.

## Usage

### Non-quantized version

```bash
python run_netv.py \
    --dataset cifar10 \
    --num_classes 10 \
    --hidden 200 \
    --root ./data \
    --batch_size 256 \
    --epochs 50 \
    --lr 0.5e-2 \
    --seed 0 \
    --device cuda
```


### Standard SGD baseline

```bash
python sgd.py \
    --dataset cifar10 \
    --num_classes 10 \
    --hidden 200 \
    --root ./data \
    --batch_size 256 \
    --epochs 50 \
    --lr 0.5e-2 \
    --device cuda
```


## Experiment Results

All experiments run on CIFAR10 with `Net_V` (non-quantized), `lr=0.005`, `batch_size=256`, `seed=0`, default LR scaling factors (`lr_w=1.0, lr_v=1.0`), on a single NVIDIA H200 GPU.

### CIFAR10, hidden=200

| Epochs | Feedforward Net | Command |
|--------|-----------------|---------|
| 20 | 49.01% | `python run_netv.py --dataset cifar10 --num_classes 10 --hidden 200 --root ./data --batch_size 256 --epochs 20 --lr 0.5e-2 --seed 0 --device cuda` |
| 50 | 50.94% | same with `--epochs 50` |

### CIFAR10, hidden=1000

| Epochs | Feedforward Net | Command |
|--------|-----------------|---------|
| 50 | 52.55% | `python run_netv.py --dataset cifar10 --num_classes 10 --hidden 1000 --root ./data --batch_size 256 --epochs 50 --lr 0.5e-2 --seed 0 --device cuda` |