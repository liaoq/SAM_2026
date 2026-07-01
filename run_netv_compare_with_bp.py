"""
Standalone flat Net_V SAL vs BP comparison (exact behavior of NeuroSGD3/run_netv_compare_with_bp.py).

No imports from NeuroSGD3. Edit REPRO below for defaults when a flag is omitted.
CLI flags match NeuroSGD3/config.py.

  python run_netv_compare_with_bp.py --dataset mnist --hidden 200 ...

Requires: torch, torchvision, numpy, tqdm. Optional: medmnist (pathmnist only).
"""
import json
import math
import os
import random
import shlex
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Reproducible defaults (match NeuroSGD3/config.py) -----------------------
REPRO = SimpleNamespace(
    dataset="mnist",
    num_classes=10,
    hidden=200,
    backward_hidden=None,
    num_hidden_layers=3,
    activation="sigmoid",
    init_as_gd=False,
    input_fold="1x1",
    v_mode="3x3_conv",
    trace_v_updates=False,
    wf_vf_wb_vb_conv_group=(1, 1, 1, 1),
    batch_size=256,
    epochs=10,
    lr=0.5e-2,
    optimizer="sgd",
    beta1=0.9,
    beta2=0.95,
    weight_decay=0.001,
    weight_decay_v=None,
    weight_decay_v_bar=None,
    momentum=0.9,
    lr_w=1.0,
    lr_w_prime=0.2,
    lr_v=1.0,
    lr_v_prime=4.0,
    seed=0,
    more_determinism=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compare_activations=False,
    grid_search=False,
    w_bits=32,
    a_bits=32,
    log_every=0,
    record_every=0,
    force_backprop=False,
    symmetric_vv=True,
    forward_norm_type="none",
    forward_nonlin="relu",
    backward_norm_type="none",
    backward_nonlin="identity",
    backward_tanh_scale=1.0,
    residual_type="none",
    residual_mode="none",
    residual_ratio=0.5,
    backproj_mode="standard",
    tv_mask_mode="none",
    sal_update="reference",
    run_name="default",
    data_root=None,
    exp_root=None,
)
# -----------------------------------------------------------------------------


def parse_run_args(argv=None):
    """Parse CLI (same flags as NeuroSGD3/config.py); defaults from REPRO."""
    import argparse

    d = vars(REPRO)
    default_data_root = d.get("data_root") if d.get("data_root") is not None else "data/"
    default_exp_root = d.get("exp_root") if d.get("exp_root") is not None else "runs"

    parser = argparse.ArgumentParser(description="Multi-dataset Training")
    parser.add_argument(
        "--dataset",
        type=str,
        default=d["dataset"],
        choices=[
            "mnist",
            "cifar10",
            "pathmnist",
            "fashion_mnist",
            "svhn",
            "stl10",
            "caltech101",
            "oxford_iiit_pet",
            "flowers102",
            "eurosat",
            "symbolic_regression",
        ],
        help="dataset to use",
    )
    parser.add_argument("--data_root", type=str, default=default_data_root,
                        help="root directory for loading/downloading datasets")
    parser.add_argument("--exp_root", type=str, default=default_exp_root,
                        help="root directory for experiment outputs (logs, figures)")
    parser.add_argument("--run_name", type=str, default=d["run_name"],
                        help="subdirectory under exp_root for this run")
    parser.add_argument("--num_classes", type=int, default=d["num_classes"],
                        help="number of classes in the dataset")
    parser.add_argument("--hidden", type=int, default=d["hidden"],
                        help="number of hidden units in the model")
    parser.add_argument("--backward_hidden", type=int, default=d["backward_hidden"],
                        help="hidden units for backward path Net2 (default: same as --hidden)")
    parser.add_argument("--num_hidden_layers", type=int, default=d["num_hidden_layers"],
                        help="number of hidden layers (default: 3)")
    parser.add_argument("--activation", type=str, default=d["activation"],
                        help="activation function to use")
    parser.add_argument("--init_as_gd", action="store_true", default=d["init_as_gd"],
                        help="Initialize v as identity matrix")
    parser.add_argument("--input_fold", type=str, default=d["input_fold"],
                        choices=["1x1", "2x2", "4x4"],
                        help="4D scripts only: fold kxk input patches into channels")
    parser.add_argument("--v_mode", type=str, default=d["v_mode"],
                        choices=["3x3_conv", "1x1_conv", "5x5_conv"],
                        help="4D SAL bridge V / V_bar")
    parser.add_argument("--trace_v_updates", action="store_true", default=d["trace_v_updates"],
                        help="TracingSGD logs |grad|, |wd*p|, |grad+wd*p| on V/V_bar")
    parser.add_argument(
        "--wf_vf_wb_vb_conv_group",
        type=int,
        nargs=4,
        metavar=("WF", "VF", "WB", "VB"),
        default=list(d["wf_vf_wb_vb_conv_group"]),
        help="conv groups for W_f V_f and W_b V_b",
    )
    parser.add_argument("--batch_size", type=int, default=d["batch_size"],
                        help="input batch size for training")
    parser.add_argument("--epochs", type=int, default=d["epochs"],
                        help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=d["lr"], help="learning rate")
    parser.add_argument("--optimizer", type=str, default=d["optimizer"],
                        choices=["sgd", "adamw"], help="optimizer for Net1 and Net2")
    parser.add_argument("--beta1", type=float, default=d["beta1"], help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=d["beta2"], help="AdamW beta2")
    parser.add_argument("--weight_decay", type=float, default=d["weight_decay"],
                        help="weight decay on W / W'")
    parser.add_argument("--weight_decay_v", type=float, default=d["weight_decay_v"],
                        help="weight decay on Net1 V")
    parser.add_argument("--weight_decay_v_bar", type=float, default=d["weight_decay_v_bar"],
                        help="weight decay on Net2 V_bar")
    parser.add_argument("--momentum", type=float, default=d["momentum"],
                        help="SGD momentum")
    parser.add_argument("--lr_w", type=float, default=d["lr_w"],
                        help="learning rate scaling factor for w")
    parser.add_argument("--lr_w_prime", type=float, default=d["lr_w_prime"],
                        help="learning rate scaling factor for w prime")
    parser.add_argument("--lr_v", type=float, default=d["lr_v"],
                        help="learning rate scaling factor for v")
    parser.add_argument("--lr_v_prime", type=float, default=d["lr_v_prime"],
                        help="learning rate scaling factor for v prime")
    parser.add_argument("--seed", type=int, default=d["seed"], help="random seed")
    parser.add_argument(
        "--more_determinism",
        action="store_true",
        default=d["more_determinism"],
        help="enable stricter CUDA/cuDNN determinism (slower; may still differ across GPU types)",
    )
    parser.add_argument("--device", type=str, default=d["device"],
                        help="device to run the model on")
    parser.add_argument("--compare_activations", action="store_true",
                        default=d["compare_activations"],
                        help="Compare different activation functions")
    parser.add_argument("--grid_search", action="store_true", default=d["grid_search"],
                        help="perform grid search for learning rates")
    parser.add_argument("--w_bits", type=int, default=d["w_bits"],
                        help="number of bits for weight quantization")
    parser.add_argument("--a_bits", type=int, default=d["a_bits"],
                        help="number of bits for activation quantization")
    parser.add_argument("--log_every", type=int, default=d["log_every"],
                        help="print intra-epoch stats every N train batches (0=disable)")
    parser.add_argument("--record_every", type=int, default=d["record_every"],
                        help="record stats every R train iterations (0=disable)")
    parser.add_argument("--force_backprop", action="store_true", default=d["force_backprop"],
                        help="replace Net1 gradients with synced Net1_bp gradients")
    parser.add_argument(
        "--symmetric_vv",
        action=argparse.BooleanOptionalAction,
        default=d["symmetric_vv"],
        help="align V and V_bar grads at each junction",
    )
    parser.add_argument(
        "--forward_norm_type",
        type=str,
        default=d["forward_norm_type"],
        choices=["none", "batchnorm", "layernorm"],
        help="non-parametric norm before each forward nonlinearity",
    )
    parser.add_argument(
        "--forward_nonlin",
        type=str,
        default=d["forward_nonlin"],
        choices=["identity", "relu", "gelu", "silu", "tanh", "sigmoid"],
        help="nonlinearity after forward-direction normalization",
    )
    parser.add_argument(
        "--backward_norm_type",
        type=str,
        default=d["backward_norm_type"],
        choices=["none", "batchnorm", "layernorm"],
        help="non-parametric norm before each backward nonlinearity",
    )
    parser.add_argument(
        "--backward_nonlin",
        type=str,
        default=d["backward_nonlin"],
        choices=["identity", "relu", "gelu", "silu", "tanh", "sigmoid"],
        help="nonlinearity after backward-direction normalization",
    )
    parser.add_argument("--backward_tanh_scale", type=float, default=d["backward_tanh_scale"],
                        help="scale before tanh on backward path")
    parser.add_argument(
        "--residual_type",
        type=str,
        default=d["residual_type"],
        choices=["none", "resnet", "transformer"],
        help="how to add residuals in norm+nl blocks",
    )
    parser.add_argument(
        "--residual_mode",
        type=str,
        default=d["residual_mode"],
        choices=[
            "none",
            "every_layer",
            "every_2",
            "every_2_nonconsec",
            "random_symmetric",
            "random_symmetric_offdiag",
        ],
        help="residual connectivity F (lower triangle)",
    )
    parser.add_argument("--residual_ratio", type=float, default=d["residual_ratio"],
                        help="Bernoulli probability for random_symmetric residual entries")
    parser.add_argument(
        "--backproj_mode",
        type=str,
        default=d["backproj_mode"],
        choices=["standard", "roundtrip", "roundtrip_learn"],
        help="Net2 backproj mode",
    )
    parser.add_argument(
        "--tv_mask_mode",
        type=str,
        default=d["tv_mask_mode"],
        choices=["none", "relu"],
        help="4D nnmodule: gate backward sha by mask from tv_out",
    )
    parser.add_argument(
        "--sal_update",
        type=str,
        default=d["sal_update"],
        choices=["reference", "combined", "nnmodule"],
        help="SAL layer update mode",
    )

    args = parser.parse_args(argv)
    if args.backward_hidden is None:
        args.backward_hidden = args.hidden
    if args.weight_decay_v is None:
        args.weight_decay_v = args.weight_decay
    if args.weight_decay_v_bar is None:
        args.weight_decay_v_bar = args.weight_decay_v
    args.run_dir = os.path.normpath(
        os.path.join(os.path.expanduser(args.exp_root), args.run_name)
    )
    return args


# === utils/utils.py ===
import torch
import numpy as np
import random

def set_seed(seed, more_determinism=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if more_determinism:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    
def converter(indices, num_classes):
    '''
    Convert indices to probability vectors
    '''
    probvec = torch.zeros(len(indices), num_classes)
    probvec[range(len(indices)), indices] = 1.0
    return probvec

# === utils/activations.py ===
import torch
import torch.nn.functional as F

def F_relu(x):
    return F.relu(x)

def F_identity(x):
    return x

def F_sigmoid(x):
    return torch.sigmoid(x)

def F_tanh(x):
    return torch.tanh(x)

def F_leaky_relu(x, negative_slope=0.01):
    return F.leaky_relu(x, negative_slope)

def F_elu(x, alpha=1.0):
    return F.elu(x, alpha)

def F_selu(x):
    return F.selu(x)

def F_softplus(x):
    return F.softplus(x)

def F_swish(x):
    return x * torch.sigmoid(x)

def F_mish(x):
    return x * torch.tanh(F.softplus(x))

def F_gelu(x):
    return F.gelu(x)

def F_silu(x):
    return F.silu(x)


def resolve_nonlin(name: str):
    """Map CLI nonlinearity name to activation callable."""
    table = {
        "identity": F_identity,
        "relu": F_relu,
        "gelu": F_gelu,
        "silu": F_silu,
        "tanh": F_tanh,
        "sigmoid": F_sigmoid,
    }
    key = (name or "relu").lower()
    if key not in table:
        raise ValueError(f"Unknown nonlinearity {name!r}; expected one of {tuple(table)}")
    return table[key]


def swish(x):
    return x * torch.sigmoid(x)

def squared(x):
    return x**2

def power_1_5(x):
    return x**1.5

def power_0_5(x):
    return x**0.5

def F_sign(x):
    return torch.sign(x)


activation_functions = {
    'relu': F.relu,
    'gelu': F.gelu,
    'swish': swish,
    'sin': torch.sin,
    'cos': torch.cos,
    'linear': lambda x: x,
    'squared': squared,
    'power_1_5': power_1_5, 
    'power_0_5': power_0_5,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'leaky_relu': F.leaky_relu,
    'elu': F.elu,
    'selu': F.selu,
    'softplus': F.softplus,
    'mish': lambda x: x * torch.tanh(F.softplus(x)),
    'hardtanh': F.hardtanh,
    'softsign': F.softsign,
    'celu': F.celu,
    'log_sigmoid': F.logsigmoid,
}


# === examples/cell_model.py ===
import torch

import torch

def quantize_ste(x, bits):

    if bits is None or bits >= 32:
        return x
    qmin = -2 ** (bits - 1)
    qmax = 2 ** (bits - 1) - 1

    max_val = x.abs().max()
    if max_val == 0:
        scale = 1.0
    else:
        scale = max_val / qmax

    x_div = x / scale
    x_q = torch.round(x_div)
    x_q = torch.clamp(x_q, qmin, qmax)
    x_dequant = x_q * scale

    return x + (x_dequant - x).detach()


NORM_TYPES = ("none", "batchnorm", "layernorm")
_NORM_EPS = 1e-5


def make_norm_fn(norm_type: str):
    """
    Non-parameterized normalization (no learnable scale/shift).
    Applied on pre-activations immediately before the nonlinearity.

    - layernorm: normalize each sample over the feature dimension (last dim)
    - batchnorm: normalize each feature over the batch dimension (dim 0)
    - none: identity
    """
    if norm_type is None or norm_type == "none":
        return lambda x: x
    if norm_type == "layernorm":

        def layernorm(x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, unbiased=False, keepdim=True)
            return (x - mean) / torch.sqrt(var + _NORM_EPS)

        return layernorm
    if norm_type == "batchnorm":

        def batchnorm(x):
            if x.dim() < 2:
                return x
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, unbiased=False, keepdim=True)
            return (x - mean) / torch.sqrt(var + _NORM_EPS)

        return batchnorm
    raise ValueError(f"Unknown norm_type {norm_type!r}; expected one of {NORM_TYPES}")


def forward_norm_nl(x, norm_type="none", nonlin=F_relu):
    """Forward-direction block (Net1): norm then nonlinearity."""
    return nonlin(make_norm_fn(norm_type)(x))


def forward_norm_nl_compute_grad(grad_output, x_pre, norm_type="none", nonlin=F_relu):
    """
    VJP of forward_norm_nl w.r.t. x_pre (autograd replay on detached x_pre).
    Replaces (gthb * (thb > 0)) in SAL when nonlin=relu and norm_type=none.
    """
    x = x_pre.detach().requires_grad_(True)
    y = forward_norm_nl(x, norm_type, nonlin)
    grad_x, = torch.autograd.grad(
        y, x, grad_outputs=grad_output, retain_graph=False, allow_unused=False
    )
    return grad_x


def backward_norm_nl(x, norm_type="none", nonlin=F_identity):
    """Backward-direction block (Net2): norm then nonlinearity."""
    return nonlin(make_norm_fn(norm_type)(x))


def backward_norm_nl_compute_grad(grad_output, x_pre, norm_type="none", nonlin=F_identity):
    """
    VJP of backward_norm_nl w.r.t. x_pre.
    Not used in SAL yet (no mask multiply on backward-direction pathway).
    """
    x = x_pre.detach().requires_grad_(True)
    y = backward_norm_nl(x, norm_type, nonlin)
    grad_x, = torch.autograd.grad(
        y, x, grad_outputs=grad_output, retain_graph=False, allow_unused=False
    )
    return grad_x


class QuantizedLinear_V(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False, backward=False, backward_more_dim=20,
                 last_layer=False, first_layer=False, init_as_gd=False,
                 w_bits=32, a_bits=32):
        super(QuantizedLinear_V, self).__init__()

        if init_as_gd:
            torch.manual_seed(42)

        self.ha = None
        self.hb = None

        self.w = nn.Linear(input_dim, output_dim, bias=bias)
        self.v = None
        if backward:
            if not last_layer:
                self.v = nn.Linear(output_dim - backward_more_dim, output_dim, bias=bias)
                if init_as_gd:
                    nn.init.eye_(self.v.weight)
        else:
            self.v = nn.Linear(input_dim, input_dim + backward_more_dim, bias=bias)
            if init_as_gd:
                nn.init.eye_(self.v.weight[:input_dim, :input_dim])
                
        self.last_layer = last_layer
        self.first_layer = first_layer

        self.w_bits = w_bits
        self.a_bits = a_bits

    def forward(self, x):
        self.ha = x
        if x.requires_grad:
            self.ha.retain_grad()

        w_q = quantize_ste(self.w.weight, self.w_bits)
        x = F.linear(x, w_q, self.w.bias)
        x = quantize_ste(x, self.a_bits)

        self.hb = x.detach().requires_grad_(True)
        self.hb.retain_grad()
        return x

    def backproj_nobias(self, y):
        self.hb.grad = y
        w_q = quantize_ste(self.w.weight, self.w_bits)
        x = torch.mm(y, w_q)
        self.ha.grad = x
        self.sha = y.mm(w_q)
        return x

def nonlins_from_args(args):
    return (
        resolve_nonlin(getattr(args, "forward_nonlin", "relu")),
        resolve_nonlin(getattr(args, "backward_nonlin", "identity")),
    )


class QuantizedNet_V(nn.Module):
    def __init__(self, in_d=5, out_d=5, F=F.relu, hidden=100, backward_more_dim=0,
                 dropout_rate=[0, 0], backward=False, init_as_gd=False,
                 w_bits=32, a_bits=32,
                 forward_norm_type="none", backward_norm_type="none",
                 forward_nonlin_fn=None, backward_nonlin_fn=None):

        super(QuantizedNet_V, self).__init__()
        self.forward_norm_type = forward_norm_type
        self.backward_norm_type = backward_norm_type
        self.forward_nonlin_fn = forward_nonlin_fn if forward_nonlin_fn is not None else F
        self.backward_nonlin_fn = backward_nonlin_fn if backward_nonlin_fn is not None else F_identity

        if backward:
            hidden += backward_more_dim

        self.layers = nn.ModuleList()
        layer_sizes = [in_d] + [hidden] * 3 + [out_d]
        for i in range(len(layer_sizes) - 1):
            is_first = i == 0
            is_last = i == len(layer_sizes) - 2
            self.layers.append(
                QuantizedLinear_V(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    bias=False,
                    backward=backward,
                    backward_more_dim=backward_more_dim,
                    first_layer=is_first,
                    last_layer=is_last,
                    init_as_gd=init_as_gd,
                    w_bits=w_bits,
                    a_bits=a_bits
                )
            )

        self.F = F
        self.dropout_forward = nn.Dropout(p=dropout_rate[0])
        self.dropout_back = nn.Dropout(p=dropout_rate[1])

    def backproj(self, x, nonlin=None):
        """
        自定义反向传播，与原来的逻辑一致
        """
        for i in reversed(range(len(self.layers))):
            x = self.layers[i].backproj_nobias(x)
            if i > 0:
                x = backward_norm_nl(
                    x, self.backward_norm_type, self.backward_nonlin_fn
                )
                if nonlin:
                    x = nonlin(x)
                x = self.dropout_back(x)
        return x

    def forward(self, x, nonlin=None):
        used_nonlin = nonlin if nonlin else self.F
        x = x.view(x.shape[0], -1)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if nonlin is None:
                x = forward_norm_nl(
                    x, self.forward_norm_type, self.forward_nonlin_fn
                )
            else:
                x = used_nonlin(x)
            x = self.dropout_forward(x)
        x = self.layers[-1](x)
        return x

def update_grad_v_no_mask2(Net1, Net2, X, Y, loss_fn, args):
    '''
    O: gradient of the loss function with respect to the output of Net1, shape: torch.Size([256, 10])
    '''

    output = Net1(X)
    output.retain_grad()
    loss = loss_fn(output, Y)
    loss.backward()

    O = output.grad.detach()
    O2 = Net2(X)
    loss2 = loss_fn(O2, Y)
    Net2.backproj(O, nonlin=None)

    for tt, ss in zip(Net1.modules(), Net2.modules()):
        if isinstance(tt, Linear_V):
            update_layer_gradients(tt, ss, args)

    return loss.item(), loss2.item()

def update_layer_gradients(t, s, args):
    '''
    Original SAL W/V updates (reference).
    t - layer in Net1 (Feedforward Net)
    s - layer in Net2 (Gradient Net)
    '''
    # Non-parametric norms (same as Net_V.forward / Net_V.backproj); wire into SAL below manually.
    forward_norm_type = getattr(args, "forward_norm_type", "none")
    backward_norm_type = getattr(args, "backward_norm_type", "none")
    forward_nl, backward_nl = nonlins_from_args(args)

    tha = t.ha
    shb = s.hb.grad
    # sha = shb.mm(s.w.weight)
    thb = tha.mm(t.w.weight.T)
    sha_transpose = s.w.weight.T.mm(shb.T)

    #if getattr(args, "forward_norm_type", "none") != "none":
    #    thb = forward_norm_fn(thb)
    #if getattr(args, "backward_norm_type", "none") != "none":
    #    sha_transpose = backward_norm_fn(sha_transpose)

    shbtha = shb.T.mm(tha)
    symmetric_vv = getattr(args, "symmetric_vv", False)

    if s.last_layer:
        twgrad = s.hb.grad.T.mm(tha)
    else:
        # s.v.weight.shape: torch.Size([200, 200])
        gthb = shb.mm(s.v.weight) # Apply v here
        twgrad = forward_norm_nl_compute_grad(
            gthb, thb, forward_norm_type, forward_nl
        ).T.mm(tha)

    t.w.weight.grad = twgrad * args.lr_w

    if not s.last_layer:
        if symmetric_vv:
            tha_next_layer = forward_norm_nl(thb, forward_norm_type, forward_nl)
            svgrad = shb.T.mm(tha_next_layer) * args.lr_v_prime
        else:
            svgrad = shbtha.mm(t.w.weight.T) * args.lr_v_prime
        s.v.weight.grad = svgrad

    if not t.first_layer:
        swgrad = shbtha.mm(t.v.weight.T) * args.lr_w_prime
        if symmetric_vv:
            shb_previous_layer = backward_norm_nl(
                sha_transpose, backward_norm_type, backward_nl
            )
            tvgrad = shb_previous_layer.mm(tha) * args.lr_v
        else:
            tvgrad = s.w.weight.T.mm(shbtha) * args.lr_v
        s.w.weight.grad = swgrad
        t.v.weight.grad = tvgrad


def update_layer_gradients_combined_norm_nl(t, s, args):
    '''
    Same as update_layer_gradients, but forward-direction norm+ReLU is one block:
      - W grad: forward_norm_nl_compute_grad replaces (gthb * (thb > 0))
      - symmetric V_bar: forward_norm_nl(thb) replaces forward_nl(thb)
      - symmetric V: backward_norm_nl(sha_transpose) replaces backward_nl(sha_transpose)
    backward_norm_nl_compute_grad is not used yet (no mask on backward-direction path).
    '''
    forward_norm_type = getattr(args, "forward_norm_type", "none")
    backward_norm_type = getattr(args, "backward_norm_type", "none")
    forward_nl, backward_nl = nonlins_from_args(args)

    tha = t.ha
    shb = s.hb.grad
    thb = tha.mm(t.w.weight.T)
    sha_transpose = s.w.weight.T.mm(shb.T)
    shbtha = shb.T.mm(tha)
    symmetric_vv = getattr(args, "symmetric_vv", False)

    if s.last_layer:
        twgrad = s.hb.grad.T.mm(tha)
    else:
        gthb = shb.mm(s.v.weight)
        twgrad = forward_norm_nl_compute_grad(
            gthb, thb, forward_norm_type, forward_nl
        ).T.mm(tha)

    t.w.weight.grad = twgrad * args.lr_w

    if not s.last_layer:
        if symmetric_vv:
            tha_next_layer = forward_norm_nl(thb, forward_norm_type, forward_nl)
            svgrad = shb.T.mm(tha_next_layer) * args.lr_v_prime
        else:
            svgrad = shbtha.mm(t.w.weight.T) * args.lr_v_prime
        s.v.weight.grad = svgrad

    if not t.first_layer:
        swgrad = shbtha.mm(t.v.weight.T) * args.lr_w_prime
        if symmetric_vv:
            shb_previous_layer = backward_norm_nl(
                sha_transpose, backward_norm_type, backward_nl
            )
            tvgrad = shb_previous_layer.mm(tha) * args.lr_v
        else:
            tvgrad = s.w.weight.T.mm(shbtha) * args.lr_v
        s.w.weight.grad = swgrad
        t.v.weight.grad = tvgrad


class _RightLinearView(nn.Module):
    """
    Directional module view for maps written as x @ linear.weight in the SAL equations.

    nn.Linear.forward uses x @ weight.T; the backward-direction W' and V_bar maps in this
    code use x @ weight. For a future arbitrary nn.Module, replace this view with that
    module's own forward.
    """
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return x.mm(self.linear.weight)


def _clear_module_grads(module):
    for p in module.parameters():
        p.grad = None


def update_layer_gradients_combined_nnmodule(t, s, args):
    """
    Symmetric-VV SAL update using only local module forwards plus .backward(upstream).

    This is the nnmodule-shaped version of update_layer_gradients_combined_norm_nl:
      - Net1 W uses t.w.forward, then forward_norm_nl, then .backward(gthb)
      - Net2 V_bar uses its directional forward shb -> shb @ V_bar, then .backward(tha_next)
      - Net2 W' uses sha = sw_fwd(shb), then sha.backward(tv_fwd(tha))
      - Net1 V uses tv_fwd(tha), then .backward(backward_norm_nl(sha).T)

    It intentionally requires --symmetric_vv so the non-symmetric shbtha branches are not used.
    """
    if not getattr(args, "symmetric_vv", False):
        raise ValueError("--sal_update nnmodule requires --symmetric_vv")

    forward_norm_type = getattr(args, "forward_norm_type", "none")
    backward_norm_type = getattr(args, "backward_norm_type", "none")
    forward_nl, backward_nl = nonlins_from_args(args)

    tha = t.ha.detach()
    shb = s.hb.grad.detach()

    if s.last_layer:
        _clear_module_grads(t.w)
        tha_leaf = tha.detach().requires_grad_(True)
        t.w(tha_leaf).backward(shb)
        twgrad = t.w.weight.grad.detach().clone()
        thb_value = None
    else:
        vbar_fwd = _RightLinearView(s.v)
        _clear_module_grads(vbar_fwd)
        shb_vbar_leaf = shb.detach().requires_grad_(True)
        gthb_out = vbar_fwd(shb_vbar_leaf)
        gthb = gthb_out.detach()

        _clear_module_grads(t.w)
        tha_leaf = tha.detach().requires_grad_(True)
        thb = t.w(tha_leaf)
        thb_value = thb.detach()
        forward_norm_nl(thb, forward_norm_type, forward_nl).backward(gthb)
        twgrad = t.w.weight.grad.detach().clone()

    t.w.weight.grad = twgrad * args.lr_w

    if not s.last_layer:
        with torch.no_grad():
            tha_next_layer = forward_norm_nl(thb_value, forward_norm_type, forward_nl)

        gthb_out.backward(tha_next_layer)
        s.v.weight.grad = s.v.weight.grad.detach().clone() * args.lr_v_prime

    if not t.first_layer:
        sw_fwd = _RightLinearView(s.w)
        _clear_module_grads(sw_fwd)
        _clear_module_grads(t.v)

        tha_v_leaf = tha.detach().requires_grad_(True)
        tv_out = t.v(tha_v_leaf)
        tv_upstream = tv_out.detach()

        shb_leaf = shb.detach().requires_grad_(True)
        sha = sw_fwd(shb_leaf)
        sha_value = sha.detach().T
        sha.backward(tv_upstream)
        s.w.weight.grad = s.w.weight.grad.detach().clone() * args.lr_w_prime

        with torch.no_grad():
            shb_previous_layer = backward_norm_nl(
                sha_value, backward_norm_type, backward_nl
            )

        tv_out.backward(shb_previous_layer.T)
        t.v.weight.grad = t.v.weight.grad.detach().clone() * args.lr_v


def get_sal_update_fn(args):
    """Select SAL trainer from args.sal_update ('reference' | 'combined' | 'nnmodule')."""
    sal_update = getattr(args, "sal_update", "reference")
    if sal_update == "nnmodule":
        return update_grad_v_no_mask2_nnmodule
    if sal_update == "combined":
        return update_grad_v_no_mask2_combined
    return update_grad_v_no_mask2


def update_grad_v_no_mask2_nnmodule(Net1, Net2, X, Y, loss_fn, args):
    """Same as update_grad_v_no_mask2 but uses update_layer_gradients_combined_nnmodule."""
    output = Net1(X)
    output.retain_grad()
    loss = loss_fn(output, Y)
    loss.backward()

    O = output.grad.detach()
    O2 = Net2(X)
    loss2 = loss_fn(O2, Y)
    Net2.backproj(O, nonlin=None)

    for tt, ss in zip(Net1.modules(), Net2.modules()):
        if isinstance(tt, Linear_V):
            update_layer_gradients_combined_nnmodule(tt, ss, args)

    return loss.item(), loss2.item()


def update_grad_v_no_mask2_combined(Net1, Net2, X, Y, loss_fn, args):
    """Same as update_grad_v_no_mask2 but uses update_layer_gradients_combined_norm_nl."""
    output = Net1(X)
    output.retain_grad()
    loss = loss_fn(output, Y)
    loss.backward()

    O = output.grad.detach()
    O2 = Net2(X)
    loss2 = loss_fn(O2, Y)
    Net2.backproj(O, nonlin=None)

    for tt, ss in zip(Net1.modules(), Net2.modules()):
        if isinstance(tt, Linear_V):
            update_layer_gradients_combined_norm_nl(tt, ss, args)

    return loss.item(), loss2.item()


class Linear_V(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False, backward=False, backward_more_dim=20,
                 last_layer=False, first_layer=False, init_as_gd=False):
        super(Linear_V, self).__init__()

        ## for initialization experiment
        if init_as_gd:
            torch.manual_seed(42)

        self.ha = None
        self.hb = None
        
        self.w = nn.Linear(input_dim, output_dim, bias=False)
        self.v = None
        if backward:
            if not last_layer:
                self.v = nn.Linear(output_dim-backward_more_dim, output_dim, bias=False)
                ## for initialization experiment
                if init_as_gd:
                    nn.init.eye_(self.v.weight)

        else:
            self.v = nn.Linear(input_dim, input_dim+backward_more_dim, bias=False)
            ## for initialization experiment
            if init_as_gd:
                nn.init.eye_(self.v.weight[:input_dim, :input_dim]) 
        
        self.last_layer = last_layer
        self.first_layer = first_layer
    
    def forward(self, x):
        '''
        x: input to the layer
        self.ha: totally the same as x, including the gradient.
        self.hb: the value of w(x), but hb itself has no gradient now. The gradient will be calculated manually in backproj_nobias.
        '''
        self.ha = x
        if x.requires_grad:
            self.ha.retain_grad()

        x = self.w(x)
        self.hb = x.detach().requires_grad_(True)
        self.hb.retain_grad()
        return x

    def backproj_nobias(self, y):
        '''
        Input:
            y - equals O, which is the gradient of the loss function with respect to the output; y.shape: torch.Size([256, 10])
        
        Parameters:
            self.w - Linear(in_features=200, out_features=10, bias=False)
            self.w.weight.shape - torch.Size([10, 200])
        
        Output: 
            x - dL/dx = dL/dy * dy/dx = W^T * dL/dy
            x.shape - torch.Size([256, 200])
            
        Loop1 | Layer 3:
            y.shape: torch.Size([256, 10])
            self.w.weight.shape: torch.Size([10, 200])
            x.shape: torch.Size([256, 200])
        
        Loop2 | Layer 2:
            y.shape: torch.Size([256, 200])
            self.w.weight.shape: torch.Size([200, 200])
            x.shape: torch.Size([256, 200])
            
        Loop3 | Layer 1:
            y.shape: torch.Size([256, 200])
            self.w.weight.shape: torch.Size([200, 200])
            x.shape: torch.Size([256, 200])
            
        Loop4 | Layer 0:
            y.shape: torch.Size([256, 200])
            self.w.weight.shape: torch.Size([200, 3072])
            x.shape: torch.Size([256, 3072])
        '''
        # self.hb.grad = y # y: dL/dy
        # x = torch.mm(y, self.w.weight) # x: dL/dx = W^T * dL/dy
        # self.ha.grad = x

        self.hb.grad = y  # y: dL/dy
        x = torch.mm(y, self.w.weight)  # x: dL/dx = W^T * dL/dy
        self.ha.grad = x
        
        self.sha = y.mm(self.w.weight)

        # 使用 v 来 gate sha
        # if self.v is not None:
            # import pdb; pdb.set_trace()
            # print("self.sha.shape:", self.sha.shape) #[256, 200]
            # print("self.v.weight.shape:", self.v.weight.shape) #[200, 200]
            # gate = torch.mm(self.sha, self.v.weight.T)
            # x = x * (gate > 0)
        
        return x


def build_net_v_sgd_param_groups(net, weight_decay_w, weight_decay_v):
    """SGD param groups: W with weight_decay_w, V with weight_decay_v."""
    w_params, v_params = [], []
    for layer in net.layers:
        w_params.append(layer.w.weight)
        if layer.v is not None:
            v_params.append(layer.v.weight)
    groups = [{"params": w_params, "weight_decay": weight_decay_w}]
    if v_params:
        groups.append({"params": v_params, "weight_decay": weight_decay_v})
    return groups


def args_to_jsonable(args):
    """vars(args) safe for json.dump (drops callables)."""
    out = {}
    for key, val in vars(args).items():
        if callable(val):
            continue
        out[key] = val
    return out

class Net_V(nn.Module):
    def __init__(self, in_d=5, out_d=5, F=F.relu, hidden=100, num_hidden_layers=3,
                 backward_more_dim=0, dropout_rate=[0,0], backward=False, init_as_gd=False,
                 forward_norm_type="none", backward_norm_type="none",
                 forward_nonlin_fn=None, backward_nonlin_fn=None):
        super(Net_V, self).__init__()

        self.forward_norm_type = forward_norm_type
        self.backward_norm_type = backward_norm_type
        self.forward_nonlin_fn = forward_nonlin_fn if forward_nonlin_fn is not None else F
        self.backward_nonlin_fn = backward_nonlin_fn if backward_nonlin_fn is not None else F_identity
        if num_hidden_layers < 0:
            raise ValueError(f"num_hidden_layers must be >= 0, got {num_hidden_layers}")

        if backward:
            hidden += backward_more_dim
        
        self.layers = nn.ModuleList()
        
        layer_sizes = [in_d] + [hidden] * num_hidden_layers + [out_d]
            
        for i in range(len(layer_sizes) - 1):
            is_first = i == 0
            is_last = i == len(layer_sizes) - 2
            self.layers.append(Linear_V(layer_sizes[i], layer_sizes[i+1], bias=False, 
                                      backward=backward, backward_more_dim=backward_more_dim,
                                      first_layer=is_first, last_layer=is_last, init_as_gd=init_as_gd))

        self.F = F
        self.dropout_forward = nn.Dropout(p=dropout_rate[0])
        self.dropout_back = nn.Dropout(p=dropout_rate[1])

    def backproj(self, x, nonlin=None):
        '''
        x is O (O = Net1(X).retain_grad.grad.detach()), which is the gradient from the forward model
        x.shape: torch.Size([256, 10]). This is because O is the gradient of the loss function with respect to the output.
        Function of the loops:
            - Update / store the gradient to ha and hb.
        '''
        
        for i in reversed(range(len(self.layers))):
            '''
            x.shape: loop 1 | layer 3: torch.Size([256, 10]) -> torch.Size([256, 200]) 
                     loop 2 | layer 2: torch.Size([256, 200]) -> torch.Size([256, 200])
                     loop 3 | layer 1: torch.Size([256, 200]) -> torch.Size([256, 200])
                     loop 4 | layer 0: torch.Size([256, 200]) -> torch.Size([256, 3072])
            '''
            x = self.layers[i].backproj_nobias(x)
            if i > 0:
                x = backward_norm_nl(
                    x, self.backward_norm_type, self.backward_nonlin_fn
                )
                if nonlin:
                    x = nonlin(x)
                x = self.dropout_back(x)
        return x

    def forward(self, x, nonlin=None):
        used_nonlin = nonlin if nonlin else self.F
        x = x.view(x.shape[0], -1)

        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if nonlin is None:
                x = forward_norm_nl(
                    x, self.forward_norm_type, self.forward_nonlin_fn
                )
            else:
                x = used_nonlin(x)
            x = self.dropout_forward(x)

        x = self.layers[-1](x)
        return x

# === tracing_sgd.py ===
"""
SGD with per-step pre-momentum update decomposition (traced V / V_bar only).

PyTorch SGD uses d_p = grad + weight_decay * p, then momentum on d_p, then p -= lr * d_p.
Tracing ignores momentum and records mean |.| *before* lr and momentum:

  - update_pre_wd:   |grad|
  - update_wd:       |weight_decay * p|
  - update_post_wd:  |grad + weight_decay * p|

(comparable across learning rates; multiply by lr for the pre-momentum step scale)

The real optimizer step still applies momentum as usual.
"""
from typing import Iterable, List, Optional, Set

import torch
from torch import Tensor
from torch.optim import Optimizer


def _mean_abs(t: Optional[Tensor]) -> float:
    if t is None:
        return float("nan")
    return t.detach().abs().mean().item()


class TracingSGD(Optimizer):
    """SGD matching PyTorch defaults; traces selected params on each step()."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        traced_params: Optional[Iterable[Tensor]] = None,
        param_layer_index: Optional[dict] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov requires momentum > 0 and dampening == 0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)
        self.traced_params: Set[Tensor] = set(traced_params or [])
        # V.weight -> layer index (required for per-layer stats when some V have no grad).
        self.param_layer_index: dict = dict(param_layer_index or {})
        self.last_v_traces: Optional[List[dict]] = None

    def _init_state(self, p: Tensor, momentum: float) -> None:
        state = self.state[p]
        if len(state) == 0:
            if momentum > 0:
                state["momentum_buffer"] = torch.zeros_like(p)
            state["trace"] = p in self.traced_params

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        traces: List[dict] = []

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                self._init_state(p, momentum)
                state = self.state[p]
                do_trace = state.get("trace", False)

                if do_trace:
                    pre_wd = grad
                    wd_only = (
                        p.detach().mul(weight_decay)
                        if weight_decay != 0
                        else torch.zeros_like(p)
                    )
                    post_wd = grad
                    if weight_decay != 0:
                        post_wd = post_wd.add(p, alpha=weight_decay)
                    n_el = p.numel()
                    traces.append(
                        {
                            "layer_idx": self.param_layer_index.get(id(p)),
                            "update_pre_wd_sum": pre_wd.abs().sum().item(),
                            "update_wd_sum": wd_only.abs().sum().item(),
                            "update_post_wd_sum": post_wd.abs().sum().item(),
                            "numel": n_el,
                            # Legacy keys used by JSON / plotter (same three terms).
                            "update_no_wd": _mean_abs(pre_wd),
                            "update_wd": _mean_abs(wd_only),
                            "update_total": _mean_abs(post_wd),
                        }
                    )

                d_p = grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = grad.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-lr)

        self.last_v_traces = traces if traces else None
        return loss


def v_params_from_param_groups(param_groups: List[dict]) -> List[Tensor]:
    """Second group is V / V_bar when build_net_v_sgd_param_groups layout is used."""
    if len(param_groups) < 2:
        return []
    return list(param_groups[1]["params"])


def v_param_layer_index(net) -> dict:
    """Map id(V.weight) -> layer index (identity-safe for Optimizer.step)."""
    out = {}
    for li, layer in enumerate(net.layers):
        v = getattr(layer, "v", None)
        if v is not None:
            out[id(v.weight)] = li
    return out


def build_tracing_sgd(
    param_groups: List[dict],
    lr: float,
    momentum: float,
    trace_v: bool = True,
    net=None,
) -> TracingSGD:
    traced = v_params_from_param_groups(param_groups) if trace_v else []
    layer_index = v_param_layer_index(net) if net is not None else {}
    return TracingSGD(
        param_groups,
        lr=lr,
        momentum=momentum,
        traced_params=traced,
        param_layer_index=layer_index,
    )


def empty_v_update_stats(num_layers: int) -> dict:
    """Full merge_v_and_vbar shape with NaNs (used when --trace_v_updates is off)."""
    nan = float("nan")
    z = [nan] * num_layers
    return {
        "v_update_no_wd_global": nan,
        "v_update_wd_global": nan,
        "v_update_total_global": nan,
        "v_update_no_wd_layer": list(z),
        "v_update_wd_layer": list(z),
        "v_update_total_layer": list(z),
        "v_bar_update_no_wd_global": nan,
        "v_bar_update_wd_global": nan,
        "v_bar_update_total_global": nan,
        "v_bar_update_no_wd_layer": list(z),
        "v_bar_update_wd_layer": list(z),
        "v_bar_update_total_layer": list(z),
        "v_weight_abs_global": nan,
        "v_bar_weight_abs_global": nan,
        "v_weight_abs_layer": list(z),
        "v_bar_weight_abs_layer": list(z),
    }


def collect_v_update_stats(tracing_opt: Optimizer, num_layers: int) -> dict:
    """Layer-ordered stats from last TracingSGD.step() (V params only)."""
    out = empty_v_update_stats(num_layers)
    if not isinstance(tracing_opt, TracingSGD):
        return out
    traces = tracing_opt.last_v_traces
    if not traces:
        return out

    def _fill(prefix: str, key: str, sum_key: str) -> None:
        layer_key = f"{prefix}_layer"
        layer_vals = [float("nan")] * num_layers
        total_abs, count = 0.0, 0
        for t in traces:
            li = t.get("layer_idx")
            if li is None or not (0 <= li < num_layers):
                continue
            layer_vals[li] = t[key]
            total_abs += t[sum_key]
            count += t["numel"]
        out[layer_key] = layer_vals
        out[f"{prefix}_global"] = total_abs / count if count else float("nan")

    # v_update_no_wd = pre-WD; v_update_wd = WD term; v_update_total = post-WD (pre-momentum).
    _fill("v_update_no_wd", "update_no_wd", "update_pre_wd_sum")
    _fill("v_update_wd", "update_wd", "update_wd_sum")
    _fill("v_update_total", "update_total", "update_post_wd_sum")
    return out


def merge_v_and_vbar_update_stats(v_opt: Optimizer, vbar_opt: Optimizer, num_layers: int) -> dict:
    """Net1 V traces + Net2 V_bar traces with v_bar_* keys."""
    v = collect_v_update_stats(v_opt, num_layers)
    vb = collect_v_update_stats(vbar_opt, num_layers)
    return {
        "v_update_no_wd_global": v["v_update_no_wd_global"],
        "v_update_wd_global": v["v_update_wd_global"],
        "v_update_total_global": v["v_update_total_global"],
        "v_update_no_wd_layer": v["v_update_no_wd_layer"],
        "v_update_wd_layer": v["v_update_wd_layer"],
        "v_update_total_layer": v["v_update_total_layer"],
        "v_bar_update_no_wd_global": vb["v_update_no_wd_global"],
        "v_bar_update_wd_global": vb["v_update_wd_global"],
        "v_bar_update_total_global": vb["v_update_total_global"],
        "v_bar_update_no_wd_layer": vb["v_update_no_wd_layer"],
        "v_bar_update_wd_layer": vb["v_update_wd_layer"],
        "v_bar_update_total_layer": vb["v_update_total_layer"],
    }


# === utils/eurosat.py ===
# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from torchvision.datasets import ImageFolder

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from PIL import ImageEnhance

transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)


class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]


    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out


identity = lambda x:x
class SimpleDataset:
    def __init__(self, root, transform, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform

        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []

        d = ImageFolder(os.path.join(root))

        for i, (data, label) in enumerate(d):
            self.meta['image_names'].append(data)
            self.meta['image_labels'].append(label)  

    def __getitem__(self, i):

        img = self.transform(self.meta['image_names'][i])
        target = self.target_transform(self.meta['image_labels'][i])

        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, root, batch_size, transform):

        self.sub_meta = {}
        self.cl_list = range(10)

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        d = ImageFolder(os.path.join(root, '2750'))

        for i, (data, label) in enumerate(d):
            self.sub_meta[label].append(data)

        for key, item in self.sub_meta.items():
            print (len(self.sub_meta[key]))
    
        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):

        img = self.transform(self.sub_meta[i])
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomResizedCrop':
            return method(self.image_size)
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Resize':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        elif transform_type=='ToTensor':
            return method()
        else:
            return method(self.image_size)

    def get_composed_transform(self, aug=False, normalise=True):
        if aug:
            if normalise:
                transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
            else:
                transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor']
        else:
            if normalise:
                transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']
            else:
                transform_list = ['Resize','CenterCrop', 'ToTensor']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class SimpleDataManager(DataManager):
    def __init__(self, root, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.root = root
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug, normalise): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug, normalise)
        dataset = SimpleDataset(self.root, transform)

        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, root, image_size, n_way=5, n_support=5, n_query=16, n_episode = 100):        
        super(SetDataManager, self).__init__()
        self.root = root
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug, normalise): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug, normalise)
        dataset = SetDataset(self.root, self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':
    pass

# === utils/data_factory.py ===


def load_data(root, dataset, batch_size):
    
    if dataset == 'pathmnist':
        
        import medmnist
        from medmnist import INFO
        data_flag = 'pathmnist'
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        
        data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[.5], std=[.5])
                ])

        train_dataset = DataClass(split='train', transform=data_transform, download=True)
        test_dataset = DataClass(split='test', transform=data_transform, download=True)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False)
        return train_loader, test_loader
        
    if dataset == 'eurosat':
        image_size = 64
        data_manager = SimpleDataManager(root, image_size, batch_size)
        train_dataloader = data_manager.get_data_loader(aug=True, normalise=True)
        test_dataloader = data_manager.get_data_loader(aug=False, normalise=True)
        return train_dataloader, test_dataloader

    if dataset in ['mnist', 'cifar10', 'fashion_mnist', 'svhn', 'stl10']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset == 'flowers102':
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to a common size for other datasets
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if dataset == 'mnist':
        ds = datasets.MNIST
    elif dataset == 'cifar10':
        ds = datasets.CIFAR10
    elif dataset == 'fashion_mnist':
        ds = datasets.FashionMNIST
    elif dataset == 'svhn':
        ds = datasets.SVHN
    elif dataset == 'stl10':
        ds = datasets.STL10
    elif dataset == 'caltech101':
        ds = datasets.Caltech101
    elif dataset == 'oxford_iiit_pet':
        ds = datasets.OxfordIIITPet
    elif dataset == 'flowers102':
        ds = datasets.Flowers102
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    # Adjust parameters based on the dataset
    if dataset in ['mnist', 'cifar10', 'fashion_mnist']:
        train_data = ds(root=root, train=True, download=True, transform=transform)
        test_data = ds(root=root, train=False, download=True, transform=transform)
    elif dataset in ['svhn']:
        train_data = ds(root=root, split='train', download=True, transform=transform)
        test_data = ds(root=root, split='test', download=True, transform=transform)
    elif dataset == 'caltech101':
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_data, test_data = random_split(dataset, [train_size, test_size])
    elif dataset == 'stl10':
        train_data = ds(root=root, split='train', download=True, transform=transform)
        test_data = ds(root=root, split='test', download=True, transform=transform)
    elif dataset in ['oxford_iiit_pet']:
        train_data = ds(root=root, split='trainval', download=True, transform=transform)
        test_data = ds(root=root, split='test', download=True, transform=transform)
    elif dataset in ['flowers102']:
        train_data = ds(root=root, split='train', download=True, transform=transform)
        test_data = ds(root=root, split='test', download=True, transform=transform)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

# === main ===
import json
import math
import os
import shlex
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm



def format_plot_command(args) -> str:
    """Shell-ready command to plot stat_timeseries.json for this run."""
    plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_stat_timeseries.py")
    parts = [
        shlex.quote(sys.executable),
        shlex.quote(plot_script),
        "--exp_root",
        shlex.quote(str(args.exp_root)),
        "--run_name",
        shlex.quote(str(args.run_name)),
    ]
    return " ".join(parts)


def copy_params_data(src: nn.Module, dst: nn.Module) -> None:
    with torch.no_grad():
        for p_dst, p_src in zip(dst.parameters(), src.parameters()):
            p_dst.data.copy_(p_src.data)


def copy_grads_data(src: nn.Module, dst: nn.Module) -> None:
    """Copy gradients from src to dst; clear dst grads when src has no grad."""
    for p_dst, p_src in zip(dst.parameters(), src.parameters()):
        if p_src.grad is None:
            p_dst.grad = None
        else:
            p_dst.grad = p_src.grad.detach().clone()


def _mean_abs_tensor(t):
    if t is None:
        return float("nan")
    return t.detach().abs().mean().item()


def _global_mean_abs_tensors(tensors):
    """Weighted mean |.| over tensors without torch.cat."""
    total, count = 0.0, 0
    for t in tensors:
        if t is None:
            continue
        td = t.detach()
        total += td.abs().sum().item()
        count += td.numel()
    return total / count if count else float("nan")


def _grad_angle_deg(g_sal, g_bp):
    if g_sal is None or g_bp is None:
        return float("nan"), float("nan")
    a = g_sal.detach().flatten().float()
    b = g_bp.detach().flatten().float()
    if a.numel() == 0 or a.norm() < 1e-12 or b.norm() < 1e-12:
        return float("nan"), float("nan")
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).clamp(-1.0, 1.0).item()
    return cos, math.degrees(math.acos(cos))


def _relu_gate_mask(x_pre: torch.Tensor) -> torch.Tensor:
    """Diagonal ReLU gate (per sample, per feature)."""
    return (x_pre.detach() > 0).to(dtype=x_pre.dtype)


def _elementwise_gate_mask(x_pre: torch.Tensor, nonlin_name: str):
    """Per-sample diagonal multipliers for identity/ReLU elementwise maps."""
    if nonlin_name == "identity":
        return torch.ones_like(x_pre.detach())
    if nonlin_name == "relu":
        return _relu_gate_mask(x_pre)
    return None


def _upstream_gate_mask(hb_prev: torch.Tensor, forward_norm_type: str, forward_nonlin: str):
    """Per-sample multipliers for D_u^l in T^l = D_u^l W^T."""
    if forward_norm_type != "none":
        return None
    return _elementwise_gate_mask(hb_prev, forward_nonlin)


def _backward_gate_mask(back_pre: torch.Tensor, backward_norm_type: str, backward_nonlin: str):
    """Per-sample multipliers for D_d^l in B^l = D_d^l W_bar V_bar^T."""
    if backward_norm_type != "none":
        return None
    return _elementwise_gate_mask(back_pre, backward_nonlin)


def _all_ones_mask(mask: torch.Tensor) -> bool:
    return bool(torch.all(mask == 1).item())


def _batch_ep_fro_norms(
    V: torch.Tensor,
    W: torch.Tensor,
    Vb: torch.Tensor,
    Wb: torch.Tensor,
    du: torch.Tensor,
    dd: torch.Tensor,
) -> tuple[list[float], list[float]]:
    """
    Per-sample ||E_p^l(x)||_F and ||F^l(x)||_F without materializing D or E.

    E_p = V D_u W^T - D_d W_bar V_bar^T (SI v2 notation; directional matmul layout).
    """
    d_in = W.shape[1]
    d_out = W.shape[0]
    bridge_dim = min(Wb.shape[0], Vb.shape[0])
    W_T = W.T[:d_in, :d_out]
    V_c = V[:, :d_in]
    Wb_paper = Wb.T[:, :bridge_dim]
    Vb_paper_t = Vb[:bridge_dim, :d_out]
    r0 = min(V_c.shape[0], Wb_paper.shape[0])
    c0 = min(W_T.shape[1], Vb_paper_t.shape[1])

    V_r = V_c[:r0, :]
    W_r = Wb_paper[:r0, :]
    WT_c = W_T[:, :c0]
    VB_c = Vb_paper_t[:, :c0]
    du_c = du[:, :d_in]
    dd_c = dd[:, :r0]

    f_quad = (V_r.T @ V_r) * (WT_c @ WT_c.T)
    f_sq = (du_c @ f_quad * du_c).sum(dim=1)

    b_base = W_r @ VB_c
    b_row_sq = (b_base * b_base).sum(dim=1)
    cross_rows = V_r * (b_base @ WT_c.T)

    if _all_ones_mask(dd_c):
        b_sq = b_row_sq.sum().expand(du_c.shape[0])
        cross = du_c @ cross_rows.sum(dim=0)
    else:
        b_sq = (dd_c.square() * b_row_sq).sum(dim=1)
        cross = torch.einsum("br,bd,rd->b", dd_c, du_c, cross_rows)

    e_sq = (f_sq + b_sq - 2.0 * cross).clamp_min(0.0)
    return (
        torch.sqrt(e_sq).detach().cpu().tolist(),
        torch.sqrt(f_sq.clamp_min(0.0)).detach().cpu().tolist(),
    )


def collect_ep_pathway_alignment(
    net1: Net_V,
    net2: Net_V,
    forward_norm_type: str = "none",
    backward_norm_type: str = "none",
    forward_nonlin: str = "relu",
    backward_nonlin: str = "identity",
) -> dict:
    """
    Pathway-alignment E_p^l(x) from PNAS_SI_ver2.tex:
      E_p^l = V^l D_u^l (W^l)^T - D_d^l W_bar^l (bar V^l)^T
    Returns mean ||E_p||_F and ||E_p||/||F|| per layer (minibatch mean).

    The operator diagnostic is implemented for identity/ReLU elementwise nonlinearities
    without normalization. Other nonlinearities/norms need a full samplewise Jacobian.
    """
    n = len(net1.layers)
    if (
        forward_norm_type != "none"
        or backward_norm_type != "none"
        or forward_nonlin not in ("identity", "relu")
        or backward_nonlin not in ("identity", "relu")
    ):
        nan_layers = [float("nan")] * n
        return {
            "ep_pathway_norm_layer": nan_layers,
            "ep_pathway_rel_layer": nan_layers,
            "ep_pathway_norm_global": float("nan"),
            "ep_pathway_rel_global": float("nan"),
        }

    layer_norms, layer_rels = [], []
    ep_tensors, f_tensors = [], []

    for l in range(n):
        t = net1.layers[l]
        s = net2.layers[l]
        if t.ha is None or t.hb is None or t.v is None or s.v is None:
            layer_norms.append(float("nan"))
            layer_rels.append(float("nan"))
            continue
        if l + 1 >= n:
            layer_norms.append(float("nan"))
            layer_rels.append(float("nan"))
            continue

        ha = t.ha.detach()
        if l == 0:
            du = torch.ones_like(ha)
        else:
            du = _upstream_gate_mask(
                net1.layers[l - 1].hb, forward_norm_type, forward_nonlin
            )
        if l == 0:
            dd = torch.ones_like(ha)
        else:
            dd = _backward_gate_mask(s.sha, backward_norm_type, backward_nonlin)
        if du is None or dd is None:
            layer_norms.append(float("nan"))
            layer_rels.append(float("nan"))
            continue

        V, W = t.v.weight.detach(), t.w.weight.detach()
        Vb, Wb = s.v.weight.detach(), s.w.weight.detach()
        ep_b, f_b = _batch_ep_fro_norms(V, W, Vb, Wb, du, dd)
        layer_norms.append(float(np.mean(ep_b)))
        layer_rels.append(
            float(np.mean([e / f if f > 1e-12 else float("nan") for e, f in zip(ep_b, f_b)]))
        )
        ep_tensors.extend(ep_b)
        f_tensors.extend(f_b)

    rel_pairs = [(e, f) for e, f in zip(ep_tensors, f_tensors) if f > 1e-12]
    return {
        "ep_pathway_norm_layer": layer_norms,
        "ep_pathway_rel_layer": layer_rels,
        "ep_pathway_norm_global": float(np.mean(ep_tensors)) if ep_tensors else float("nan"),
        "ep_pathway_rel_global": float(np.mean([e / f for e, f in rel_pairs]))
        if rel_pairs
        else float("nan"),
    }


def collect_v_grad_stats(net1: Net_V, net2: Net_V):
    """
    Mean |grad| for V (Net1.v) and V_bar (Net2.v), per same layer index (not junction-paired).
    """
    v_layer, v_bar_layer = [], []
    v_tensors, v_bar_tensors = [], []
    for l1, l2 in zip(net1.layers, net2.layers):
        g_v = l1.v.weight.grad if l1.v is not None else None
        g_vb = l2.v.weight.grad if l2.v is not None else None
        v_layer.append(_mean_abs_tensor(g_v))
        v_bar_layer.append(_mean_abs_tensor(g_vb))
        if g_v is not None:
            v_tensors.append(g_v)
        if g_vb is not None:
            v_bar_tensors.append(g_vb)
    return {
        "v_grad_abs_layer": v_layer,
        "v_bar_grad_abs_layer": v_bar_layer,
        "v_grad_abs_global": _global_mean_abs_tensors(v_tensors),
        "v_bar_grad_abs_global": _global_mean_abs_tensors(v_bar_tensors),
    }


def collect_v_weight_stats(net1: Net_V, net2: Net_V):
    """Mean |V| / |V_bar| (abs mean of bridge conv weights) after optimizer step."""
    v_layer, v_bar_layer = [], []
    v_tensors, v_bar_tensors = [], []
    for l1, l2 in zip(net1.layers, net2.layers):
        w_v = l1.v.weight if l1.v is not None else None
        w_vb = l2.v.weight if l2.v is not None else None
        v_layer.append(_mean_abs_tensor(w_v))
        v_bar_layer.append(_mean_abs_tensor(w_vb))
        if w_v is not None:
            v_tensors.append(w_v)
        if w_vb is not None:
            v_bar_tensors.append(w_vb)
    return {
        "v_weight_abs_layer": v_layer,
        "v_bar_weight_abs_layer": v_bar_layer,
        "v_weight_abs_global": _global_mean_abs_tensors(v_tensors),
        "v_bar_weight_abs_global": _global_mean_abs_tensors(v_bar_tensors),
    }


def merge_v_trace_stats(v_update_stats, v_weight_stats) -> dict:
    out = dict(v_update_stats)
    out.update(v_weight_stats)
    return out


def _paired_v_aligned(w_next, w_bar, symmetric_vv: bool):
    """Aligned V_{i+1} and V_bar_i blocks (transpose V_bar when not symmetric_vv)."""
    w_bar_cmp = w_bar if symmetric_vv else w_bar.T
    if w_next.shape != w_bar_cmp.shape:
        r = min(w_next.shape[0], w_bar_cmp.shape[0])
        c = min(w_next.shape[1], w_bar_cmp.shape[1])
        return w_next[:r, :c], w_bar_cmp[:r, :c]
    return w_next, w_bar_cmp


def _paired_v_diff_matrix(w_next, w_bar, symmetric_vv: bool):
    a, b = _paired_v_aligned(w_next, w_bar, symmetric_vv)
    return (a - b).abs()


def collect_v_cross_weight_diff(net1: Net_V, net2: Net_V, symmetric_vv: bool = False):
    """
    Junction-paired weight distance: Net1.layers[i+1].v vs Net2.layers[i].v.
    With --symmetric_vv: |V_{i+1} - V_bar_i| (same orientation; grads target equality).
    Otherwise: |V_{i+1} - V_bar_i^T| (legacy).
    """
    layer_diffs = []
    total_abs, total_numel = 0.0, 0
    n = len(net1.layers)
    for i in range(n - 1):
        v_next = net1.layers[i + 1].v
        v_bar = net2.layers[i].v
        if v_next is None or v_bar is None:
            layer_diffs.append(float("nan"))
            continue
        diff = _paired_v_diff_matrix(
            v_next.weight.detach(), v_bar.weight.detach(), symmetric_vv
        )
        layer_diffs.append(diff.mean().item())
        total_abs += diff.sum().item()
        total_numel += diff.numel()
    global_diff = total_abs / total_numel if total_numel else float("nan")
    label = "V_{i+1} - V_bar_i" if symmetric_vv else "V_{i+1} - V_bar_i^T"
    return {
        "v_cross_weight_diff_layer": layer_diffs,
        "v_cross_weight_diff_global": global_diff,
        "v_cross_weight_diff_label": label,
    }


def collect_v_paired_weight_cosine(net1: Net_V, net2: Net_V, symmetric_vv: bool = False):
    """Cosine similarity between junction-paired V weights (same pairing as |V diff|)."""
    layer_cos = []
    flat_a, flat_b = [], []
    n = len(net1.layers)
    for i in range(n - 1):
        v_next = net1.layers[i + 1].v
        v_bar = net2.layers[i].v
        if v_next is None or v_bar is None:
            layer_cos.append(float("nan"))
            continue
        a, b = _paired_v_aligned(
            v_next.weight.detach(), v_bar.weight.detach(), symmetric_vv
        )
        cos, _ = _grad_angle_deg(a, b)
        layer_cos.append(cos)
        flat_a.append(a.reshape(-1))
        flat_b.append(b.reshape(-1))
    if flat_a:
        global_cos, _ = _grad_angle_deg(torch.cat(flat_a), torch.cat(flat_b))
    else:
        global_cos = float("nan")
    pair_label = "cos(V_{i+1}, V_bar_i)" if symmetric_vv else "cos(V_{i+1}, V_bar_i^T)"
    return {
        "v_paired_weight_cos_layer": layer_cos,
        "v_paired_weight_cos_global": global_cos,
        "v_paired_weight_cos_label": pair_label,
    }


def collect_v_paired_grad_diff(net1: Net_V, net2: Net_V, symmetric_vv: bool = False):
    """|grad V_{i+1} - grad V_bar_i| (or vs grad V_bar_i^T if not symmetric_vv)."""
    layer_diffs = []
    total_abs, total_numel = 0.0, 0
    n = len(net1.layers)
    for i in range(n - 1):
        v_next = net1.layers[i + 1].v
        v_bar = net2.layers[i].v
        if (
            v_next is None
            or v_bar is None
            or v_next.weight.grad is None
            or v_bar.weight.grad is None
        ):
            layer_diffs.append(float("nan"))
            continue
        diff = _paired_v_diff_matrix(
            v_next.weight.grad.detach(),
            v_bar.weight.grad.detach(),
            symmetric_vv,
        )
        layer_diffs.append(diff.mean().item())
        total_abs += diff.sum().item()
        total_numel += diff.numel()
    global_diff = total_abs / total_numel if total_numel else float("nan")
    return {
        "v_paired_grad_diff_layer": layer_diffs,
        "v_paired_grad_diff_global": global_diff,
    }


def compare_forward_w_grads(net1: Net_V, net1_bp: Net_V):
    layer_cos, layer_angles = [], []
    sal_chunks, bp_chunks = [], []
    for layer_sal, layer_bp in zip(net1.layers, net1_bp.layers):
        g_sal, g_bp = layer_sal.w.weight.grad, layer_bp.w.weight.grad
        cos, ang = _grad_angle_deg(g_sal, g_bp)
        layer_cos.append(cos)
        layer_angles.append(ang)
        if g_sal is not None:
            sal_chunks.append(g_sal.reshape(-1))
        if g_bp is not None:
            bp_chunks.append(g_bp.reshape(-1))
    g_sal = torch.cat(sal_chunks) if sal_chunks else None
    g_bp = torch.cat(bp_chunks) if bp_chunks else None
    global_cos, global_angle = _grad_angle_deg(g_sal, g_bp)
    return layer_cos, layer_angles, global_cos, global_angle


def minibatch_accuracy_net1(X, y, net1):
    """Net1 classification accuracy on the current train minibatch (%)."""
    y_cls = y.argmax(dim=1) if y.dim() > 1 else y
    with torch.no_grad():
        acc = (net1(X).argmax(1) == y_cls).float().mean().item() * 100.0
    return {"net1": acc}


def _mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    return float(np.mean(xs)), float(np.std(xs))


def _aggregate_running(buf):
    """Mean per scalar list; mean per list-of-lists (layerwise)."""
    out = {}
    for key, vals in buf.items():
        if not vals:
            out[key] = float("nan")
            continue
        if isinstance(vals[0], list):
            n_layers = len(vals[0])
            out[key] = [
                float(np.nanmean([v[i] for v in vals if len(v) > i]))
                for i in range(n_layers)
            ]
        else:
            out[key] = float(np.mean(vals))
    return out


def _batch_snap(
    loss,
    loss2,
    layer_cos,
    layer_angles,
    g_cos,
    g_angle,
    v_stats,
    cross_stats,
    paired_cos_stats,
    paired_grad_stats,
    ep_stats=None,
    v_update_stats=None,
):
    """Per-minibatch stat dict (instantaneous, not window-averaged)."""
    snap = {
        "loss_ff": float(loss),
        "loss_gn": float(loss2),
        "w_cos_global": float(g_cos),
        "w_angle_global": float(g_angle),
        "w_cos_layer": list(layer_cos),
        "w_angle_layer": list(layer_angles),
        "v_grad_abs_global": float(v_stats["v_grad_abs_global"]),
        "v_bar_grad_abs_global": float(v_stats["v_bar_grad_abs_global"]),
        "v_grad_abs_layer": list(v_stats["v_grad_abs_layer"]),
        "v_bar_grad_abs_layer": list(v_stats["v_bar_grad_abs_layer"]),
        "v_cross_weight_diff_global": float(cross_stats["v_cross_weight_diff_global"]),
        "v_cross_weight_diff_layer": list(cross_stats["v_cross_weight_diff_layer"]),
        "v_cross_weight_diff_label": cross_stats.get(
            "v_cross_weight_diff_label", "V_{i+1} - V_bar_i^T"
        ),
        "v_paired_weight_cos_global": float(paired_cos_stats["v_paired_weight_cos_global"]),
        "v_paired_weight_cos_layer": list(paired_cos_stats["v_paired_weight_cos_layer"]),
        "v_paired_weight_cos_label": paired_cos_stats.get(
            "v_paired_weight_cos_label", "cos(V_{i+1}, V_bar_i^T)"
        ),
        "v_paired_grad_diff_global": float(paired_grad_stats["v_paired_grad_diff_global"]),
        "v_paired_grad_diff_layer": list(paired_grad_stats["v_paired_grad_diff_layer"]),
    }
    if ep_stats is not None:
        snap.update(
            {
                "ep_pathway_norm_global": float(ep_stats["ep_pathway_norm_global"]),
                "ep_pathway_rel_global": float(ep_stats["ep_pathway_rel_global"]),
                "ep_pathway_norm_layer": list(ep_stats["ep_pathway_norm_layer"]),
                "ep_pathway_rel_layer": list(ep_stats["ep_pathway_rel_layer"]),
            }
        )
    if v_update_stats is not None:
        snap.update(
            {
                "v_update_no_wd_global": float(v_update_stats["v_update_no_wd_global"]),
                "v_update_wd_global": float(v_update_stats["v_update_wd_global"]),
                "v_update_total_global": float(v_update_stats["v_update_total_global"]),
                "v_bar_update_no_wd_global": float(v_update_stats["v_bar_update_no_wd_global"]),
                "v_bar_update_wd_global": float(v_update_stats["v_bar_update_wd_global"]),
                "v_bar_update_total_global": float(
                    v_update_stats["v_bar_update_total_global"]
                ),
                "v_update_no_wd_layer": list(v_update_stats["v_update_no_wd_layer"]),
                "v_update_wd_layer": list(v_update_stats["v_update_wd_layer"]),
                "v_update_total_layer": list(v_update_stats["v_update_total_layer"]),
                "v_bar_update_no_wd_layer": list(v_update_stats["v_bar_update_no_wd_layer"]),
                "v_bar_update_wd_layer": list(v_update_stats["v_bar_update_wd_layer"]),
                "v_bar_update_total_layer": list(
                    v_update_stats["v_bar_update_total_layer"]
                ),
                "v_weight_abs_global": float(v_update_stats["v_weight_abs_global"]),
                "v_bar_weight_abs_global": float(
                    v_update_stats["v_bar_weight_abs_global"]
                ),
                "v_weight_abs_layer": list(v_update_stats["v_weight_abs_layer"]),
                "v_bar_weight_abs_layer": list(v_update_stats["v_bar_weight_abs_layer"]),
            }
        )
    return snap


class StatsRecorder:
    """Append training stats every R global iterations for JSON + plots."""

    _SCALAR_KEYS_BASE = (
        "loss_ff",
        "loss_gn",
        "w_cos_global",
        "w_angle_global",
        "v_grad_abs_global",
        "v_bar_grad_abs_global",
        "v_cross_weight_diff_global",
        "v_paired_weight_cos_global",
        "v_paired_grad_diff_global",
        "ep_pathway_norm_global",
        "ep_pathway_rel_global",
        "acc_net1",
    )
    _V_UPDATE_SCALAR_KEYS = (
        "v_update_no_wd_global",
        "v_update_wd_global",
        "v_update_total_global",
        "v_bar_update_no_wd_global",
        "v_bar_update_wd_global",
        "v_bar_update_total_global",
        "v_weight_abs_global",
        "v_bar_weight_abs_global",
    )
    _LAYER_KEYS_BASE = (
        "w_cos_layer",
        "w_angle_layer",
        "v_grad_abs_layer",
        "v_bar_grad_abs_layer",
        "v_cross_weight_diff_layer",
        "v_paired_weight_cos_layer",
        "v_paired_grad_diff_layer",
        "ep_pathway_norm_layer",
        "ep_pathway_rel_layer",
    )
    _V_UPDATE_LAYER_KEYS = (
        "v_update_no_wd_layer",
        "v_update_wd_layer",
        "v_update_total_layer",
        "v_bar_update_no_wd_layer",
        "v_bar_update_wd_layer",
        "v_bar_update_total_layer",
        "v_weight_abs_layer",
        "v_bar_weight_abs_layer",
    )

    def __init__(self, record_every: int, trace_v_updates: bool = False):
        self.record_every = max(0, int(record_every))
        self.trace_v_updates = bool(trace_v_updates)
        self.global_step = 0
        self._last_snap = None
        self._last_accs = None
        self._last_epoch = 0
        self._last_batch = 0
        self._scalar_keys = self._SCALAR_KEYS_BASE + (
            self._V_UPDATE_SCALAR_KEYS if self.trace_v_updates else ()
        )
        self._layer_keys = self._LAYER_KEYS_BASE + (
            self._V_UPDATE_LAYER_KEYS if self.trace_v_updates else ()
        )
        self.data = {"step": [], "epoch": [], "batch": []}
        for k in self._scalar_keys:
            self.data[k] = []
        for k in self._layer_keys:
            self.data[k] = []

    def enabled(self):
        return self.record_every > 0

    def _append(self, snap, epoch, batch_idx, accuracies=None):
        self.data["step"].append(self.global_step)
        self.data["epoch"].append(int(epoch))
        self.data["batch"].append(int(batch_idx))
        for k in self._scalar_keys:
            if k.startswith("acc_"):
                if accuracies is None:
                    self.data[k].append(float("nan"))
                else:
                    key = k.replace("acc_", "")
                    self.data[k].append(float(accuracies[key]))
            else:
                self.data[k].append(float(snap[k]))
        for k in self._layer_keys:
            self.data[k].append(list(snap[k]))

    def on_batch(self, snap, epoch, batch_idx, accuracies=None):
        if not self.enabled():
            return
        self.global_step += 1
        self._last_snap = snap
        self._last_epoch = int(epoch)
        self._last_batch = int(batch_idx)
        if accuracies is not None:
            self._last_accs = accuracies
        if self.global_step % self.record_every == 0:
            self._append(snap, epoch, batch_idx, accuracies)

    def finalize(self):
        """Record final iteration if it was not aligned with record_every."""
        if not self.enabled() or self._last_snap is None:
            return
        if self.data["step"] and self.data["step"][-1] == self.global_step:
            return
        self._append(
            self._last_snap,
            self._last_epoch,
            self._last_batch,
            getattr(self, "_last_accs", None),
        )

    def to_jsonable(self):
        return dict(self.data)

    def build_save_payload(self, num_layers: int, record_every: int, epoch_metrics=None):
        payload = {
            "num_layers": int(num_layers),
            "record_every": int(record_every),
            "code_names": {
                "W": "Net1.layer.w",
                "W_prime": "Net2.layer.w",
                "V": "Net1.layer.v",
                "V_bar": "Net2.layer.v",
                "v_cross": "junction |Net1.layers[i+1].v - Net2.layers[i].v| (or .v.T if not symmetric_vv)",
                "v_paired_grad": "junction |grad V_{i+1} - grad V_bar_i|",
                "E_p": "pathway-alignment ||V D_u W^T - D_d W_bar V_bar^T||_F",
            },
            "series": self.to_jsonable(),
        }
        if epoch_metrics is not None:
            payload["epoch_metrics"] = epoch_metrics
        return payload


def build_epoch_metrics(losses, test_losses, accuracies1, train_accuracies1=None):
    """Per-epoch mean train/test loss (Net1 FF) and train/test accuracy (%)."""
    n = len(losses)
    em = {
        "epoch": list(range(1, n + 1)),
        "train_loss_ff": [float(l[0]) for l in losses],
        "train_loss_gn": [float(l[1]) for l in losses],
        "test_loss_ff": [float(x) for x in test_losses],
        "test_acc_net1": [float(a) for a in accuracies1],
    }
    if train_accuracies1 is not None:
        em["train_acc_net1"] = [float(a) for a in train_accuracies1]
    return em


def _print_epoch_summary(
    epoch,
    total_epochs,
    test_acc1,
    train_loss_ff,
    train_loss_gn,
    test_loss_ff,
    train_acc1=None,
    epoch_cmp=None,
    epoch_v=None,
):
    """Prominent epoch summary: Net1 train/test loss, accuracy, and SAL/BP diagnostics."""
    bar = "=" * 72
    print()
    print(bar)
    print(f"  EPOCH {epoch:3d} / {total_epochs}  |  NET1 (SAL) EPOCH SUMMARY")
    print(f"    Train loss (epoch mean)  FF={train_loss_ff:.4f}  GN={train_loss_gn:.4f}")
    if train_acc1 is not None:
        print(f"    Train accuracy (full epoch) {train_acc1:7.2f}%")
    print(f"    Test loss  (full set)    FF={test_loss_ff:.4f}")
    print(f"    Test accuracy (full set) {test_acc1:7.2f}%")
    if epoch_cmp is not None and epoch_v is not None:
        _print_epoch_diagnostics(epoch_cmp, epoch_v)
    print(bar)
    print()


def _fmt_v_trace(x):
    """Scientific notation for small pre-lr V trace magnitudes."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    return f"{float(x):.6e}"


def _fmt_v_trace_list(xs):
    return [_fmt_v_trace(x) for x in xs]


def _print_epoch_diagnostics(epoch_cmp, epoch_v):
    """SAL/BP diagnostics (printed inside the epoch summary banner)."""
    print(
        f"    W vs BP: cos={epoch_cmp['global_cosine_mean']:.4f} "
        f"angle={epoch_cmp['global_angle_deg_mean']:.2f}°"
    )
    print(
        f"    V: |grad|={epoch_v['v_grad_abs_global_mean']:.6f} "
        f"|V_i+1-Vbar_i|={epoch_v['v_cross_weight_diff_global_mean']:.6f} "
        f"cos(V)={epoch_v['v_paired_weight_cos_global_mean']:.4f} "
        f"|gradV pair|={epoch_v['v_paired_grad_diff_global_mean']:.6f} "
        f"||E_p||={epoch_v.get('ep_pathway_norm_global_mean', float('nan')):.6f}"
    )


def _print_intra_epoch_stats(
    epoch, batch_idx, n_batches, snap, accuracies=None, symmetric_vv=False
):
    print(
        f"  [epoch {epoch} batch {batch_idx}/{n_batches}] "
        f"loss FF={snap['loss_ff']:.4f} GN={snap['loss_gn']:.4f} | "
        f"W cos(global)={snap['w_cos_global']:.4f} angle(global)={snap['w_angle_global']:.2f}°"
    )
    w_cos_layer = snap.get("w_cos_layer")
    w_angle_layer = snap.get("w_angle_layer")
    if w_cos_layer is not None and w_angle_layer is not None:
        print(
            f"    W cos(layer)={[round(x, 4) if not np.isnan(x) else None for x in w_cos_layer]} "
            f"angle(layer)={[round(x, 2) if not np.isnan(x) else None for x in w_angle_layer]}"
        )
    if accuracies is not None:
        print(f"    minibatch acc Net1(SAL)={accuracies['net1']:.2f}%")
    if "v_update_no_wd_global" in snap:
        print(
            f"    |grad| V pre-WD(global)={_fmt_v_trace(snap['v_update_no_wd_global'])} "
            f"|wd*p|={_fmt_v_trace(snap['v_update_wd_global'])} "
            f"|g+wd*p|={_fmt_v_trace(snap['v_update_total_global'])}"
        )
        print(
            f"    |grad| V pre-WD(layer)={_fmt_v_trace_list(snap['v_update_no_wd_layer'])}"
        )
        print(f"    |wd*p| V WD(layer)={_fmt_v_trace_list(snap['v_update_wd_layer'])}")
        print(
            f"    |g+wd*p| V post-WD(layer)="
            f"{_fmt_v_trace_list(snap['v_update_total_layer'])}"
        )
        print(
            f"    |grad| V_bar pre-WD(global)={_fmt_v_trace(snap['v_bar_update_no_wd_global'])} "
            f"|wd*p|={_fmt_v_trace(snap['v_bar_update_wd_global'])} "
            f"|g+wd*p|={_fmt_v_trace(snap['v_bar_update_total_global'])}"
        )
        print(
            f"    |grad| V_bar pre-WD(layer)="
            f"{_fmt_v_trace_list(snap['v_bar_update_no_wd_layer'])}"
        )
        print(
            f"    |wd*p| V_bar WD(layer)={_fmt_v_trace_list(snap['v_bar_update_wd_layer'])}"
        )
        print(
            f"    |g+wd*p| V_bar post-WD(layer)="
            f"{_fmt_v_trace_list(snap['v_bar_update_total_layer'])}"
        )
        if "v_weight_abs_global" in snap:
            print(
                f"    |V|(global)={_fmt_v_trace(snap['v_weight_abs_global'])} "
                f"|V_bar|(global)={_fmt_v_trace(snap['v_bar_weight_abs_global'])} "
                f"(|wd*p| ~ weight_decay_v * |V|)"
            )
            print(
                f"    |V|(layer)={_fmt_v_trace_list(snap['v_weight_abs_layer'])}"
            )
            print(
                f"    |V_bar|(layer)={_fmt_v_trace_list(snap['v_bar_weight_abs_layer'])}"
            )
    else:
        print(
            f"    |grad| V(global)={snap['v_grad_abs_global']:.6f} "
            f"V_bar(global)={snap['v_bar_grad_abs_global']:.6f}"
        )
        print(
            f"    |grad| V(layer)="
            f"{[round(x, 6) if not np.isnan(x) else None for x in snap['v_grad_abs_layer']]}"
        )
        print(
            f"    |grad| V_bar(layer)="
            f"{[round(x, 6) if not np.isnan(x) else None for x in snap['v_bar_grad_abs_layer']]}"
        )
    vxl = snap.get(
        "v_cross_weight_diff_label",
        "V_{i+1} - V_bar_i" if symmetric_vv else "V_{i+1} - V_bar_i^T",
    )
    vcos_lbl = snap.get(
        "v_paired_weight_cos_label",
        "cos(V_{i+1}, V_bar_i)" if symmetric_vv else "cos(V_{i+1}, V_bar_i^T)",
    )
    print(
        f"    |{vxl}|(global)={snap['v_cross_weight_diff_global']:.6f} "
        f"(layer)={[round(x, 6) if not np.isnan(x) else None for x in snap['v_cross_weight_diff_layer']]}"
    )
    print(
        f"    {vcos_lbl}(global)={snap['v_paired_weight_cos_global']:.4f} "
        f"(layer)={[round(x, 4) if not np.isnan(x) else None for x in snap['v_paired_weight_cos_layer']]} "
        f"  [cos=angle; |diff|=entry mean; low cos + similar |diff| => different directions/norms]"
    )
    print(
        f"    |grad V_{{i+1}} - grad V_bar_i|(global)={snap['v_paired_grad_diff_global']:.6f} "
        f"(layer)={[round(x, 6) if not np.isnan(x) else None for x in snap['v_paired_grad_diff_layer']]}"
    )
    if "ep_pathway_norm_global" in snap:
        print(
            f"  ||E_p||(global)={snap['ep_pathway_norm_global']:.6f} "
            f"rel={snap.get('ep_pathway_rel_global', float('nan')):.4f} "
            f"(layer)={[round(x, 6) if not np.isnan(x) else None for x in snap['ep_pathway_norm_layer']]}"
        )


def train_epoch_with_bp_compare(
    dataloader,
    net1,
    net2,
    net1_bp,
    loss_fn,
    args,
    test_dataloader=None,
    epoch=1,
    recorder=None,
    sal_update_fn=None,
):
    if sal_update_fn is None:
        sal_update_fn = get_sal_update_fn(args)
    pg1 = build_net_v_sgd_param_groups(net1, args.weight_decay, args.weight_decay_v)
    pg2 = build_net_v_sgd_param_groups(net2, args.weight_decay, args.weight_decay_v_bar)
    trace_v = getattr(args, "trace_v_updates", False)
    if trace_v:
        opt1 = build_tracing_sgd(
            pg1, lr=args.lr, momentum=args.momentum, trace_v=True, net=net1
        )
        opt2 = build_tracing_sgd(
            pg2, lr=args.lr, momentum=args.momentum, trace_v=True, net=net2
        )
    else:
        opt1 = optim.SGD(pg1, lr=args.lr, momentum=args.momentum)
        opt2 = optim.SGD(pg2, lr=args.lr, momentum=args.momentum)

    epoch_losses = [0.0, 0.0]
    total_batches = 0
    train_correct = 0
    train_seen = 0
    num_layers = len(net1.layers)
    n_batches = len(dataloader)
    log_every = max(0, int(args.log_every))

    batch_layer_cos = [[] for _ in range(num_layers)]
    batch_layer_angles = [[] for _ in range(num_layers)]
    batch_global_cos = []
    batch_global_angles = []

    running = defaultdict(list)
    epoch_v_buf = defaultdict(list)
    net1.train()
    net2.train()
    net1_bp.train()

    for batch_idx, (X, y) in enumerate(dataloader):
        y = converter(y, args.num_classes).to(args.device)
        X, y = X.to(args.device), y.to(args.device)

        opt1.zero_grad()
        opt2.zero_grad()

        copy_params_data(net1, net1_bp)
        net1_bp.zero_grad()
        out_bp = net1_bp(X)
        loss_bp = loss_fn(out_bp, y)
        loss_bp.backward()

        loss, loss2 = sal_update_fn(net1, net2, X, y, loss_fn, args)

        layer_cos, layer_angles, g_cos, g_angle = compare_forward_w_grads(net1, net1_bp)
        for i, (c, a) in enumerate(zip(layer_cos, layer_angles)):
            if not np.isnan(c):
                batch_layer_cos[i].append(c)
                batch_layer_angles[i].append(a)
        if not np.isnan(g_cos):
            batch_global_cos.append(g_cos)
            batch_global_angles.append(g_angle)

        log_step = log_every > 0 and (batch_idx + 1) % log_every == 0
        record_step = (
            recorder is not None
            and recorder.enabled()
            and (recorder.global_step + 1) % recorder.record_every == 0
        )

        v_stats = collect_v_grad_stats(net1, net2)
        sym_vv = getattr(args, "symmetric_vv", False)
        cross_stats = collect_v_cross_weight_diff(net1, net2, symmetric_vv=sym_vv)
        paired_cos_stats = collect_v_paired_weight_cosine(net1, net2, symmetric_vv=sym_vv)
        paired_grad_stats = collect_v_paired_grad_diff(net1, net2, symmetric_vv=sym_vv)
        ep_stats = collect_ep_pathway_alignment(
            net1,
            net2,
            forward_norm_type=getattr(args, "forward_norm_type", "none"),
            backward_norm_type=getattr(args, "backward_norm_type", "none"),
            forward_nonlin=getattr(args, "forward_nonlin", "relu"),
            backward_nonlin=getattr(args, "backward_nonlin", "identity"),
        )
        for k in ("v_grad_abs_global", "v_bar_grad_abs_global"):
            epoch_v_buf[k].append(v_stats[k])
        epoch_v_buf["v_cross_weight_diff_global"].append(
            cross_stats["v_cross_weight_diff_global"]
        )
        epoch_v_buf["v_grad_abs_layer"].append(v_stats["v_grad_abs_layer"])
        epoch_v_buf["v_bar_grad_abs_layer"].append(v_stats["v_bar_grad_abs_layer"])
        epoch_v_buf["v_cross_weight_diff_layer"].append(
            cross_stats["v_cross_weight_diff_layer"]
        )
        epoch_v_buf["v_paired_grad_diff_global"].append(
            paired_grad_stats["v_paired_grad_diff_global"]
        )
        epoch_v_buf["v_paired_grad_diff_layer"].append(
            paired_grad_stats["v_paired_grad_diff_layer"]
        )
        epoch_v_buf["v_paired_weight_cos_global"].append(
            paired_cos_stats["v_paired_weight_cos_global"]
        )
        epoch_v_buf["v_paired_weight_cos_layer"].append(
            paired_cos_stats["v_paired_weight_cos_layer"]
        )
        epoch_v_buf["ep_pathway_norm_global"].append(ep_stats["ep_pathway_norm_global"])
        epoch_v_buf["ep_pathway_rel_global"].append(ep_stats["ep_pathway_rel_global"])
        epoch_v_buf["ep_pathway_norm_layer"].append(ep_stats["ep_pathway_norm_layer"])
        epoch_v_buf["ep_pathway_rel_layer"].append(ep_stats["ep_pathway_rel_layer"])

        if getattr(args, "force_backprop", False):
            copy_grads_data(net1_bp, net1)

        params = list(net1.parameters()) + list(net2.parameters())
        torch.nn.utils.clip_grad_norm_(params, 50)
        opt1.step()
        opt2.step()
        v_update_stats = None
        if trace_v:
            v_update_stats = merge_v_trace_stats(
                merge_v_and_vbar_update_stats(opt1, opt2, num_layers),
                collect_v_weight_stats(net1, net2),
            )

        if log_every > 0:
            running["loss_ff"].append(loss)
            running["loss_gn"].append(loss2)
            running["w_cos_global"].append(g_cos)
            running["w_angle_global"].append(g_angle)
            running["w_cos_layer"].append(layer_cos)
            running["w_angle_layer"].append(layer_angles)
            running["v_cross_weight_diff_global"].append(
                cross_stats["v_cross_weight_diff_global"]
            )
            running["v_cross_weight_diff_layer"].append(
                cross_stats["v_cross_weight_diff_layer"]
            )
            running["v_paired_grad_diff_global"].append(
                paired_grad_stats["v_paired_grad_diff_global"]
            )
            running["v_paired_grad_diff_layer"].append(
                paired_grad_stats["v_paired_grad_diff_layer"]
            )
            running["v_paired_weight_cos_global"].append(
                paired_cos_stats["v_paired_weight_cos_global"]
            )
            running["v_paired_weight_cos_layer"].append(
                paired_cos_stats["v_paired_weight_cos_layer"]
            )
            running["ep_pathway_norm_global"].append(ep_stats["ep_pathway_norm_global"])
            running["ep_pathway_rel_global"].append(ep_stats["ep_pathway_rel_global"])
            running["ep_pathway_norm_layer"].append(ep_stats["ep_pathway_norm_layer"])
            running["ep_pathway_rel_layer"].append(ep_stats["ep_pathway_rel_layer"])
            if trace_v and v_update_stats is not None:
                for k, v in v_update_stats.items():
                    running[k].append(v)
            else:
                for k in ("v_grad_abs_global", "v_bar_grad_abs_global"):
                    running[k].append(v_stats[k])
                running["v_grad_abs_layer"].append(v_stats["v_grad_abs_layer"])
                running["v_bar_grad_abs_layer"].append(v_stats["v_bar_grad_abs_layer"])

        epoch_losses[0] += loss
        epoch_losses[1] += loss2
        total_batches += 1

        y_cls = y.argmax(dim=1) if y.dim() > 1 else y
        with torch.no_grad():
            train_correct += (net1(X).argmax(1) == y_cls).sum().item()
            train_seen += y_cls.shape[0]

        accs = None
        if log_step or record_step:
            accs = minibatch_accuracy_net1(X, y, net1)

        if recorder is not None and recorder.enabled():
            batch_snap = _batch_snap(
                loss,
                loss2,
                layer_cos,
                layer_angles,
                g_cos,
                g_angle,
                v_stats,
                cross_stats,
                paired_cos_stats,
                paired_grad_stats,
                ep_stats=ep_stats,
                v_update_stats=v_update_stats,
            )
            recorder.on_batch(batch_snap, epoch, batch_idx + 1, accs)

        if log_step:
            snap = _aggregate_running(running)
            _print_intra_epoch_stats(
                epoch, batch_idx + 1, n_batches, snap, accs, symmetric_vv=sym_vv
            )
            running.clear()

    epoch_compare = {
        "layer_cosine_mean": [_mean_std(batch_layer_cos[i])[0] for i in range(num_layers)],
        "layer_cosine_std": [_mean_std(batch_layer_cos[i])[1] for i in range(num_layers)],
        "layer_angle_deg_mean": [_mean_std(batch_layer_angles[i])[0] for i in range(num_layers)],
        "layer_angle_deg_std": [_mean_std(batch_layer_angles[i])[1] for i in range(num_layers)],
        "global_cosine_mean": _mean_std(batch_global_cos)[0],
        "global_cosine_std": _mean_std(batch_global_cos)[1],
        "global_angle_deg_mean": _mean_std(batch_global_angles)[0],
        "global_angle_deg_std": _mean_std(batch_global_angles)[1],
    }

    epoch_v = {
        "v_grad_abs_layer_mean": [
            _mean_std([r[i] for r in epoch_v_buf["v_grad_abs_layer"] if len(r) > i])[0]
            for i in range(num_layers)
        ],
        "v_bar_grad_abs_layer_mean": [
            _mean_std([r[i] for r in epoch_v_buf["v_bar_grad_abs_layer"] if len(r) > i])[0]
            for i in range(num_layers)
        ],
        "v_grad_abs_global_mean": _mean_std(epoch_v_buf["v_grad_abs_global"])[0],
        "v_bar_grad_abs_global_mean": _mean_std(epoch_v_buf["v_bar_grad_abs_global"])[0],
        "v_cross_weight_diff_layer_mean": [
            _mean_std([r[i] for r in epoch_v_buf["v_cross_weight_diff_layer"] if len(r) > i])[0]
            for i in range(max(0, num_layers - 1))
        ],
        "v_cross_weight_diff_global_mean": _mean_std(epoch_v_buf["v_cross_weight_diff_global"])[0],
        "v_paired_grad_diff_layer_mean": [
            _mean_std([r[i] for r in epoch_v_buf["v_paired_grad_diff_layer"] if len(r) > i])[0]
            for i in range(max(0, num_layers - 1))
        ],
        "v_paired_grad_diff_global_mean": _mean_std(epoch_v_buf["v_paired_grad_diff_global"])[0],
        "v_paired_weight_cos_layer_mean": [
            _mean_std([r[i] for r in epoch_v_buf["v_paired_weight_cos_layer"] if len(r) > i])[0]
            for i in range(max(0, num_layers - 1))
        ],
        "v_paired_weight_cos_global_mean": _mean_std(epoch_v_buf["v_paired_weight_cos_global"])[0],
        "ep_pathway_norm_layer_mean": [
            _mean_std([r[i] for r in epoch_v_buf["ep_pathway_norm_layer"] if len(r) > i])[0]
            for i in range(num_layers)
        ],
        "ep_pathway_rel_layer_mean": [
            _mean_std([r[i] for r in epoch_v_buf["ep_pathway_rel_layer"] if len(r) > i])[0]
            for i in range(num_layers)
        ],
        "ep_pathway_norm_global_mean": _mean_std(epoch_v_buf["ep_pathway_norm_global"])[0],
        "ep_pathway_rel_global_mean": _mean_std(epoch_v_buf["ep_pathway_rel_global"])[0],
    }

    avg_losses = [
        epoch_losses[0] / max(total_batches, 1),
        epoch_losses[1] / max(total_batches, 1),
    ]
    train_acc = 100.0 * train_correct / max(train_seen, 1)
    return avg_losses, epoch_compare, epoch_v, train_acc


def eval_net1(dataloader, model, loss_fn, args, max_batches=0):
    """Full-dataloader Net1 accuracy (%) and mean cross-entropy loss (FF)."""
    total_loss = 0.0
    correct = 0
    seen = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for bi, (X, y) in enumerate(dataloader):
            if max_batches > 0 and bi >= max_batches:
                break
            X, y = X.to(args.device), y.to(args.device)
            out = model(X)
            bs = y.shape[0]
            total_loss += loss_fn(out, y).item() * bs
            correct += (out.argmax(1) == y).sum().item()
            seen += bs
    if was_training:
        model.train()
    seen = max(seen, 1)
    return 100.0 * correct / seen, total_loss / seen


def resolve_input_dim(dataset: str) -> int:
    if dataset == "cifar10":
        return 3072
    if dataset in ("mnist", "fashion_mnist"):
        return 784
    if dataset == "svhn":
        return 3072
    if dataset == "stl10":
        return 27648
    raise ValueError(f"Dataset {dataset} not supported")


def main():
    args = parse_run_args()
    os.makedirs(args.run_dir, exist_ok=True)
    set_seed(args.seed, more_determinism=args.more_determinism)

    input_dim = resolve_input_dim(args.dataset)
    out_dim = args.num_classes
    loss_fn = nn.CrossEntropyLoss()

    train_dataloader, test_dataloader = load_data(args.data_root, args.dataset, args.batch_size)

    norm_kw = dict(
        forward_norm_type=args.forward_norm_type,
        backward_norm_type=args.backward_norm_type,
    )
    forward_nl = resolve_nonlin(args.forward_nonlin)
    backward_nl = resolve_nonlin(args.backward_nonlin)
    net_kw = dict(
        in_d=input_dim,
        out_d=out_dim,
        hidden=args.hidden,
        num_hidden_layers=args.num_hidden_layers,
        forward_nonlin_fn=forward_nl,
        backward_nonlin_fn=backward_nl,
        **norm_kw,
    )
    net1 = Net_V(F=forward_nl, **net_kw).to(args.device)
    net2 = Net_V(F=forward_nl, backward=True, **net_kw).to(args.device)
    net1_bp = Net_V(F=forward_nl, **net_kw).to(args.device)
    copy_params_data(net1, net1_bp)

    print(
        f"Dataset: {args.dataset}, Hidden: {args.hidden}, "
        f"num_hidden_layers: {args.num_hidden_layers}, Epochs: {args.epochs}, LR: {args.lr}"
    )
    print(
        f"Model: Net_V ({args.num_hidden_layers} hidden + output = "
        f"{args.num_hidden_layers + 1} W layers) | SAL vs BP on W; "
        "stats for V (Net1.v) and V_bar (Net2.v)"
    )
    print(f"Device: {args.device} | Data: {args.data_root} | Run: {args.run_dir}")
    if args.more_determinism:
        print(
            "more_determinism=True: cuda.manual_seed_all, cudnn.deterministic, "
            "cudnn.benchmark=False, TF32 disabled"
        )
    print(f"log_every={args.log_every} (0=intra-epoch logging off)")
    print(f"record_every={args.record_every} (0=stat_timeseries.json off)")
    print("Intra-epoch 'minibatch acc' (log/record) = current train batch only.")
    print("Full test-set accuracy = banner block printed once per epoch below.")
    print(
        f"weight_decay(W)={args.weight_decay} weight_decay(V)={args.weight_decay_v} "
        f"weight_decay(V_bar)={args.weight_decay_v_bar}"
    )
    print(
        f"forward_norm_type={args.forward_norm_type} "
        f"backward_norm_type={args.backward_norm_type} (non-parametric, pre-nonlinearity)"
    )
    print(
        f"forward_nonlin={args.forward_nonlin} "
        f"backward_nonlin={args.backward_nonlin}"
    )
    sal_update_fn = get_sal_update_fn(args)
    print(f"sal_update={args.sal_update} ({sal_update_fn.__name__})")
    if args.force_backprop:
        print("force_backprop=True: Net1 optimizer step uses Net1_bp gradients")
    if getattr(args, "trace_v_updates", False):
        print(
            "trace_v_updates=True: pre-lr |grad|, |wd*p|, |grad+wd*p| on V/V_bar pre-momentum "
            "(SGD; step still applies lr and momentum)"
        )
    print("-" * 72)

    recorder = StatsRecorder(
        args.record_every, trace_v_updates=getattr(args, "trace_v_updates", False)
    )
    losses, test_losses, accuracies1, train_accuracies1 = [], [], [], []
    compare_history = {
        "num_layers": len(net1.layers),
        "epochs": [],
        "layer_cosine_mean": [],
        "layer_cosine_std": [],
        "layer_angle_deg_mean": [],
        "layer_angle_deg_std": [],
        "global_cosine_mean": [],
        "global_cosine_std": [],
        "global_angle_deg_mean": [],
        "global_angle_deg_std": [],
    }
    v_stats_history = []

    for t in tqdm(range(args.epochs), desc="Training"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        loss_epoch, epoch_cmp, epoch_v, train_acc_epoch = train_epoch_with_bp_compare(
            train_dataloader,
            net1,
            net2,
            net1_bp,
            loss_fn,
            args,
            test_dataloader=test_dataloader,
            epoch=t + 1,
            recorder=recorder,
            sal_update_fn=sal_update_fn,
        )
        losses.append(loss_epoch)
        train_accuracies1.append(train_acc_epoch)
        v_stats_history.append(epoch_v)
        for key in compare_history:
            if key == "num_layers":
                continue
            if key == "epochs":
                compare_history[key].append(t + 1)
            else:
                compare_history[key].append(epoch_cmp[key])

        acc1, test_loss_ff = eval_net1(test_dataloader, net1, loss_fn, args)
        accuracies1.append(acc1)
        test_losses.append(test_loss_ff)

        _print_epoch_summary(
            t + 1,
            args.epochs,
            acc1,
            loss_epoch[0],
            loss_epoch[1],
            test_loss_ff,
            train_acc1=train_acc_epoch,
            epoch_cmp=epoch_cmp,
            epoch_v=epoch_v,
        )

    print()
    print("=" * 72)
    print("  TRAINING COMPLETE  |  NET1 (SAL) FINAL (full test set)")
    print(f"    Test accuracy  {accuracies1[-1]:7.2f}%")
    print(f"    Test loss FF   {test_losses[-1]:7.4f}")
    print("=" * 72)
    print()

    epoch_metrics = build_epoch_metrics(
        losses, test_losses, accuracies1, train_accuracies1=train_accuracies1
    )

    ts_path = os.path.join(args.run_dir, "stat_timeseries.json")
    if recorder.enabled():
        copy_params_data(net1, net1_bp)
        recorder.finalize()
    ts_payload = (
        recorder.build_save_payload(len(net1.layers), args.record_every, epoch_metrics)
        if recorder.enabled()
        else {
            "num_layers": len(net1.layers),
            "record_every": 0,
            "epoch_metrics": epoch_metrics,
            "series": {"step": []},
        }
    )
    with open(ts_path, "w") as f:
        json.dump(ts_payload, f, indent=2)
    plot_cmd = format_plot_command(args)
    n_pts = len(recorder.data["step"]) if recorder.enabled() else 0
    print(f"Stat JSON ({n_pts} intra-epoch points, {len(epoch_metrics['epoch'])} epochs) -> {ts_path}")
    print("Plot figure:")
    print(f"  {plot_cmd}")

    summary_path = os.path.join(args.run_dir, "sal_bp_compare_run.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "config": args_to_jsonable(args),
                "code_names": {
                    "W": "Net1.layer.w",
                    "W_prime": "Net2.layer.w",
                    "V": "Net1.layer.v (upstream cross)",
                    "V_bar": "Net2.layer.v (downstream cross)",
                    "v_cross": "junction paired: Net1.layers[i+1].v vs Net2.layers[i].v (no .T if symmetric_vv)",
                },
                "epoch_metrics": epoch_metrics,
                "losses": {
                    "feedforward": epoch_metrics["train_loss_ff"],
                    "gradient": epoch_metrics["train_loss_gn"],
                },
                "accuracies": {
                    "net1_sal": epoch_metrics["test_acc_net1"],
                    "net1_sal_train": epoch_metrics.get("train_acc_net1", []),
                },
                "test_losses": {"feedforward": epoch_metrics["test_loss_ff"]},
                "compare_sal_vs_bp_forward_w": compare_history,
                "v_and_cross_stats": v_stats_history,
                "stat_timeseries_path": ts_path,
                "plot_command": plot_cmd,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {summary_path}")


if __name__ == "__main__":
    main()