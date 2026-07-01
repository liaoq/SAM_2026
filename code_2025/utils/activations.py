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
