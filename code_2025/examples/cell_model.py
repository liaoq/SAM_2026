import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.activations import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.activations import *

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

class QuantizedNet_V(nn.Module):
    def __init__(self, in_d=5, out_d=5, F=F.relu, hidden=100, backward_more_dim=0,
                 dropout_rate=[0, 0], backward=False, init_as_gd=False,
                 w_bits=32, a_bits=32):

        super(QuantizedNet_V, self).__init__()

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
            if i > 0:  # 除了最后一层，其它层之后可加激活与 dropout
                if nonlin:
                    x = nonlin(x)
                x = self.dropout_back(x)
        return x

    def forward(self, x, nonlin=None):
        used_nonlin = nonlin if nonlin else self.F
        x = x.view(x.shape[0], -1)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
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
    t - layer in Net1 (Feedforward Net)
    s - layer in Net2 (Gradient Net)
    '''
    tha = t.ha
    shb = s.hb.grad
    sha = shb.mm(s.w.weight)
    thb = tha.mm(t.w.weight.T)
    shbtha = shb.T.mm(tha)
    
    if s.last_layer:
        twgrad = s.hb.grad.T.mm(tha)
    else:
        # s.v.weight.shape: torch.Size([200, 200])
        gthb = shb.mm(s.v.weight) # Apply v here
        twgrad = (gthb * (thb > 0)).T.mm(tha)

    t.w.weight.grad = twgrad * args.lr_w

    if not s.last_layer:
        svgrad = shbtha.mm(t.w.weight.T) * args.lr_v_prime  # scaling factor
        ### added
        # svgrad = (shbtha * (thb > 0)).mm(t.w.weight.T) * args.lr_v_prime
        ####
        
        s.v.weight.grad = svgrad

    if not s.first_layer:
        # t.v.weight.shape: torch.Size([200, 200])
        swgrad = shbtha.mm(t.v.weight.T) * args.lr_w_prime  # scaling factor ## Apply v here
        tvgrad = s.w.weight.T.mm(shbtha) * args.lr_v
        s.w.weight.grad = swgrad
        t.v.weight.grad = tvgrad

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

class Net_V(nn.Module):
    def __init__(self, in_d=5, out_d=5, F=F.relu, hidden=100, backward_more_dim=0,
                 dropout_rate=[0,0], backward=False, init_as_gd=False):
        super(Net_V, self).__init__()

        if backward:
            hidden += backward_more_dim
        
        self.layers = nn.ModuleList()
        
        layer_sizes = [in_d] + [hidden] * 3 + [out_d]
            
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
            if i > 0:  # Don't apply nonlinearity or dropout after the first layer (last in backprop)
                if nonlin:
                    x = nonlin(x)
                x = self.dropout_back(x)
        return x

    def forward(self, x, nonlin=None):
        used_nonlin = nonlin if nonlin else self.F
        x = x.view(x.shape[0], -1)

        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = used_nonlin(x)
            x = self.dropout_forward(x)

        x = self.layers[-1](x)
        return x
