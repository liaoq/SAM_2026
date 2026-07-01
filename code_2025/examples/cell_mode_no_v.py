import torch
import torch.nn as nn

class Linear(nn.Module):

    def __init__(self, in_dim, out, bias=False):
        super(Linear, self).__init__()

        self.w = nn.Linear(in_dim, out, bias=False)

    def forward(self, x):
        self.ha = x
        if x.requires_grad:
            self.ha.retain_grad()

        x = self.w(x)
        self.hb = x
        self.hb.retain_grad()
        return x

    def backproj_nobias(self,y):
        # assume no bias

        self.hb.grad=y
        #self.hb.retain_grad()

        x = torch.mm(y,self.w.weight)

        self.ha.grad=x
        #self.ha.retain_grad()

        return x

def update_grad(Net1, Net2, X, Y, loss_fn):
    x_masks=[]
    output = Net1(X,x_masks=x_masks)
    output.retain_grad()
    loss = loss_fn(output, Y)
    loss.backward()
    # Just to make sure we aren't getting gradients
    Net1.zero_grad()
    Net2.zero_grad()

    O = output.grad.detach()
    O2 = Net2(X,x_masks=x_masks)

    loss_2_true = loss_fn(O2, Y)

    Net2.backproj(O,x_masks=x_masks,nonlin=None)

    for t, s in zip(Net1.modules(), Net2.modules()):
        if isinstance(t, Linear):
            h = t.ha
            gh = s.hb.grad

            grad = gh.T.mm(h)

            t.w.weight.grad = grad
            s.w.weight.grad = grad

    return loss.item(), loss_2_true.item()

