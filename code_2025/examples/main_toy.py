import numpy as np
import matplotlib.pyplot as plt
import torch
from cell_model import Linear_V, F_relu, F_identity, update_grad_v_no_mask2, Net_V_Single
import torch.optim as optim

def mse(Z,Y):
    return ((Z - Y) **2).sum(-1).mean()


dropout_rate = [0,0]

noise_range = [0.01]#np.arange(0.1, 5, 0.8)
matrices = []

d = 64

losses = []

loss_fn = mse

for noise in noise_range:

    for lr in [0.005]:
        torch.manual_seed(10)
        print(noise, lr)
        hids =100
        backward_more_dim=0
        model1 = Net_V_Single(width=d, depth=1, in_d=d,hidden=hids,backward_more_dim=backward_more_dim,out_d=d,
                     F=F_relu,dropout_rate=dropout_rate).cuda()
        model2 = Net_V_Single(width=d, depth=1, in_d=d,hidden=hids,backward_more_dim=backward_more_dim,out_d=d,
                     F=F_identity,dropout_rate=dropout_rate,backward=True).cuda()

        opt1 = optim.SGD(model1.parameters(), lr=0.1e-2, weight_decay=0.03e-0)
        opt2 = optim.SGD(model2.parameters(), lr=0.1e-2, weight_decay=0.03e-0)

        STEP = 10000

        W = torch.zeros(d, d).cuda().normal_()
        W[d//2:] *= 0

        for i in range(STEP):
          N = 256
          X = torch.zeros([N, d]).normal_(0,1).cuda()

          eps = noise * torch.zeros([N, d]).normal_(0,1).cuda()

          Y =  X + eps.mm(W)

          opt1.zero_grad()
          opt2.zero_grad()

          update_algorithm = update_grad_v_no_mask2
          loss, loss_2_true = update_algorithm(model1, model2, X, Y, loss_fn)

          clip_by_val= 20
          clip_by_norm = 30
          for model in [model1, model2]:
            for param in model.parameters():
                if param.grad is not None:
                    torch.nn.utils.clip_grad_value_(param, clip_by_val)
                torch.nn.utils.clip_grad_norm_(param, clip_by_norm, norm_type=2)

          losses.append([loss, loss_2_true])

          if i % 30 ==0:
            print(loss, loss_2_true)

          opt1.step()
          opt2.step()

losses = np.array(losses)
plt.plot(losses[:,0], alpha=0.7, lw=1, label='feedforward net')
plt.plot(losses[:,1], alpha=0.7, lw=1, label='gradient net')
plt.ylabel('objective')
plt.xlabel('iteration')
plt.legend()
dropout_str = '_'.join(str(x) for x in dropout_rate)
plt.savefig(update_algorithm.__name__+ '_regression_dropout_' + dropout_str + '.pdf', bbox_inches='tight')