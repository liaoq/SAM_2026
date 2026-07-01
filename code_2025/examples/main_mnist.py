import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.optim.optimizer import Optimizer, required
## two layer neural network
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

from examples.cell_model import Linear, Net, F_relu, F_identity

model = Net()



def update_grad_old_autograd(Net1, Net2, X, Y):

          x_masks=[]
          output = Net1(X,x_masks=x_masks)
          output.retain_grad()
          loss = loss_fn(output, Y)
          loss.backward()

          # print(len(x_masks))
          O = output.grad.detach()
          #print(O.shape)
          O2 = Net2(X,x_masks=x_masks)
          loss2 = torch.trace(O.mm(O2.T))
          loss2.backward()
          loss_2_true = loss_fn(O2, Y)
          for t, s in zip(Net1.modules(), Net2.modules()):
            #print(t, s)
            if isinstance(t, Linear):
              #print(1)
              h = t.ha
              gh = s.hb.grad
              #print(h, gh)

              grad = gh.T.mm(h)

              t.w.weight.grad = grad
              #t.w.grad += grad
              s.w.weight.grad = grad

          return loss.item(), loss_2_true.item()


def update_grad(Net1, Net2, X, Y):

          x_masks=[]
          output = Net1(X,x_masks=x_masks)
          output.retain_grad()
          loss = loss_fn(output, Y)
          loss.backward()

          # print(len(x_masks))
          O = output.grad.detach()
          #print(O.shape)
          O2 = Net2(X,x_masks=x_masks)
          loss2 = torch.trace(O.mm(O2.T))
          # loss2.backward()
          loss_2_true = loss_fn(O2, Y)

          In2_backproj = Net2.backproj(O,x_masks=x_masks,nonlin=None)

          for t, s in zip(Net1.modules(), Net2.modules()):
            #print(t, s)
            if isinstance(t, Linear):
              #print(1)
              h = t.ha
              gh = s.hb.grad
              #print(h, gh)

              grad = gh.T.mm(h)

              t.w.weight.grad = grad
              #t.w.grad += grad
              s.w.weight.grad = grad

          return loss.item(), loss_2_true.item()

def mse(Z,Y):
  return ((Z - Y) **2).sum(-1).mean()

noise_range = [0.01]#np.arange(0.1, 5, 0.8)
matrices = []

d = 64

losses = []

loss_fn = mse

for noise in noise_range:

    for lr in [0.005]:
        torch.manual_seed(10)
        print(noise, lr)
        #

        model1 = Net(width=d, depth=1, in_d=d,out_d=d).cuda()
        model2 = Net(width=d, depth=1, in_d=d,out_d=d,F=F_identity).cuda()

        #opt = optim.Adam(model1.parameters(), lr=1e-3, weight_decay=1e-)
        #opt = optim.Adam(model2.parameters(), lr=1e-3, weight_decay=1e-3)
        opt1 = optim.SGD(model1.parameters(), lr=0.1e-1, weight_decay=0.15e-1)#.3e-1)
        opt2 = optim.SGD(model2.parameters(), lr=0.1e-1, weight_decay=0.15e-1)#.3e-1)

        STEP = 10000
        #criterion = F.mse_loss
        #sum_loss = 0

        W = torch.zeros(d, d).cuda().normal_()
        W[d//2:] *= 0

        for i in range(STEP):


          N = 100
          X = torch.zeros([N, d]).normal_(0,1).cuda()#* torch.Tensor(np.arange(0.01, 2.01, 2 / d)).cuda()

          eps = noise * torch.zeros([N, d]).normal_(0,1).cuda()#* torch.Tensor(np.arange(0.01, 2.01, 2 / d)).cuda()
          #eps[2:] = 0


          Y =  X + eps.mm(W) #* mask #.mean(dim=1, keepdim=True) #+ eps
          #print(X.shape, Y.shape)

          opt1.zero_grad()
          opt2.zero_grad()




          #print(output.shape, Y.shape)
          loss, loss_2_true = update_grad(model1, model2, X, Y)


          if False:
            loss = ((output - Y) **2).sum(-1).mean()
            loss.backward()

            O = output.grad.detach()
            #print(O.shape)
            O2 = model2(X)
            loss2 = torch.trace(O.mm(O2.T))
            loss2.backward()
            for t, s in zip(model1.modules(), model2.modules()):
              #print(t, s)
              if isinstance(t, Linear):
                #print(1)
                h = t.ha
                gh = s.hb.grad
                #print(h, gh)

                grad = gh.T.mm(h)

                t.w.weight.grad = grad
                #t.w.grad += grad
                s.w.weight.grad = grad
          torch.nn.utils.clip_grad_norm_(model2.parameters(), 10)
          torch.nn.utils.clip_grad_norm_(model1.parameters(), 10)


          #for t, s in zip(model1.modules(), model2.modules()):
            #print(t, s)
          #    if isinstance(t, Linear):
           #     print(t.w.weight.grad.norm())
          #      print(s.w.weight.grad.norm())
              #t.w.grad += grad
              #s.w.weight.grad = grad

          #print(loss.item())

          #print(O2.shape, O.shape)
          losses.append([loss, loss_2_true])

          if i % 30 ==0:
            print(loss, loss_2_true)

          opt1.step()
          opt2.step()
            #with torch.no_grad():

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 256

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

#loss_fn = nn.MSELoss()
device = 'cuda'
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def converter(indices):
    probvec = torch.zeros(len(indices), 10)
    for number, index in enumerate(indices):
      probvec[number][index] = 1.0
    return probvec

def train(dataloader, model1, model2, loss_fn, opt1, opt2):
    global losses
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        y = converter(y)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        #pred = model(X)
        #loss = loss_fn(pred, y)
        opt1.zero_grad()
        opt2.zero_grad()
        loss, loss2 = update_grad(model1, model2, X, y)

        # Backpropagation

        opt1.step()
        opt2.step()

        if batch % 50 == 0:
            #loss, current = loss.item(), batch * len(X)
            print(loss, loss2)
        losses.append([loss, loss2])

def test(dataloader, model, loss_fn,nonlin=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    #model.eval()
    test_loss, correct = 0, 0
    #with torch.no_grad():
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X,nonlin=nonlin)
        #test_loss += loss_fn(pred, converter(y)).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 10
## model1 is the feedforward, model2 is the gradient net

model1 = Net(in_d=784,out_d=10).cuda()
model2 = Net(in_d=784,out_d=10,F=F_identity).cuda()
opt1 = optim.SGD(model1.parameters(), lr=0.5e-1, weight_decay=0.001, momentum=0.9)#.3e-1)
opt2 = optim.SGD(model2.parameters(), lr=0.5e-1, weight_decay=0.001, momentum=0.9)#.3e-1)

losses =[]

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model1, model2, loss_fn, opt1, opt2)
    test(test_dataloader, model1, loss_fn)
    test(test_dataloader, model2, loss_fn, nonlin=F_relu)

print("Done!")