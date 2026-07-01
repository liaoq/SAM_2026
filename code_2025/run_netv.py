import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from examples.cell_model import Net_V, Linear_V, update_grad_v_no_mask2
from utils.data_factory import load_data
from utils.activations import F_relu, F_identity
from config import get_args


def converter(indices, num_classes):
    probvec = torch.zeros(len(indices), num_classes)
    probvec[range(len(indices)), indices] = 1.0
    return probvec


def train_epoch(dataloader, model1, model2, loss_fn, args, weight_decay=0.001):
    opt1 = optim.SGD(model1.parameters(), lr=args.lr, weight_decay=weight_decay, momentum=0.9)
    opt2 = optim.SGD(model2.parameters(), lr=args.lr, weight_decay=weight_decay, momentum=0.9)
    epoch_losses = [0, 0]
    total_batches = 0
    model1.train()
    model2.train()

    for batch, (X, y) in enumerate(dataloader):
        y = converter(y, args.num_classes).to(args.device)
        X, y = X.to(args.device), y.to(args.device)
        opt1.zero_grad()
        opt2.zero_grad()
        loss, loss2 = update_grad_v_no_mask2(model1, model2, X, y, loss_fn, args)

        parameters = list(model1.parameters()) + list(model2.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, 50)

        opt1.step()
        opt2.step()

        epoch_losses[0] += loss
        epoch_losses[1] += loss2
        total_batches += 1

    return [epoch_losses[0] / total_batches, epoch_losses[1] / total_batches]


def test_model(dataloader, model, loss_fn, args, nonlin=None):
    size = len(dataloader.dataset)
    correct = 0
    for X, y in dataloader:
        X, y = X.to(args.device), y.to(args.device)
        pred = model(X, nonlin=nonlin)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    accuracy = 100 * correct / size
    return accuracy


def main():
    args = get_args()

    if args.dataset == 'cifar10':
        input_dim = 3072
    elif args.dataset in ('mnist', 'fashion_mnist'):
        input_dim = 784
    elif args.dataset == 'svhn':
        input_dim = 3072
    elif args.dataset == 'stl10':
        input_dim = 27648
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    out_dim = args.num_classes
    loss_fn = nn.CrossEntropyLoss()

    train_dataloader, test_dataloader = load_data(args.root, args.dataset, args.batch_size)

    model1 = Net_V(in_d=input_dim, out_d=out_dim, hidden=args.hidden, F=F_relu).to(args.device)
    model2 = Net_V(in_d=input_dim, out_d=out_dim, hidden=args.hidden, F=F_identity, backward=True).to(args.device)

    print(f"Dataset: {args.dataset}, Hidden: {args.hidden}, Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Model: Net_V (non-quantized, using Linear_V with v matrices)")
    print(f"Device: {args.device}")
    print(f"lr_w={args.lr_w}, lr_w_prime={args.lr_w_prime}, lr_v={args.lr_v}, lr_v_prime={args.lr_v_prime}")
    print("-" * 60)

    for t in tqdm(range(args.epochs), desc="Training"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        loss = train_epoch(train_dataloader, model1, model2, loss_fn, args=args)

        acc1 = test_model(test_dataloader, model1, loss_fn, args=args)
        acc2 = test_model(test_dataloader, model2, loss_fn, args=args, nonlin=F_relu)

        print(f"Epoch {t+1}/{args.epochs} | Loss FF={loss[0]:.4f} GN={loss[1]:.4f} | "
              f"Acc FF={acc1:.2f}% GN={acc2:.2f}%")

    print("=" * 60)
    print(f"Final: Feedforward Net Acc = {acc1:.2f}%, Gradient Net Acc = {acc2:.2f}%")


if __name__ == "__main__":
    main()
