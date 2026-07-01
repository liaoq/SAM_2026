import torch
import torch.nn as nn
import torch.optim as optim
from examples.cell_model import QuantizedNet_V, Net_V
from utils.data_factory import load_data
from utils.activations import F_relu

def train_standard_sgd(dataloader, model, loss_fn, optimizer, device):

    model.train()
    total_loss = 0.0
    total_samples = 0

    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, Y)
        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    average_loss = total_loss / total_samples
    return average_loss

def evaluate_standard(dataloader, model, loss_fn, device):

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, Y)
            total_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == Y).sum().item()
            total += X.size(0)

    average_loss = total_loss / total
    accuracy = correct / total
    return average_loss, accuracy

def baseline_training(args):

    train_loader, test_loader = load_data(args.root, args.dataset, args.batch_size)

    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        input_dim = 784
    elif args.dataset == 'cifar10' or args.dataset == 'svhn':
        input_dim = 3072
    else:
        raise ValueError(f"Dataset {args.dataset} not supported for baseline.")
    
    out_dim = args.num_classes

    model = QuantizedNet_V(in_d=input_dim, out_d=out_dim, hidden=args.hidden, F=F_relu, w_bits=args.w_bits, a_bits=args.a_bits).to(args.device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    training_losses = []
    test_accuracies = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_standard_sgd(train_loader, model, loss_fn, optimizer, args.device)
        test_loss, test_acc = evaluate_standard(test_loader, model, loss_fn, args.device)

        training_losses.append(train_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch}/{args.epochs}: Train Loss = {train_loss:.4f}, "
              f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc*100:.2f}%")

    return training_losses, test_accuracies, model

if __name__ == "__main__":
    from config import get_args
    args = get_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_losses, test_accuracies, model = baseline_training(args)
