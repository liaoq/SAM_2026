import torch.optim as optim
from .data_factory import load_data
from utils import converter
from examples.cell_model import update_grad_v_no_mask2
import torch
from medmnist import Evaluator

def train(dataloader, model1, model2, loss_fn, update_algorithm, args, weight_decay=0.001):
    opt1 = optim.SGD(model1.parameters(), lr=args.lr, weight_decay=weight_decay, momentum=0.9)
    opt2 = optim.SGD(model2.parameters(), lr=args.lr, weight_decay=weight_decay, momentum=0.9)
    epoch_losses = [0, 0]
    total_batches = 0
    model1.train(), model2.train()

    for batch, (X, y) in enumerate(dataloader):
        y = converter(y, args.num_classes).to(args.device)
        X, y = X.to(args.device), y.to(args.device)
        opt1.zero_grad(), opt2.zero_grad()
        loss, loss2 = update_algorithm(model1, model2, X, y, loss_fn, args)

        if update_algorithm.__name__ == update_grad_v_no_mask2.__name__:
            parameters = list(model1.parameters()) + list(model2.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, 50)
                
        opt1.step(), opt2.step()
        if batch % 200 == 0: print(loss, loss2)
            
        epoch_losses[0] += loss
        epoch_losses[1] += loss2
        total_batches += 1
            
    return [epoch_losses[0]/total_batches, epoch_losses[1]/total_batches]

def test_medmnist(model, data_flag, task, test_loader):
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true.to('cpu'), targets.to('cpu')), 0)
            y_score = torch.cat((y_score.to('cpu'), outputs.to('cpu')), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        
        evaluator = Evaluator(data_flag, 'test')
        metrics = evaluator.evaluate(y_score)
        print(f"Test Error: \n Accuracy: {metrics[1]:>0.1f}%")
    
        return metrics[1]

def test(dataloader, model, loss_fn,nonlin=None, args=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    #model.eval()
    test_loss, correct = 0, 0
    #with torch.no_grad():
    for X, y in dataloader:
        X, y = X.to(args.device), y.to(args.device)
        pred = model(X,nonlin=nonlin)
        #test_loss += loss_fn(pred, converter(y)).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    accuracy = 100*correct
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    result = accuracy
    return result
