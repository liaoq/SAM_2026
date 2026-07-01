import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Multi-dataset Training')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='mnist', 
                    choices=['mnist', 'cifar10', 'pathmnist', 'fashion_mnist', 
                            'svhn', 'stl10', 'caltech101', 'oxford_iiit_pet', 
                            'flowers102', 'eurosat', 'symbolic_regression'],
                    help='dataset to use')
    parser.add_argument('--root', type=str, default='data/', 
                    help='root dir to datasets')
    parser.add_argument('--num_classes', type=int, default=10, 
                    help='number of classes in the dataset')
    
    # Model parameters
    parser.add_argument('--hidden', type=int, default=200, 
                    help='number of hidden units in the model')
    parser.add_argument('--activation', type=str, default='sigmoid', 
                    help='activation function to use')
    parser.add_argument('--init_as_gd', action='store_true',
                    help='Initialize v as identity matrix instead of using kaiming uniform')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256, 
                    help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=10, 
                    help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.5e-2, 
                    help='learning rate (default: 0.5e-2)')
    parser.add_argument('--weight_decay', type=float, default=0.001, 
                    help='weight decay (default: 0.001)')
    
    # Learning rate scaling factors
    parser.add_argument('--lr_w', type=float, default=1.0, 
                    help='learning rate scaling factor for w')
    parser.add_argument('--lr_w_prime', type=float, default=0.2, 
                    help='learning rate scaling factor for w prime')
    parser.add_argument('--lr_v', type=float, default=1.0, 
                    help='learning rate scaling factor for v')
    parser.add_argument('--lr_v_prime', type=float, default=4.0, 
                    help='learning rate scaling factor for v prime')
    
    # Utility parameters
    parser.add_argument('--seed', type=int, default=0, 
                    help='random seed (default: 0)')
    parser.add_argument('--device', type=str, 
                    default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='device to run the model on')
    parser.add_argument('--compare_activations', action='store_true',
                    help='Compare different activation functions')
    parser.add_argument('--grid_search', action='store_true',
                    help='perform grid search for learning rates')
    parser.add_argument('--w_bits', type=int, default=32,
                    help='number of bits for weight quantization')
    parser.add_argument('--a_bits', type=int, default=32,
                    help='number of bits for activation quantization')
    
    args = parser.parse_args()
    return args 