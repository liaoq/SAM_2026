import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import json
from examples.cell_model import Net_V, update_grad_v_no_mask2, QuantizedNet_V
from utils.data_factory import load_data
from utils.train import train, test, test_medmnist
from tqdm import tqdm
from utils.get_v_w_visualize import extract_w, extract_v
import torch.nn.functional as F
from utils.activations import *
from config import get_args
from matplotlib.colors import LinearSegmentedColormap

    
def save_results(losses, accuracies1, accuracies2, args):
    results = {
        'config': vars(args),
        'training': {
            'losses': {
                'feedforward': [loss[0] for loss in losses],
                'gradient': [loss[1] for loss in losses]
            },
            'iterations': len(losses)
        },
        'evaluation': {
            'accuracies': {
                'feedforward': accuracies1,
                'gradient': accuracies2
            },
            'epochs': len(accuracies1)
        }
    }
    
    filename = f'results_{args.dataset}_epochs{args.epochs}_lr{args.lr}_hidden{args.hidden}_w_bits{args.w_bits}_a_bits{args.a_bits}.json'
    with open(os.path.join('logs', filename), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

def save_accuracies(accuracies1, accuracies2, args):
    """Save accuracies to a JSON file"""
    accuracies_data = {
        'config': vars(args),
        'accuracies': {
            'feedforward': accuracies1,
            'gradient': accuracies2
        }
    }
    
    filename = f'accuracies_{args.dataset}_epochs{args.epochs}_lr{args.lr}_hidden{args.hidden}.json'
    filepath = os.path.join('logs', filename)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(accuracies_data, f, indent=2)
    print(f"Accuracies saved to {filepath}")

def main(args):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    train_dataloader, test_dataloader = load_data(args.root, args.dataset, args.batch_size)
    loss_fn = nn.CrossEntropyLoss()
    
    out_dim = args.num_classes
    
    activation_fn = activation_functions[args.activation]
    model1 = QuantizedNet_V(in_d=input_dim, out_d=out_dim, hidden=args.hidden, F=F_relu, w_bits=args.w_bits, a_bits=args.a_bits).to(args.device)
    model2 = QuantizedNet_V(in_d=input_dim, out_d=out_dim, hidden=args.hidden, F=F_identity, backward=True, w_bits=args.w_bits, a_bits=args.a_bits).to(args.device)

    # visualize_gram_matrices(model1, model2, args, suffix='init')

    losses = []
    accuracies1, accuracies2 = [], []
    epoch_data = {
        'w': {'start': extract_w(model1, model2)[0]},
        'v': {'start': extract_v(model1, model2)[0]},
    }
    
    for t in tqdm(range(args.epochs), desc="Training Progress"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        loss = train(train_dataloader, model1, model2, loss_fn, update_algorithm=update_grad_v_no_mask2, args=args)
        
        losses.append(loss)
        if args.dataset == 'pathmnist':
            accuracies1.append(test_medmnist(model1, args.dataset, 'multi-label', test_dataloader))
            accuracies2.append(test_medmnist(model2, args.dataset, 'multi-label', test_dataloader))
        else:
            accuracies1.append(test(test_dataloader, model1, loss_fn, args=args))
            accuracies2.append(test(test_dataloader, model2, loss_fn, args=args, nonlin=F_relu))
        
        # Save weights at the end of each epoch
        epoch_data['w'][str(t + 1)] = extract_w(model1, model2)[0]
        epoch_data['v'][str(t + 1)] = extract_v(model1, model2)[0]
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 可视化训练后的 Gram matrices
    # visualize_gram_matrices(model1, model2, args, suffix='final')

    return losses, accuracies1, accuracies2, model1, model2, epoch_data

def plot_activation_comparison(results, args):
    plt.figure(figsize=(12, 8))
    for act_name, accuracies in results.items():
        plt.plot(range(1, args.epochs + 1), accuracies, label=act_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title(f'Activation Function Comparison ({args.dataset})')
    plt.legend()
    plt.grid(True)
    filename = f'{args.dataset}_activation_comparison.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    # 清理matplotlib缓存
    plt.clf()
    plt.close('all')

def plot_loss_acc(losses, accuracies1, accuracies2, args):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    losses_feedforward = [loss[0] for loss in losses]
    losses_gradient = [loss[1] for loss in losses]
    plt.plot(losses_feedforward, alpha=0.7, lw=1, label='Feedforward net')
    plt.plot(losses_gradient, alpha=0.7, lw=1, label='Gradient net')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(accuracies1, alpha=0.7, lw=1, label='Feedforward net')
    plt.plot(accuracies2, alpha=0.7, lw=1, label='Gradient net')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    filename = f'{args.dataset}_epochs{args.epochs}_lr{args.lr}_hidden{args.hidden}_Relu.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    plt.clf()
    plt.close('all')

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

def compute_cosine_similarity(tensor1, tensor2):
    # Flatten tensors and ensure they're on CPU
    flat1 = tensor1.flatten() if isinstance(tensor1, torch.Tensor) else torch.from_numpy(tensor1).flatten()
    flat2 = tensor2.flatten() if isinstance(tensor2, torch.Tensor) else torch.from_numpy(tensor2).flatten()
    # Compute cosine similarity
    return F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)).item()

def compare_models_similarity(args):
    # Add debug prints to check epoch_data structure
    losses, accuracies1, accuracies2, model1, model2, epoch_data = main(args)
    
    print("Available keys in epoch_data:", epoch_data.keys())
    print("Available epochs in w:", epoch_data['w'].keys())
    print("Available epochs in ww:", epoch_data['ww'].keys())
    
    # Initialize lists to store similarities for all weight matrices
    similarities = {
        'w1': [],    # model1's w
        'w2': [],    # model2's w (w')
        'v1': [],    # model1's v
        'v2': [],    # model2's v (v')
        'ww1': [],   # model1's w with V=I initialization
        'ww2': [],   # model2's w with V=I initialization
        'vv1': [],   # model1's v with V=I initialization
        'vv2': []    # model2's v with V=I initialization
    }
    
    # Get initial weights (epoch 0)
    w1_initial = epoch_data['w']['start'][0]
    w2_initial = epoch_data['w']['start'][1]
    v1_initial = epoch_data['v']['start'][0]
    v2_initial = epoch_data['v']['start'][1]
    ww1_initial = epoch_data['ww']['start'][0]
    ww2_initial = epoch_data['ww']['start'][1]
    vv1_initial = epoch_data['vv']['start'][0]
    vv2_initial = epoch_data['vv']['start'][1]
    
    # Convert initial weights to tensors
    initial_weights = {
        'w1': torch.cat([torch.from_numpy(np.array(w)).flatten() for w in w1_initial]),
        'w2': torch.cat([torch.from_numpy(np.array(w)).flatten() for w in w2_initial]),
        'v1': torch.cat([torch.from_numpy(np.array(v)).flatten() for v in v1_initial]),
        'v2': torch.cat([torch.from_numpy(np.array(v)).flatten() for v in v2_initial]),
        'ww1': torch.cat([torch.from_numpy(np.array(w)).flatten() for w in ww1_initial]),
        'ww2': torch.cat([torch.from_numpy(np.array(w)).flatten() for w in ww2_initial]),
        'vv1': torch.cat([torch.from_numpy(np.array(v)).flatten() for v in vv1_initial]),
        'vv2': torch.cat([torch.from_numpy(np.array(v)).flatten() for v in vv2_initial])
    }
    
    # Compare weights at each epoch with initial weights
    epochs = list(range(1, args.epochs + 1))  # Add epochs list definition
    
    for epoch in range(args.epochs):
        epoch_str = str(epoch + 1)
        if epoch_str in epoch_data['w'] and epoch_str in epoch_data['ww']:
            # Extract current weights for all types
            current_weights = {}
            
            # Extract w and v weights
            w_current = epoch_data['w'][epoch_str]
            v_current = epoch_data['v'][epoch_str]
            ww_current = epoch_data['ww'][epoch_str]
            vv_current = epoch_data['vv'][epoch_str]
            
            # Add debug prints
            print(f"\nEpoch {epoch_str} data shapes:")
            print(f"w_current shape: {len(w_current)}")
            print(f"v_current shape: {len(v_current)}")
            print(f"ww_current shape: {len(ww_current)}")
            print(f"vv_current shape: {len(vv_current)}")
            
            # When converting weights to tensors, add try-except blocks
            try:
                current_weights['w1'] = torch.cat([torch.from_numpy(np.array(w)).flatten() for w in w_current[0]])
                current_weights['w2'] = torch.cat([torch.from_numpy(np.array(w)).flatten() for w in w_current[1]])
                current_weights['v1'] = torch.cat([torch.from_numpy(np.array(v)).flatten() for v in v_current[0]])
                current_weights['v2'] = torch.cat([torch.from_numpy(np.array(v)).flatten() for v in v_current[1]])
                current_weights['ww1'] = torch.cat([torch.from_numpy(np.array(w)).flatten() for w in ww_current[0]])
                current_weights['ww2'] = torch.cat([torch.from_numpy(np.array(w)).flatten() for w in ww_current[1]])
                current_weights['vv1'] = torch.cat([torch.from_numpy(np.array(v)).flatten() for v in vv_current[0]])
                current_weights['vv2'] = torch.cat([torch.from_numpy(np.array(v)).flatten() for v in vv_current[1]])
            except Exception as e:
                print(f"Error converting weights at epoch {epoch_str}: {str(e)}")
            
            # Compute similarities with initial weights for all types
            for key in similarities:
                if key in current_weights and key in initial_weights:
                    similarities[key].append(
                        compute_cosine_similarity(current_weights[key], initial_weights[key]))
                else:
                    print(f"Warning: Missing data for {key} at epoch {epoch}")
                    similarities[key].append(None)
        else:
            print(f"Warning: Missing data for ww or vv at epoch {epoch}")
            for key in similarities:
                similarities[key].append(None)
    
    # Create two subplots - one for W and one for V
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot W matrices in first subplot
    for key, style, label, color in [
        ('w1', '-o', 'W (random init)', 'blue'),
        ('w2', '-s', "W' (random init)", 'red'),
        ('ww1', '--o', 'W (V=I init)', 'lightblue'),
        ('ww2', '--s', "W' (V=I init)", 'lightcoral')
    ]:
        if len(similarities[key]) > 0:
            valid_data = [(i+1, val) for i, val in enumerate(similarities[key]) if val is not None]
            if valid_data:
                x_vals, y_vals = zip(*valid_data)
                ax1.plot(x_vals, np.abs(y_vals), style, label=label, color=color, 
                        markersize=4, markevery=2, alpha=0.8)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Absolute Cosine Similarity')
    ax1.set_title('Weight Matrices (W) Similarity to Initial State')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Plot V matrices in second subplot
    for key, style, label, color in [
        ('v1', '-^', 'V (random init)', 'green'),
        ('v2', '-D', "V' (random init)", 'magenta'),
        ('vv1', '--^', 'V (V=I init)', 'lightgreen'),
        ('vv2', '--D', "V' (V=I init)", 'plum')
    ]:
        if len(similarities[key]) > 0:
            valid_data = [(i+1, val) for i, val in enumerate(similarities[key]) if val is not None]
            if valid_data:
                x_vals, y_vals = zip(*valid_data)
                ax2.plot(x_vals, np.abs(y_vals), style, label=label, color=color, 
                        markersize=4, markevery=2, alpha=0.8)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Absolute Cosine Similarity')
    ax2.set_title('Weight Matrices (V) Similarity to Initial State')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    filename = f'similarity_comparison_{args.dataset}_epochs{args.epochs}_lr{args.lr}_hidden{args.hidden}.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    plt.clf()
    plt.close('all')

    return similarities
    
def compute_layer_cosine_similarity(tensor1, tensor2):
    """Compute cosine similarity for a single layer's weights"""
    # Ensure inputs are tensors
    t1 = tensor1 if isinstance(tensor1, torch.Tensor) else torch.from_numpy(np.array(tensor1))
    t2 = tensor2 if isinstance(tensor2, torch.Tensor) else torch.from_numpy(np.array(tensor2))
    return F.cosine_similarity(t1.flatten().unsqueeze(0), t2.flatten().unsqueeze(0)).item()

def compare_models_layer_similarity(args, layer_idx=0):
    """
    Compare weight similarities for a specific layer
    
    Args:
        args: command line arguments
        layer_idx: index of the layer to analyze (0-based)
    """
    losses, accuracies1, accuracies2, model1, model2, epoch_data = main(args)
    
    # Initialize dictionaries to store layer-specific similarities
    similarities = {
        'w1': [],    # model1's w for specific layer
        'w2': [],    # model2's w for specific layer
        'v1': [],    # model1's v for specific layer
        'v2': [],    # model2's v for specific layer
        'ww1': [],   # model1's w with V=I init for specific layer
        'ww2': [],   # model2's w with V=I init for specific layer
        'vv1': [],   # model1's v with V=I init for specific layer
        'vv2': []    # model2's v with V=I init for specific layer
    }
    
    # Get initial weights for the specific layer
    initial_weights = {
        'w1': epoch_data['w']['start'][0][layer_idx],
        'w2': epoch_data['w']['start'][1][layer_idx],
        'v1': epoch_data['v']['start'][0][layer_idx],
        'v2': epoch_data['v']['start'][1][layer_idx],
        'ww1': epoch_data['ww']['start'][0][layer_idx],
        'ww2': epoch_data['ww']['start'][1][layer_idx],
        'vv1': epoch_data['vv']['start'][0][layer_idx],
        'vv2': epoch_data['vv']['start'][1][layer_idx]
    }
    
    # Compare weights at each epoch with initial weights
    for epoch in range(args.epochs):
        epoch_str = str(epoch + 1)
        if epoch_str in epoch_data['w'] and epoch_str in epoch_data['ww']:
            # Get current weights for the specific layer
            current_weights = {
                'w1': epoch_data['w'][epoch_str][0][layer_idx],
                'w2': epoch_data['w'][epoch_str][1][layer_idx],
                'v1': epoch_data['v'][epoch_str][0][layer_idx],
                'v2': epoch_data['v'][epoch_str][1][layer_idx],
                'ww1': epoch_data['ww'][epoch_str][0][layer_idx],
                'ww2': epoch_data['ww'][epoch_str][1][layer_idx],
                'vv1': epoch_data['vv'][epoch_str][0][layer_idx],
                'vv2': epoch_data['vv'][epoch_str][1][layer_idx]
            }
            
            # Compute similarities
            for key in similarities:
                try:
                    sim = compute_layer_cosine_similarity(
                        current_weights[key],
                        initial_weights[key]
                    )
                    similarities[key].append(sim)
                except Exception as e:
                    print(f"Error computing similarity for {key} at epoch {epoch}, layer {layer_idx}: {str(e)}")
                    similarities[key].append(None)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot W matrices
    for key, style, label, color in [
        ('w1', '-o', f'W layer {layer_idx} (random init)', 'blue'),
        ('w2', '-s', f"W' layer {layer_idx} (random init)", 'red'),
        ('ww1', '--o', f'W layer {layer_idx} (V=I init)', 'lightblue'),
        ('ww2', '--s', f"W' layer {layer_idx} (V=I init)", 'lightcoral')
    ]:
        if len(similarities[key]) > 0:
            valid_data = [(i+1, val) for i, val in enumerate(similarities[key]) if val is not None]
            if valid_data:
                x_vals, y_vals = zip(*valid_data)
                ax1.plot(x_vals, np.abs(y_vals), style, label=label, color=color, 
                        markersize=4, markevery=2, alpha=0.8)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Absolute Cosine Similarity')
    ax1.set_title(f'Weight Matrices (W) Similarity to Initial State - Layer {layer_idx}')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Plot V matrices
    for key, style, label, color in [
        ('v1', '-^', f'V layer {layer_idx} (random init)', 'green'),
        ('v2', '-D', f"V' layer {layer_idx} (random init)", 'magenta'),
        ('vv1', '--^', f'V layer {layer_idx} (V=I init)', 'lightgreen'),
        ('vv2', '--D', f"V' layer {layer_idx} (V=I init)", 'plum')
    ]:
        if len(similarities[key]) > 0:
            valid_data = [(i+1, val) for i, val in enumerate(similarities[key]) if val is not None]
            if valid_data:
                x_vals, y_vals = zip(*valid_data)
                ax2.plot(x_vals, np.abs(y_vals), style, label=label, color=color, 
                        markersize=4, markevery=2, alpha=0.8)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Absolute Cosine Similarity')
    ax2.set_title(f'Weight Matrices (V) Similarity to Initial State - Layer {layer_idx}')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    filename = f'layer{layer_idx}_similarity_{args.dataset}_epochs{args.epochs}_lr{args.lr}_hidden{args.hidden}.pdf'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    # 清理matplotlib缓存
    plt.clf()
    plt.close('all')

    return similarities

def experiment_hidden_sizes(args):
    """
    Run experiments with different hidden sizes and plot the results for multiple datasets
    """

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Create a range of hidden sizes from 50 to 2000
    hidden_sizes = np.linspace(50, 2000, 10, dtype=int)
    
    # Dictionary to store results for each dataset
    all_results = {}
    
    # Color scheme for different datasets
    colors = {
        'svhn': '#2ecc71',    # green
        'stl10': '#e74c3c',   # red
        'cifar10': '#3498db'  # blue
    }
    
    # Store original hidden size and dataset
    original_hidden = args.hidden
    original_dataset = args.dataset
    
    # Run experiments for each dataset
    for dataset in ['svhn', 'stl10', 'cifar10']:
        args.dataset = dataset
        results = {
            'hidden_sizes': [],
            'accuracies1': []
        }
        
        # Set input dimension based on dataset
        if dataset == 'svhn' or dataset == 'cifar10':
            input_dim = 3072
        else:  # stl10
            input_dim = 27648
        
        for hidden_size in tqdm(hidden_sizes, desc=f"Testing {dataset} with different hidden sizes"):
            args.hidden = hidden_size
            try:
                losses, accuracies1, _, _, _, _ = main(args)
                
                # Store only feedforward net results
                results['hidden_sizes'].append(hidden_size)
                results['accuracies1'].append(accuracies1[-1])
                
                # 每次实验后清理缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error with hidden_size {hidden_size}: {str(e)}")
                continue
        
        all_results[dataset] = results
    
    # Restore original settings
    args.hidden = original_hidden
    args.dataset = original_dataset
    
    # Plot results
    plt.figure(figsize=(6, 4))
    
    for dataset, results in all_results.items():
        plt.plot(results['hidden_sizes'], results['accuracies1'], 'o-', 
                label=dataset.upper(), color=colors[dataset], alpha=0.7)
        
        # Add value annotations (only for selected points to avoid cluttering)
        for i, (hidden, acc) in enumerate(zip(results['hidden_sizes'], results['accuracies1'])):
            if i % 2 == 0:  # Annotate every other point
                plt.annotate(f'{acc:.3f}', (hidden, acc), 
                           textcoords="offset points", 
                           xytext=(0,8), ha='center', 
                           fontsize=7, color=colors[dataset])
    
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Test Accuracy')
    plt.title('Feedforward Net Performance vs Hidden Layer Size')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    filename = f'hidden_size_comparison_multiple_datasets_epochs{args.epochs}_lr{args.lr}.pdf'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    
    plt.clf()
    plt.close('all')
    
    return all_results

def grid_search_learning_rates(args):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    lr_ranges = {
        'lr_w': [0.1, 0.5, 1.0, 2.0],
        'lr_w_prime': [0.1, 0.2, 0.5, 1.0],
        'lr_v': [0.1, 0.5, 1.0, 2.0],
        'lr_v_prime': [1.0, 2.0, 4.0, 8.0]
    }
    
    results = []
    
    total_experiments = len(lr_ranges['lr_w']) * len(lr_ranges['lr_w_prime']) * \
                       len(lr_ranges['lr_v']) * len(lr_ranges['lr_v_prime'])
    
    experiment_count = 0
    
    for lr_w in lr_ranges['lr_w']:
        for lr_w_prime in lr_ranges['lr_w_prime']:
            for lr_v in lr_ranges['lr_v']:
                for lr_v_prime in lr_ranges['lr_v_prime']:
                    experiment_count += 1
                    print(f"\nExperiment {experiment_count}/{total_experiments}")
                    print(f"Testing lr_w={lr_w}, lr_w_prime={lr_w_prime}, lr_v={lr_v}, lr_v_prime={lr_v_prime}")
                    
                    args.lr_w = lr_w
                    args.lr_w_prime = lr_w_prime
                    args.lr_v = lr_v
                    args.lr_v_prime = lr_v_prime
                    
                    try:
                        losses, accuracies1, accuracies2, _, _, _ = main(args)
                        
                        result = {
                            'lr_w': lr_w,
                            'lr_w_prime': lr_w_prime,
                            'lr_v': lr_v,
                            'lr_v_prime': lr_v_prime,
                            'accuracy1': accuracies1[-1],
                            'accuracy2': accuracies2[-1],
                            'avg_accuracy': (accuracies1[-1] + accuracies2[-1]) / 2
                        }
                        results.append(result)
                        
                    except Exception as e:
                        print(f"Error in experiment: {str(e)}")
                        continue
    
    sorted_results = sorted(results, key=lambda x: x['avg_accuracy'], reverse=True)
    
    filename = f'grid_search_results_{args.dataset}_epochs{args.epochs}_hidden{args.hidden}.json'
    with open(os.path.join('logs', filename), 'w') as f:
        json.dump({
            'config': vars(args),
            'results': sorted_results
        }, f, indent=2)
    
    print("\nTop 5 Configurations:")
    for i, result in enumerate(sorted_results[:5]):
        print(f"\nRank {i+1}:")
        print(f"Learning rates: w={result['lr_w']}, w'={result['lr_w_prime']}, "
              f"v={result['lr_v']}, v'={result['lr_v_prime']}")
        print(f"Accuracies: FF={result['accuracy1']:.4f}, GN={result['accuracy2']:.4f}, "
              f"Avg={result['avg_accuracy']:.4f}")
        
    visualize_lr_results(results)
    
    return sorted_results

def visualize_lr_results(results):
    """
    Visualize how different learning rates affect model accuracy
    Args:
        results: List of dictionaries containing learning rates and accuracies
    """
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each learning rate parameter
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # Group results by each learning rate
    lr_w_groups = {}
    lr_v_groups = {}
    lr_w_prime_groups = {}
    lr_v_prime_groups = {}
    
    for result in results:
        # Group by lr_w
        lr_w = result['lr_w']
        if lr_w not in lr_w_groups:
            lr_w_groups[lr_w] = []
        lr_w_groups[lr_w].append(result['avg_accuracy'])
        
        # Group by lr_v
        lr_v = result['lr_v']
        if lr_v not in lr_v_groups:
            lr_v_groups[lr_v] = []
        lr_v_groups[lr_v].append(result['avg_accuracy'])
        
        # Group by lr_w_prime
        lr_w_prime = result['lr_w_prime']
        if lr_w_prime not in lr_w_prime_groups:
            lr_w_prime_groups[lr_w_prime] = []
        lr_w_prime_groups[lr_w_prime].append(result['avg_accuracy'])
        
        # Group by lr_v_prime
        lr_v_prime = result['lr_v_prime']
        if lr_v_prime not in lr_v_prime_groups:
            lr_v_prime_groups[lr_v_prime] = []
        lr_v_prime_groups[lr_v_prime].append(result['avg_accuracy'])
    
    # Plot for lr_w
    plot_lr_accuracy(ax1, lr_w_groups, 'lr_w', 'Learning Rate W')
    
    # Plot for lr_v
    plot_lr_accuracy(ax2, lr_v_groups, 'lr_v', 'Learning Rate V')
    
    # Plot for lr_w_prime
    plot_lr_accuracy(ax3, lr_w_prime_groups, 'lr_w_prime', "Learning Rate W'")
    
    # Plot for lr_v_prime
    plot_lr_accuracy(ax4, lr_v_prime_groups, 'lr_v_prime', "Learning Rate V'")
    
    plt.tight_layout()
    plt.savefig('learning_rates_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_lr_accuracy(ax, lr_groups, lr_name, title):
    """Helper function to plot learning rate vs accuracy"""
    lr_values = sorted(lr_groups.keys())
    accuracies_mean = [np.mean(lr_groups[lr]) for lr in lr_values]
    accuracies_std = [np.std(lr_groups[lr]) for lr in lr_values]
    
    # Plot mean accuracy
    ax.plot(lr_values, accuracies_mean, 'o-', label='Mean Accuracy')
    
    # Add error bars
    ax.fill_between(lr_values,
                   [m - s for m, s in zip(accuracies_mean, accuracies_std)],
                   [m + s for m, s in zip(accuracies_mean, accuracies_std)],
                   alpha=0.2)
    
    ax.set_xlabel(f'{title}')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Accuracy vs {title}')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xscale('log')  # Use log scale for learning rates
    
    # Add value annotations
    for x, y in zip(lr_values, accuracies_mean):
        ax.annotate(f'{y:.3f}', 
                   (x, y), 
                   textcoords="offset points", 
                   xytext=(0,10), 
                   ha='center')

def compute_effective_rank(matrix):
    """
    Compute the effective rank of a matrix using singular value entropy
    """
    # Compute singular values
    s = np.linalg.svd(matrix, compute_uv=False)
    
    # Normalize singular values
    s_normalized = s / np.sum(s)
    
    # Remove zeros to avoid log(0)
    s_normalized = s_normalized[s_normalized > 1e-12]
    
    # Compute entropy
    effective_rank = -np.sum(s_normalized * np.log(s_normalized))
    
    return effective_rank

def visualize_gram_matrices(model1, model2, args, suffix=''):
    """
    Visualize Gram matrices with non-uniform color distribution
    """
    plt.clf()
    
    # Extract matrices
    w_matrices1 = []
    v_matrices2 = []
    
    for layer in model1.layers:
        if hasattr(layer, 'w'):
            w_matrices1.append(layer.w.weight.detach().cpu().numpy())
    
    for layer in model2.layers:
        if hasattr(layer, 'v') and layer.v is not None:
            v_matrices2.append(layer.v.weight.detach().cpu().numpy())
    
    # Create color list with non-uniform spacing
    # More colors in blue range, fewer in white range
    colors = [
        '#2E6B9D',  # Deep blue
        '#3579AB',  # Deep medium blue
        '#3D87B9',  # Medium deep blue
        '#4595C7',  # Medium blue
        '#4DA3D5',  # Medium light blue
        '#55B1E3',  # Light medium blue
        '#71B2D9',  # Light blue
        '#8BC3E1',  # Lighter blue
        '#A5D4E9',  # Very light blue
        '#CFE5EF',  # Extra light blue
        '#E8F1F5',  # Nearly white blue
        '#FFFFFF',  # White
    ]
    
    # Create non-uniform spacing for colors
    positions = np.array([
        0.0,    # Deep blue
        0.15,   # Deep medium blue
        0.25,   # Medium deep blue
        0.35,   # Medium blue
        0.45,   # Medium light blue
        0.55,   # Light medium blue
        0.65,   # Light blue
        0.75,   # Lighter blue
        0.82,   # Very light blue
        0.90,   # Extra light blue
        0.95,   # Nearly white blue
        1.0     # White
    ])
    
    custom_cmap = LinearSegmentedColormap.from_list('custom_blue_white', 
                                                   list(zip(positions, colors[::-1])))
    
    def enhance_matrix(mat):
        """Enhanced contrast function with non-linear mapping"""
        # Normalize the matrix
        mat_norm = mat / np.max(np.abs(mat))
        
        # Apply moderate power transformation
        mat_enhanced = np.sign(mat_norm) * np.power(np.abs(mat_norm), 0.5)
        
        # Apply percentile-based normalization
        percentiles = np.percentile(mat_enhanced, [5, 95])
        mat_enhanced = np.clip(mat_enhanced, percentiles[0], percentiles[1])
        mat_enhanced = (mat_enhanced - percentiles[0]) / (percentiles[1] - percentiles[0])
        
        # Apply non-linear transformation to compress white region
        mat_enhanced = np.power(mat_enhanced, 1.5)
        
        # Invert the values
        return 1 - mat_enhanced
    
    def plot_gram_matrix(matrix, title, filename, hidden_dim):
        plt.figure(figsize=(6, 5))
        
        plt.imshow(matrix,
                  cmap=custom_cmap,
                  aspect='equal',
                  interpolation='nearest',
                  vmin=0,
                  vmax=1,
                  origin='lower')
        
        plt.title(title, pad=10, fontsize=14)  # Increased font size
        plt.colorbar(fraction=0.046, pad=0.04)
        
        plt.xlabel('Hidden dimension', fontsize=12)  # Increased font size
        plt.ylabel('Hidden dimension', fontsize=12)  # Increased font size
        
        # Create ticks every 10 steps
        tick_step = 10
        tick_positions = np.arange(0, hidden_dim, tick_step)
        if hidden_dim - 1 not in tick_positions:
            tick_positions = np.append(tick_positions, hidden_dim - 1)
        
        # Create tick labels (starting from 0)
        tick_labels = [str(pos) for pos in tick_positions]
        
        plt.xticks(tick_positions, tick_labels, fontsize=10)  # Increased font size
        plt.yticks(tick_positions, tick_labels, fontsize=10)  # Increased font size
        
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
    
    # Plot each matrix separately
    for layer_idx in range(len(w_matrices1)):
        # Plot WW^T
        W = w_matrices1[layer_idx]
        WWT = np.dot(W, W.T)
        WWT_enhanced = enhance_matrix(WWT)
        hidden_dim = WWT.shape[0]
        
        suffix_str = f'_{suffix}' if suffix else ''
        filename = f'gram_matrix_W_layer{layer_idx+1}_hidden{args.hidden}_{args.dataset}{suffix_str}.pdf'
        plot_gram_matrix(WWT_enhanced, 
                        f'Layer {layer_idx+1} - WW^T',
                        filename,
                        hidden_dim)
        
        # Plot V'V'^T if available
        if layer_idx < len(v_matrices2):
            V = v_matrices2[layer_idx]
            VVT = np.dot(V, V.T)
            VVT_enhanced = enhance_matrix(VVT)
            hidden_dim = VVT.shape[0]
            
            filename = f'gram_matrix_V_layer{layer_idx+1}_hidden{args.hidden}_{args.dataset}{suffix_str}.pdf'
            plot_gram_matrix(VVT_enhanced,
                           f"Layer {layer_idx+1} - V'V'^T",
                           filename,
                           hidden_dim)
    
    plt.clf()
    plt.close('all')

def plot_effective_ranks(model1, model2, args, suffix=''):
    """
    Plot effective ranks across layers for WW^T, V'V'^T, and W_barW_bar^T
    """
    plt.clf()
    
    # Extract matrices
    w_matrices1 = []
    v_matrices2 = []
    w_bar_matrices = []  # 新增：用于存储W_bar矩阵
    
    for layer in model1.layers:
        if hasattr(layer, 'w'):
            w = layer.w.weight.detach().cpu().numpy()
            w_matrices1.append(w)
            # 计算W_bar = W/||W||_F
            w_norm = np.linalg.norm(w, 'fro')
            w_bar = w / w_norm if w_norm > 0 else w
            w_bar_matrices.append(w_bar)
    
    for layer in model2.layers:
        if hasattr(layer, 'v') and layer.v is not None:
            v_matrices2.append(layer.v.weight.detach().cpu().numpy())
    
    # Compute effective ranks for each layer
    w_ranks = []
    v_ranks = []
    w_bar_ranks = []  # 新增：存储W_bar的有效秩
    
    for W in w_matrices1:
        WWT = np.dot(W, W.T)
        w_ranks.append(compute_effective_rank(WWT))
    
    for V in v_matrices2:
        VVT = np.dot(V, V.T)
        v_ranks.append(compute_effective_rank(VVT))
    
    for W_bar in w_bar_matrices:  # 新增：计算W_bar的有效秩
        W_barW_barT = np.dot(W_bar, W_bar.T)
        w_bar_ranks.append(compute_effective_rank(W_barW_barT))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot effective ranks
    w_layers = range(1, len(w_ranks) + 1)
    v_layers = range(1, len(v_ranks) + 1)
    w_bar_layers = range(1, len(w_bar_ranks) + 1)
    
    plt.plot(w_layers, w_ranks, 'o-', color='#2E86C1', label='WW^T', linewidth=2, markersize=8)
    plt.plot(v_layers, v_ranks, 's-', color='#E74C3C', label="V'V'^T", linewidth=2, markersize=8)
    plt.plot(w_bar_layers, w_bar_ranks, '^-', color='#27AE60', label='W̄W̄^T', linewidth=2, markersize=8)  # 新增：绘制W_bar的曲线
    
    # Add value annotations
    for i, w_rank in enumerate(w_ranks):
        plt.annotate(f'{w_rank:.2f}', 
                    (i + 1, w_rank), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    color='#2E86C1')
    
    for i, v_rank in enumerate(v_ranks):
        plt.annotate(f'{v_rank:.2f}', 
                    (i + 1, v_rank), 
                    textcoords="offset points", 
                    xytext=(0, -15), 
                    ha='center',
                    color='#E74C3C')
    
    for i, w_bar_rank in enumerate(w_bar_ranks):  # 新增：W_bar的标注
        plt.annotate(f'{w_bar_rank:.2f}', 
                    (i + 1, w_bar_rank), 
                    textcoords="offset points", 
                    xytext=(0, 25), 
                    ha='center',
                    color='#27AE60')
    
    # Customize the plot
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Effective Rank', fontsize=12)
    plt.title(f'Effective Rank Across Layers\n{args.dataset.upper()}', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Set integer ticks for x-axis
    max_layers = max(len(w_ranks), len(v_ranks), len(w_bar_ranks))
    plt.xticks(range(1, max_layers + 1))
    
    # Add some padding to y-axis
    plt.margins(y=0.2)
    
    # Save the plot
    suffix_str = f'_{suffix}' if suffix else ''
    filename = f'effective_ranks_hidden{args.hidden}_{args.dataset}_epochs{args.epochs}_lr{args.lr}{suffix_str}.pdf'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    
    plt.clf()
    plt.close('all')
    
    return w_ranks, v_ranks, w_bar_ranks

if __name__ == "__main__":
    args = get_args()
    
    if args.dataset == 'symbolic_regression':
        input_dim = 4
    elif args.dataset == 'eurosat':
        input_dim = 64 * 64 * 3
    elif args.dataset == 'pathmnist':
        input_dim = 28 * 28 * 3
    elif args.dataset == 'cifar10':
        input_dim = 3072
    elif args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        input_dim = 784
    elif args.dataset == 'svhn':
        input_dim = 3072
    elif args.dataset == 'stl10':
        input_dim = 27648  # 96x96x3
    elif args.dataset == 'caltech101' or args.dataset == 'oxford_iiit_pet' or args.dataset == 'flowers102':
        input_dim = 27648  # 224x224x3
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    losses, accuracies1, accuracies2, model1, model2, epoch_data = main(args)
    save_accuracies(accuracies1, accuracies2, args)
    plot_loss_acc(losses, accuracies1, accuracies2, args)
    # visualize_gram_matrices(model1, model2, args, suffix='final')
    # plot_effective_ranks(model1, model2, args, suffix='final')






