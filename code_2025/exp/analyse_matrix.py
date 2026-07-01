import glob
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

dir='./checkpoints'

def plot_eigenvalue_spectrum(matrix, title):
    eigenvalues = np.linalg.eigvals(matrix.cpu().numpy())
    magnitudes = np.abs(eigenvalues)
    sorted_magnitudes = np.sort(magnitudes)[::-1]  # Sort in descending order
    
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(sorted_magnitudes) + 1), sorted_magnitudes, 'b-', marker='o', markersize=4, alpha=0.7)
    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Eigenvalue magnitude', fontsize=14)
    plt.title(f'Eigenvalue spectrum of {title}', fontsize=16, fontweight='bold')
    plt.yscale('log')
    # plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Add colorbar to represent phase
    phases = np.angle(eigenvalues)
    scatter = plt.scatter(range(1, len(sorted_magnitudes) + 1), sorted_magnitudes, c=phases, cmap='hsv', alpha=0.7)
    plt.colorbar(scatter, label='Phase', ax=plt.gca())
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure with higher DPI
    plt.savefig(f'{title}_eigenvalue_spectrum.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def load_model_weights(directory):
    weight_files = glob.glob(os.path.join(directory, '*.pth'))
    weights = {}
    
    for file in weight_files:
        name = os.path.basename(file)
        weights[name] = torch.load(file)
    
    return weights

model_weights = load_model_weights(dir)

for name in model_weights.keys():
    print(name)
    if isinstance(model_weights[name], dict):
        print('model_weights[name]', model_weights[name])
        if 'w' in model_weights[name] and 'w2' in model_weights[name] and 'w3' in model_weights[name]:
            # This is a model_w_weights.pth file
            for key in ['w', 'w2', 'w3']:
                value = model_weights[name][key]
                if isinstance(value, torch.Tensor):
                    # Ensure the tensor is 2D and square
                    if value.dim() > 2:
                        value = value.view(-1, value.size(-1))
                    if value.shape[0] != value.shape[1]:
                        min_dim = min(value.shape[0], value.shape[1])
                        value = value[:min_dim, :min_dim]
                    plot_eigenvalue_spectrum(value, f"{name}_{key}")
        elif 'weight' in model_weights[name]:
            # This is a regular dict with 'weight' key
            value = model_weights[name]['weight']
            if isinstance(value, torch.Tensor):
                # Ensure the tensor is 2D and square
                if value.dim() > 2:
                    value = value.view(-1, value.size(-1))
                if value.shape[0] != value.shape[1]:
                    min_dim = min(value.shape[0], value.shape[1])
                    value = value[:min_dim, :min_dim]
                plot_eigenvalue_spectrum(value, f"{name}_weight")
    elif isinstance(model_weights[name], torch.Tensor):
        value = model_weights[name]
        # Ensure the tensor is 2D and square
        if value.dim() > 2:
            value = value.view(-1, value.size(-1))
        if value.shape[0] != value.shape[1]:
            min_dim = min(value.shape[0], value.shape[1])
            value = value[:min_dim, :min_dim]
        plot_eigenvalue_spectrum(value, name)

target_file = 'model2_v_weights_layer_1.pth'
if target_file in model_weights:
    print(f"\n{target_file} 's shape:{model_weights[target_file]['weight'].shape}")
    print(model_weights[target_file]['weight'].shape)
else:
    print(f"\n cannot find the {target_file}")
    
target_file = 'model1_v_weights_layer_1.pth'
if target_file in model_weights:
    print(f"\n{target_file} 's shape:{model_weights[target_file]['weight'].shape}")
    print(model_weights[target_file]['weight'].shape)
else:
    print(f"\n cannot find the {target_file}")

target_file = 'model1_w_weights.pth'
if target_file in model_weights:
    print(f"\n{target_file} 's contents'")
    for key, value in model_weights[target_file].items():
        if isinstance(value, torch.Tensor):
            print(f"  {key} 's shape: {value.shape}")
        else:
            print(f"  {key} is not tensor")
else:
    print(f"\n cannot find the {target_file}")