import numpy as np
import matplotlib.pyplot as plt
from examples.cell_model import Linear_V
import torch
from torchvision import transforms, datasets
import torchvision
import seaborn as sns

def extract_parameters(model1, model2, param_type):
    """Extract parameters (v, w, tha, thb, shb, sha) from both models."""
    params = {
        'model1': [],
        'model2': []
    }
    for m1_module, m2_module in zip(model1.modules(), model2.modules()):
        if isinstance(m1_module, Linear_V):
            for model, module in [('model1', m1_module), ('model2', m2_module)]:
                if param_type in ['v', 'w']:
                    param = getattr(module, param_type, None)
                    if param is not None:
                        params[model].append(param.weight.cpu().detach().numpy())
                elif param_type in ['tha', 'thb', 'shb', 'sha']:
                    param = getattr(module, param_type, None)
                    if param is not None:
                        params[model].append(param.cpu().detach().numpy())
    return params['model1'], params['model2']

def extract_v(model1, model2):
    return extract_parameters(model1, model2, 'v')

def extract_w(model1, model2):
    return extract_parameters(model1, model2, 'w')

def visualize(model1, model2, args, vis_type='matrix', title=None, model_choice='model1', param_choice='v', wwt=False, epoch_data=None):
    """Visualize parameters of the models."""
    extract_func = extract_v if param_choice == 'v' else extract_w
    model1_data, model2_data = extract_func(model1, model2)
    data_to_visualize = model1_data if model_choice == 'model1' else model2_data

    num_layers = len(data_to_visualize)
    fig, axes = plt.subplots(1, num_layers, figsize=(6*num_layers, 5))
    axes = [axes] if num_layers == 1 else axes

    vis_functions = {
        'matrix': lambda ax, data, _: ax.imshow(data, aspect='auto', cmap='viridis'),
        'hist': lambda ax, data, _: ax.hist(data.reshape(-1), bins=50, density=True, alpha=0.7),
        'eigval': plot_eigenvalues,
        'rank': lambda ax, data, _: plot_rank(ax, data, wwt)
    }

    for i, (ax, data) in enumerate(zip(axes, data_to_visualize)):
        layer_epoch_data = {epoch: epoch_data[epoch][i] for epoch in epoch_data} if epoch_data else None
        vis_functions[vis_type](ax, data, layer_epoch_data)
        ax.set_title(f'Layer {i+1}')

    plt.suptitle(f'{param_choice.upper()} Parameters - {model_choice.capitalize()}', fontsize=16)
    plt.tight_layout()
    filename = f'{title}_visualize_{param_choice}_{model_choice}_{vis_type}_epoch_{args.epochs}_wwt_{wwt}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_eigenvalues(ax, data, epoch_data=None):
    """Plot singular values of the data for different epochs."""
    def get_singular_values(matrix):
        U, S, V = np.linalg.svd(matrix)
        return S

    colors = ['r', 'g', 'b', 'k']  # Colors for different epochs and current
    labels = ['Start', 'Middle', 'End', 'Current']  # Labels for different epochs and current

    if epoch_data:
        for i, (epoch, matrix) in enumerate(epoch_data.items()):
            singular_values = get_singular_values(matrix)
            ax.plot(singular_values, color=colors[i], label=f'{labels[i]} (Epoch {epoch})')

    # Plot current data
    current_singular_values = get_singular_values(data)
    ax.plot(current_singular_values, color=colors[-1], label=labels[-1])

    ax.axvline(10, color='r', linestyle='--')
    ax.set_yscale('log')
    ax.set_xticks([])
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Print some statistics for the current data
    print(f"Matrix shape: {data.shape}")
    print(f"Number of singular values: {len(current_singular_values)}")
    print(f"Max singular value: {current_singular_values[0]}")
    print(f"Min singular value: {current_singular_values[-1]}")

def plot_rank(ax, data, wwt, threshold=0.3e-8):
    """Plot rank of the data."""
    if wwt == 'wwt':
        ww_t = np.dot(data, data.T)
    elif wwt == 'w' or wwt == 'v':
        ww_t = data
    elif wwt == 'wtw':
        ww_t = np.dot(data.T, data)
    
    # Apply threshold to the data
    thresholded_data = np.where(np.abs(ww_t) < threshold, 0, ww_t)
    
    # Create a heatmap of the thresholded data
    sns.heatmap(thresholded_data, ax=ax, cmap='viridis', cbar=True, center=0)
    
    rank = np.linalg.matrix_rank(thresholded_data)
    
    print(f"Matrix shape: {ww_t.shape}")
    print(f"Rank: {rank}")
    print(f"Is low rank: {'Yes' if rank < min(ww_t.shape) / 2 else 'No'}")
    
    # Print the maximum and minimum singular values of the matrix
    _, S, _ = np.linalg.svd(thresholded_data)
    max_singular_value = S[0]
    min_singular_value = S[-1] if len(S) > 0 else 0
    print(f"Maximum singular value: {max_singular_value}")
    print(f"Minimum singular value: {min_singular_value}")
    
    # Calculate and print the condition number
    condition_number = max_singular_value / min_singular_value if min_singular_value != 0 else float('inf')
    print(f"Condition number: {condition_number}")

def visualize_feature_maps(model, model_choice, image_path, param_choice='v'):
    """Visualize how v or w parameters capture image features as heatmaps."""
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Use CIFAR-10 dataset
    cifar10_path = './data'
    trainset = torchvision.datasets.CIFAR10(root=cifar10_path, train=True, download=True, transform=transform)
    
    # Choose a random image from the dataset
    random_idx = torch.randint(0, len(trainset), (1,)).item()
    input_tensor, _ = trainset[random_idx]
    
    # Move the model to the same device as the input tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Forward pass through the model
    _ = model(input_tensor)

    # Extract activation maps
    activation_maps = []
    for module in model.modules():
        if isinstance(module, Linear_V):
            param = getattr(module, param_choice, None)
            if param is not None:
                # Reshape input tensor to match the weight matrix dimensions
                reshaped_input = input_tensor.view(1, -1)
                if reshaped_input.shape[1] != param.weight.shape[1]:
                    # If dimensions don't match, pad or truncate the input
                    if reshaped_input.shape[1] < param.weight.shape[1]:
                        padding = torch.zeros(1, param.weight.shape[1] - reshaped_input.shape[1]).to(device)
                        reshaped_input = torch.cat([reshaped_input, padding], dim=1)
                    else:
                        reshaped_input = reshaped_input[:, :param.weight.shape[1]]
                
                activation = torch.matmul(reshaped_input, param.weight.t())
                # Calculate the size of the activation map
                activation_size = int(np.sqrt(activation.numel()))
                # If the calculated size is too large, use a smaller fixed size
                if activation_size > 32:
                    activation_size = 32
                # Reshape the activation, taking only the first activation_size^2 elements
                activation_map = activation.view(-1)[:activation_size**2].view(1, activation_size, activation_size).detach().cpu().numpy()
                activation_maps.append(activation_map)

    # Visualize activation maps as heatmaps
    num_layers = len(activation_maps)
    fig, axes = plt.subplots(1, num_layers + 1, figsize=(5*(num_layers + 1), 5))

    # Plot original image
    axes[0].imshow(np.transpose(input_tensor.squeeze().cpu().numpy(), (1, 2, 0)))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    for i, activation_map in enumerate(activation_maps):
        # Normalize the activation map to [0, 1]
        normalized_map = (activation_map[0] - np.min(activation_map[0])) / (np.max(activation_map[0]) - np.min(activation_map[0]))
        # Invert the colormap so that smaller values are closer to white
        im = axes[i+1].imshow(normalized_map, cmap='viridis_r', interpolation='nearest')
        axes[i+1].set_title(f'Layer {i+1} Activation')
        axes[i+1].axis('off')
        plt.colorbar(im, ax=axes[i+1])

    plt.suptitle(f'{param_choice.upper()} Activation Heatmaps', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{model_choice}_activation_heatmaps_{param_choice}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Activation heatmaps have been saved.")

def extract_activations(model1, model2):
    """Extract ha, hb from model1 and model2."""
    activations = {
        'model1': {'ha': [], 'hb': []},
        'model2': {'ha': [], 'hb': []}
    }
    for i, (m1_module, m2_module) in enumerate(zip(model1.modules(), model2.modules())):
        if isinstance(m1_module, Linear_V):
            print(f"Layer {i}:")
            for model, module in [('model1', m1_module), ('model2', m2_module)]:
                for attr in ['ha', 'hb']:
                    if hasattr(module, attr):
                        data = getattr(module, attr)
                        if isinstance(data, torch.Tensor):
                            activations[model][attr].append(data.cpu().detach().numpy())
                            print(f"  {model} - {attr}: Shape {data.shape}")
                        else:
                            print(f"  {model} - {attr}: Not a tensor")
                    else:
                        print(f"  {model} - {attr}: Attribute not found")
    return activations

def visualize_activations(model1, model2, title=None):
    """Visualize covariance of activations (tha, thb, shb, sha) of the models."""
    activations = extract_activations(model1, model2)
    
    for model in ['model1', 'model2']:
        for act_type in activations[model]:
            data = activations[model][act_type]
            
            if len(data) == 0:
                print(f"No data found for {model} - {act_type}")
                continue
            
            # Calculate covariance matrix
            cov_matrix = np.cov(data, rowvar=False)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Visualize covariance matrix
            im = ax.imshow(cov_matrix, aspect='auto', cmap='viridis')
            plt.colorbar(im, ax=ax)
            
            ax.set_title(f'{model} - {act_type} Covariance')
            
            plt.tight_layout()
            plt.savefig(f'{title}_covariance_{model}_{act_type}.png', dpi=300, bbox_inches='tight')
            plt.close()

def visualize_weights(model1, model2, args, title):
    plt.figure(figsize=(20, 30))
    
    # Visualize w weights
    plt.subplot(4, 3, 1)
    visualize(model1, model2, args, vis_type='rank', title=f"{title}_w", model_choice='model1', param_choice='w',wwt='w')
    plt.title(f"{title} - w weights (model1)")
    
    plt.subplot(4, 3, 2)
    visualize(model1, model2, args, vis_type='rank', title=f"{title}_w", model_choice='model2', param_choice='w',wwt='w')
    plt.title(f"{title} - w weights (model2)")
    
    # Visualize v weights
    plt.subplot(4, 3, 4)
    visualize(model1, model2, args, vis_type='rank', title=f"{title}_v", model_choice='model1', param_choice='v',wwt='v')
    plt.title(f"{title} - v weights (model1)")
    
    plt.subplot(4, 3, 5)
    visualize(model1, model2, args, vis_type='rank', title=f"{title}_v", model_choice='model2', param_choice='v',wwt='v')
    plt.title(f"{title} - v weights (model2)")
    
    # Visualize wwt for w weights
    plt.subplot(4, 3, 7)
    visualize(model1, model2, args, vis_type='rank', title=f"{title}_w_wwt", model_choice='model1', param_choice='w', wwt='wwt')
    plt.title(f"{title} - w weights WWT (model1)")
    
    plt.subplot(4, 3, 8)
    visualize(model1, model2, args, vis_type='rank', title=f"{title}_w_wwt", model_choice='model2', param_choice='w', wwt='wwt')
    plt.title(f"{title} - w weights WWT (model2)")
    
    # Visualize wtw for w weights
    plt.subplot(4, 3, 10)
    visualize(model1, model2, args, vis_type='rank', title=f"{title}_w_wtw", model_choice='model1', param_choice='w', wwt='wtw')
    plt.title(f"{title} - w weights WTW (model1)")
    
    plt.subplot(4, 3, 11)
    visualize(model1, model2, args, vis_type='rank', title=f"{title}_w_wtw", model_choice='model2', param_choice='w', wwt='wtw')
    plt.title(f"{title} - w weights WTW (model2)")
    
    # Visualize vvt for v weights
    plt.subplot(4, 3, 3)
    visualize(model1, model2, args, vis_type='rank', title=f"{title}_v_vvt", model_choice='model1', param_choice='v', wwt='wwt')
    plt.title(f"{title} - v weights VVT (model1)")
    
    plt.subplot(4, 3, 6)
    visualize(model1, model2, args, vis_type='rank', title=f"{title}_v_vvt", model_choice='model2', param_choice='v', wwt='wwt')
    plt.title(f"{title} - v weights VVT (model2)")
    
    # Visualize vtv for v weights
    plt.subplot(4, 3, 9)
    visualize(model1, model2, args, vis_type='rank', title=f"{title}_v_vtv", model_choice='model1', param_choice='v', wwt='wtw')
    plt.title(f"{title} - v weights VTV (model1)")
    
    plt.subplot(4, 3, 12)
    visualize(model1, model2, args, vis_type='rank', title=f"{title}_v_vtv", model_choice='model2', param_choice='v', wwt='wtw')
    plt.title(f"{title} - v weights VTV (model2)")
    
    plt.tight_layout()
    filename = f'{title}_weights_with_wwt_wtw_vvt_vtv.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Weight visualizations (including WWT, WTW, VVT, and VTV) saved to {filename}")

def visualize_representations(model1, model2, title=None):
    """Visualize representations (ha ha^T and hb hb^T) of the models."""
    activations = extract_activations(model1, model2)
    
    for model in ['model1', 'model2']:
        for act_type in ['ha', 'hb']:
            if act_type not in activations[model]:
                continue
            
            data = activations[model][act_type]
            
            if len(data) == 0:
                print(f"No data found for {model} - {act_type}")
                continue
            
            num_layers = len(data)
            fig, axes = plt.subplots(1, num_layers, figsize=(6*num_layers, 5))
            axes = [axes] if num_layers == 1 else axes
            
            for i, (ax, layer_data) in enumerate(zip(axes, data)):
                # Calculate representation matrix
                rep_matrix = np.dot(layer_data.T, layer_data)
                
                # Visualize representation matrix
                im = ax.imshow(rep_matrix, aspect='auto', cmap='viridis')
                ax.set_title(f'{model} - {act_type} Layer {i+1}')
                plt.colorbar(im, ax=ax)
            
            plt.suptitle(f'{model} - {act_type} Representations', fontsize=16)
            plt.tight_layout()
            filename = f'{title}_representations_{model}_{act_type}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Representation visualizations for {model} - {act_type} have been saved to {filename}")