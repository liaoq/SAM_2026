import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from cellular_backpropmata.examples.cell_model import Linear_V, Net_V, F_identity, F_relu, update_grad_v_no_mask2

# Set random seed
seed = 3
np.random.seed(seed)
torch.manual_seed(seed)

# Create toy dataset
def f(data):
    x1 = data[:,[0]]
    x2 = data[:,[1]]
    x3 = data[:,[2]]
    x4 = data[:,[3]]
    out = np.transpose(np.array([np.sqrt((x1-x2)**2+(x3-x4)**2)]))
    return out

# Generate data
d_in = 4
d_out = 1

inputs = np.random.rand(100,d_in)*2-1
labels = f(inputs)
inputs = torch.tensor(inputs, dtype=torch.float, requires_grad=True)
labels = torch.tensor(labels, dtype=torch.float, requires_grad=True)

inputs_test = np.random.rand(100,d_in)*2-1
labels_test = f(inputs_test)
inputs_test = torch.tensor(inputs_test, dtype=torch.float, requires_grad=True)
labels_test = torch.tensor(labels_test, dtype=torch.float, requires_grad=True)

# Define model using the existing Linear and Net classes

# Training setup
model1 = Net_V(in_d=4, out_d=1, hidden=20, F=F_relu, use_batch_norm=None).to('cuda')
model2 = Net_V(in_d=4, out_d=1, hidden=20, F=F_relu, use_batch_norm=None).to('cuda')
loss_fn = nn.MSELoss()

# Training loop
epochs = 10
losses = []
for epoch in range(epochs):
    loss1, loss2 = update_grad_v_no_mask2(model1, model2, inputs, labels)
    losses.append((loss1, loss2))

# Visualization
def visualize_toy_network(model, save_path=None):
    plt.figure(figsize=(10, 6))
    
    # Define layer sizes
    layer_sizes = [d_in, 20, 20, 20, d_out]
    
    # Extract weights
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data.cpu().numpy())
    
    # Normalize weights for visualization
    max_weight = max([np.abs(w).max() for w in weights])
    min_weight = min([np.abs(w).min() for w in weights])
    
    # Plot neurons and connections
    for i, (n_neurons, next_n_neurons) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        # Plot neurons in current layer
        y_positions = np.linspace(0, 1, n_neurons)
        for y in y_positions:
            circle = plt.Circle((i, y), 0.02, color='skyblue', fill=True)
            plt.gca().add_patch(circle)
        
        # Plot neurons in next layer
        if i < len(layer_sizes) - 1:
            next_y_positions = np.linspace(0, 1, next_n_neurons)
            for y_next in next_y_positions:
                circle = plt.Circle((i+1, y_next), 0.02, color='skyblue', fill=True)
                plt.gca().add_patch(circle)
            
            # Plot connections
            weight_matrix = weights[i]
            for j, y1 in enumerate(y_positions):
                for k, y2 in enumerate(next_y_positions):
                    weight = np.abs(weight_matrix[k, j])
                    width = 0.5 * (weight - min_weight) / (max_weight - min_weight) + 0.1
                    alpha = 0.3 * (weight - min_weight) / (max_weight - min_weight) + 0.2
                    plt.plot([i, i+1], [y1, y2], 'gray', linewidth=width, alpha=alpha)
    
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.title('Toy Network Structure\n(Line thickness represents weight magnitude)')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

# Visualize the network
visualize_toy_network(model1, save_path='toy_network_structure.pdf')