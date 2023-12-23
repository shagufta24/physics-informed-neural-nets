# Import dependencies
import torch
import torch.nn as nn
from torch import save, load
import matplotlib.pyplot as plt 
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss
import sys
import numpy as np
import time
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)
from torch.utils.data import random_split

from config import * 
from physics import *
from globals import *
from helper import *

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
    mps_device = torch.device("cpu")
else:
    print("MPS available.")
    mps_device = torch.device("mps")

# Generate normalized Input Data for ANN
def input_data_large():
    grid_flattened = grid_gen.grid_pts.flatten().reshape(1, 7080) # [x1, y1, x2, y2, ....]
    # grid_pts : (3540, 2) -> grid_flattened : (1, 7080)

    X = np.zeros((S, input_var))  # generating input matrix (grid points + mach + alpha)
    # X : (1206, 7082)
    for i in range(S): # i goes from 0 to 1205
        for j in range(input_var): # j goes from 0 to 7081
            if j == (input_var - 2): # if j = 7080 (insert mach)
                X[i][j] = pairs[i][0]
            elif j == (input_var - 1): # if j = 7081 (insert alpha)
                X[i][j] = pairs[i][1]
            else:
                X[i][j] = grid_flattened[0][j] # for all other js, insert grid point
    return X

def normalize_column(col):
    max_val = np.max(col)
    min_val = np.min(col)
    normalized_column = (2 * col - (max_val + min_val)) / (max_val - min_val)
    return normalized_column

def input_data_small():
    X = pairs # X = [[m1, a1], [m2, a2], ...]
    
    # Normalizing X
    mach_val = X[:, 0]
    alpha_val = X[:, 1]
    normalized_mach = normalize_column(mach_val)
    normalized_alpha = normalize_column(alpha_val)
    normalized_X = np.column_stack((normalized_mach, normalized_alpha))
    return normalized_X

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.hidden_layer = nn.Linear(input_var, 1206)
        self.output_layer = nn.Linear(1206, output_var_clipped)


        # Initializing weights and biases from a uniform distribution with mean 0
        lower = -0.001
        upper = 0.001

        nn.init.uniform_(self.hidden_layer.weight, a=lower, b=upper)  # a and b are the lower and upper bounds of the uniform distribution
        nn.init.uniform_(self.output_layer.weight, a=lower, b=upper)
        nn.init.uniform_(self.hidden_layer.bias, a=lower, b=upper)
        nn.init.uniform_(self.output_layer.bias, a=lower, b=upper)

    def forward(self, x):
        x = torch.tanh(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

def print_weights(model):
    # Access the state_dict of the model
    state_dict = model.state_dict()

    # Print the weights of a specific layer (e.g., fc1)
    print("\nWeights:")
    print(state_dict['hidden_layer.weight'])

    # Print the biases of a specific layer (e.g., fc1)
    print("\nBiases:")
    print(state_dict['hidden_layer.bias'])

# Get sums of residues
def res(X, phis):
    # Convert from tensors to numpy arrays
    phis_np = phis.detach().cpu().numpy() # phis : (1206, 3306)
    X_np = X.detach().cpu().numpy() # X: (1206, 2)

    # rows in the number of samples in the minibatch
    rows = phis_np.shape[0] # 1206 in case of full batch

    # Array of residues of all samples. Size: (rows, 3540)
    residues = np.zeros(shape=(rows, output_var))

    for sample in range(rows):
        grid_phis = phis_np[sample] # (3306,)
        rearranged_grid_phis = box_1d_to_2d(grid_phis) # (58, 57)
        
        # Add outer boundary points to phis (outer points are zero)
        phis_extended = np.zeros((60, 59))
        phis_extended[1:-1, 1:-1] = rearranged_grid_phis # (60, 59)

        rearranged_grid_pts = rearrange_1d_to_2d(grid_gen.grid_pts) # (60, 59)
        # rearranged_grid_pts = torch.tensor(rearranged_grid_pts).to(torch.float).to(mps_device)

        # Compute residues for one sample of phis, alpha and mach
        grid_res = calculation(phis_extended, rearranged_grid_pts, X_np[sample][0], X_np[sample][1], grid_gen.h, grid_gen.k)
        grid_res = np.abs(grid_res) # (60, 59)

        revert_grid_res = revert_2d_to_1d(grid_res) # (3540, )
        revert_grid_res = np.array(revert_grid_res)
        resval = revert_grid_res.reshape(1, 3540) # (1, 3540)

        # Convert from numpy array to grad enabled tensor
        # resval = torch.tensor(resval, requires_grad=True)

        # Append sample residues to array of all residues 
        residues[sample] = resval

    return residues
    # return torch.tensor(residues, requires_grad=True)

# Loss function
def loss_function(X, phis): # X: (1206, 2), phis: (1206, 3306)

    # Compute residues at all points for all samples
    residues = res(X, phis) # np array of size (rows, 3045)
    
    # Note: rows = size of minibatch
    rows = residues.shape[0]

    # Storing the individual node residues for all samples in minibatch
    # Size will be (3045, rows)
    node_wise_residues = np.transpose(residues, (1, 0))

    # Rearrange in the form of a 3D grid of size (60, 59, rows) 
    # to make it easier to access all samples at each grid point location
    grid_residues = np.zeros(shape=(60, 59, rows))
    for j in range(60):
        for i in range(59):
            grid_residues[j][i] = node_wise_residues[int(i * 60) + j]

    # Weightages
    lambda_normal = 1
    lambda_on_plate = 10000
    lambda_off_plate = 1
    lambda_on_wake = 10000
    lambda_off_wake = 1

    # lambda_normal = 1
    # lambda_on_plate = 1000
    # lambda_off_plate = 100
    # lambda_on_wake = 100
    # lambda_off_wake = 10

    losses = []

    # Compute node wise losses by iterating over the 3D grid
    for j in range(0, 60):  # From 0 to 59
        for i in range(0, 59):  # From 0 to 58

            # points on the plate
            if 17 <= i <= 37 and (j == 29 or j == 30):
                losses.append((lambda_on_plate * np.linalg.norm(grid_residues[j][i])) / rows)

            # points off the plate
            elif 17 <= i <= 37 and (j == 28 or j == 31):
                losses.append((lambda_off_plate * np.linalg.norm(grid_residues[j][i]) / rows))

            # points in the wake
            elif 38 <= i <= 57 and (j == 29 or j == 30):
                losses.append((lambda_on_wake * np.linalg.norm(grid_residues[j][i])) / rows)

            # points off the wake
            elif 38 <= i <= 57 and (j == 28 or j == 31):
                losses.append((lambda_off_wake * np.linalg.norm(grid_residues[j][i])) / rows)

            # all other points
            else:
                losses.append((lambda_normal * np.linalg.norm(grid_residues[j][i])) / rows)

    return torch.tensor(losses, requires_grad=True)

def zeroeth_iter(X):

    X_np = X.detach().cpu().numpy() # X: (1206, 2)
    rows = X_np.shape[0] # 1206 in case of full batch

    # Array of phis of all samples. size (rows, 3540)
    phis_initial_all = np.zeros(shape = (rows, output_var))

    rearranged_grid_pts = rearrange_1d_to_2d(grid_gen.grid_pts) # (60, 59)
    
    for sample in range(rows):
        phis_zeros = np.zeros(shape = (60, 59))
        phis_initial = phi_calc(phis_zeros, rearranged_grid_pts, X_np[sample][0], X_np[sample][1], grid_gen.k) # (60, 59)

        revert_phis_initial = revert_2d_to_1d(phis_initial) # (3540, )
        revert_phis_initial = np.array(revert_phis_initial)
        phival = revert_phis_initial.reshape(1, 3540) # (1, 3540)

        # Append sample phis to array of all phis 
        phis_initial_all[sample] = phival
    return phis_initial_all

# Training Flow
def train(model, optimizer, num_epochs, dataloader, X):
    with open(filename, "w") as f1:
        f1.write("Architecture:\n")
        for layer in model.children():
            f1.write(str(layer) + "\n")
        f1.write(f"Batch Size: {batch_size}\n")
        f1.write(f"Epochs: {num_epochs}\n")
        f1.write(f"Learning Rate: {lr}\n")
        f1.write(f"Activation function: Tanh\n")
        f1.write(f"Initialization: None\n")
        f1.write(f"Optimizer: {optimizer.__class__.__name__}\n")
        f1.write(f"Regularization : None\n\n")
        f1.write("Epoch, Loss\n")
        f1.close()

    for epoch in range(num_epochs):
        batch_id = 0
        for batch in tqdm(dataloader):

            # Send data to MPS
            batch = batch.to(mps_device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            if epoch == 0:
                output = zeroeth_iter(X)
                output = torch.tensor(output)
            else:
                output = model(batch) # Size: [1206, 3306] since full batch
        
            # Compute the loss
            loss_per_node = loss_function(X, output) # Size: [3306]

            # Backward pass per node
            for i in range(output_var):
                loss_per_node[i].backward()

            # Or, mean backward pass
            # loss_per_node.mean().backward()
            
            # Update the weights
            optimizer.step()
            batch_id += 1

            # print("Weights:")
            # print(model.hidden_layer.weight)
            # print(model.hidden_layer.bias)
            # print(model.output_layer.weight)
            # print(model.output_layer.bias)

        # Print and save the mean loss across all output nodes at the end of each epoch
        print(f"Epoch: {epoch + 1}, Loss: {loss_per_node.mean().item()}")
        with open(filename, "a") as f1:
            f1.write(f"{epoch + 1}, {loss_per_node.mean().item()}\n")

        with open(filename[:-4] + '_state.pt', 'wb') as f2:
            save(model.state_dict(), f2)
    return

if __name__ == "__main__":
    
    # Generate Grid Points. 3540 points, each with (x,y) values
    grid_gen = GridGenerator()
    grid_gen.grid_generation()
    print("Grid generated. Shape: ", grid_gen.grid_pts.shape)
    
    # Generate input martix
    X = input_data_small()
    print("Input data size: ", X.shape)
    np.savetxt('input_data.txt', X, fmt='%.2f', delimiter=', ')

    # Convert matrix from numpy array to tensor and send to MPS
    X = torch.tensor(X).to(torch.float).to(mps_device)

    # Load dataset
    dataloader = DataLoader(X, batch_size, shuffle=True)
    print("Dataset loaded.")

    # Define model
    model = ANN().to(mps_device)
    print("Model: ", model)

    # Define optimizer
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

    # Train model
    print("Training starting")
    train(model, optimizer, num_epochs, dataloader, X)