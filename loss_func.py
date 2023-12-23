import numpy as np
import torch

# X is the input matrix of size: (1206, 2), phis is the ANN output matrix of size: (1206, 3306)
def loss_function(X, phis): 

    # Note: rows = size of minibatch. In our case of full batch SGD, rows = 1206
    rows = X.shape[0]

    # Compute residues at all points for all samples
    residues = res(X, phis) # Size: (1206, 3045)
    
    # Transposing the residues mmatrix such that we have individual node residues for each of the samples
    node_wise_residues = np.transpose(residues, (1, 0)) # Size: (3045, 1206)

    # Rearranging in the form of a 3D grid of size (60, 59, 1206)
    # This makes it easier to access losses for all samples at each grid point locations
    grid_residues = np.zeros(shape=(60, 59, rows))
    for j in range(60):
        for i in range(59):
            grid_residues[j][i] = node_wise_residues[int(i * 60) + j]
    # Each grid_residues[j][i] is now an array of losses of size 1206

    # Weightages
    lambda_normal = 1
    lambda_on_plate = 10000
    lambda_off_plate = 1
    lambda_on_wake = 10000
    lambda_off_wake = 1

    losses = []

    # Compute node wise losses by iterating over the 3D grid grid_residues
    for j in range(0, 60):  # From 0 to 59
        for i in range(0, 59):  # From 0 to 58

            # points on the plate
            if 17 <= i <= 37 and (j == 29 or j == 30):
                losses.append((lambda_on_plate * np.linalg.norm(grid_residues[j][i])) / rows) # Sum is divided by number of rows to get the average loss for the batch

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