import numpy as np

# Helper functions to reshape matrices

def rearrange_1d_to_2d(inp):  # maps 3540 points to (60, 59)
    arr = np.empty(shape=(60, 59), dtype=list)
    for j in range(60):
        for i in range(59):
            arr[j][i] = inp[int(i * 60) + j]
    return arr

def box_1d_to_2d(inp):  # maps 3306 points to (58, 57)
    arr = np.empty(shape=(58, 57), dtype=list)
    for j in range(58):
        for i in range(57):
            arr[j][i] = inp[int(i * 58) + j]
    return arr

def revert_2d_to_1d(inp):  # maps (60, 59) to 3540 points
    arr = []
    for i in range(59):
        for j in range(60):
            arr.append(inp[j][i])
    return arr

def box_2d_to_1d(inp):  # maps (58, 57) to 3306 points
    arr = []
    for i in range(57):
        for j in range(58):
            arr.append(inp[j][i])
    return arr
