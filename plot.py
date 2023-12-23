import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Read the data from the text file
epochs = []
losses = []
path = "/Users/shagufta/code/pinns 3/out_ind_loss.txt"
with open(path, "r") as file:
    lines = file.readlines()
    for line in lines[12:]:
        epoch, loss = line.strip().split(", ")
        epochs.append(int(epoch))
        losses.append(float(loss))

# Plot the training loss
plt.plot(epochs, losses, marker='o', color='b')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 1)

# Add information to the graph
arch = "2, 1206, 3306"
batch_size = "Full"
learning_rate = 0.001
optimiser = "SGD with mom 0.9"
activation = "Tanh"
initializer = "Uniform random distribution\nwith range [-0.001, 0.001]"
loss_type = "Individual output node losses"
info_text = f"Architecture: {arch}\nBatch size = {batch_size}\nLearning Rate = {learning_rate}\nOptimiser = {optimiser}\nActivation = {activation}\nInitializer = {initializer}\nLoss Type = {loss_type}"
plt.text(0.98, 0.97, info_text, transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))

# Custom formatter function for y axis
def custom_format(x, pos):
    return f'{round(x, 6):,}'

# Apply the custom formatter to the y-axis
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(custom_format))

# Show the plot
plt.show()