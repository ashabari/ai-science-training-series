import torch
import torchvision
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import time
import contextlib
import sys

# Function to redirect print statements to a file
@contextlib.contextmanager
def stdout_redirect_to_file(filename):
    with open(filename, 'w') as f:
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Use the context manager to redirect stdout to a file
output_file = "output_nonlinear.txt"
with stdout_redirect_to_file(output_file):
    # Load the MNIST training dataset
    training_data = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    # Load the MNIST test dataset
    test_data = torchvision.datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    # Split the training dataset into training and validation sets
    training_data, validation_data = torch.utils.data.random_split(training_data, [48000, 12000], generator=torch.Generator().manual_seed(55))

    # Print the number of examples in each dataset
    print('MNIST data loaded: train:', len(training_data), 'examples, validation:', len(validation_data), 'examples, test:', len(test_data), 'examples')

    # Print the shape of the input data for the first example in the training dataset
    print('Input shape', training_data[0][0].shape)

    # Plot the first 10 images from the training dataset
    pltsize = 1
    plt.figure(figsize=(10 * pltsize, pltsize))

    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.axis('off')
        plt.imshow(np.reshape(training_data[i][0], (28, 28)), cmap="gray")
        plt.title('Class: ' + str(training_data[i][1]))

    plt.show()

    # Define batch size
    batch_size = 128

    # Create data loaders for training, validation, and test datasets
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # Define a nonlinear classifier
    class NonlinearClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.layers_stack = nn.Sequential(
                nn.Linear(28 * 28, 50),
                nn.ReLU(),
                nn.Linear(50, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            )
        
        def forward(self, x):
            x = self.flatten(x)
            x = self.layers_stack(x)
            return x

    # Initialize the nonlinear classifier model
    nonlinear_model = NonlinearClassifier()
    print(nonlinear_model)

    # Define the loss function and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(nonlinear_model.parameters(), lr=0.1)

    # Function to train the model for one epoch
    def train_one_epoch(dataloader, model, loss_fn, optimizer):
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Function to evaluate the model
    def evaluate(dataloader, model, loss_fn):
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        loss /= num_batches
        correct /= size
        accuracy = 100 * correct
        return accuracy, loss

    # Measure the time taken for training
    start_time = time.time()

    # Number of epochs to train the model
    epochs = 5
    for j in range(epochs):
        train_one_epoch(train_dataloader, nonlinear_model, loss_fn, optimizer)
        acc, loss = evaluate(train_dataloader, nonlinear_model, loss_fn)
        print(f"Epoch {j}: training loss: {loss}, accuracy: {acc}")

    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")

    # Visualize how the model is doing on the first 10 examples from the training set
    pltsize = 1
    plt.figure(figsize=(10 * pltsize, pltsize))
    nonlinear_model.eval()
    batch = next(iter(train_dataloader))
    predictions = nonlinear_model(batch[0])

    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.axis('off')
        plt.imshow(batch[0][i, 0, :, :], cmap="gray")
        plt.title('%d' % predictions[i, :].argmax())

    plt.show()

    # Evaluate the model on the validation data
    acc_val, loss_val = evaluate(val_dataloader, nonlinear_model, loss_fn)
    print("Validation loss: %.4f, validation accuracy: %.2f%%" % (loss_val, acc_val))

    # Evaluate the model on the test data
    acc_test, loss_test = evaluate(test_dataloader, nonlinear_model, loss_fn)
    print("Test loss: %.4f, test accuracy: %.2f%%" % (loss_test, acc_test))

    # Function to show failures
    def show_failures(model, dataloader, maxtoshow=10):
        model.eval()
        batch = next(iter(dataloader))
        predictions = model(batch[0])
        
        rounded = predictions.argmax(1)
        errors = rounded != batch[1]
        print('Showing max', maxtoshow, 'first failures. '
              'The predicted class is shown first and the correct class in parentheses.')
        ii = 0
        plt.figure(figsize=(maxtoshow, 1))
        for i in range(batch[0].shape[0]):
            if ii >= maxtoshow:
                break
            if errors[i]:
                plt.subplot(1, maxtoshow, ii + 1)
                plt.axis('off')
                plt.imshow(batch[0][i, 0, :, :], cmap="gray")
                plt.title("%d (%d)" % (rounded[i], batch[1][i]))
                ii += 1
        plt.show()

    # Show the first 10 failures on the validation data
    show_failures(nonlinear_model, val_dataloader)
    # Show the first 10 failures on the test data
    show_failures(nonlinear_model, test_dataloader)

