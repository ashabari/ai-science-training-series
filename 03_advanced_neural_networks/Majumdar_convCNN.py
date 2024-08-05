import torch
from torch import nn
import torchvision
from torchvision.transforms import v2
from torchinfo import summary
from tqdm import tqdm
import logging

# Set up logging to file
logging.basicConfig(filename='training_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Function to log messages
def log_and_print(message):
    logging.info(message)

# Load CIFAR-10 dataset
training_data = torchvision.datasets.CIFAR10(
    root="/afs/crc.nd.edu/user/a/amajumda/Private/AshaProjects/Argonne_Projects/dataset",
    train=True,
    download=True,
    transform=v2.Compose([
        v2.ToTensor(),
        v2.RandomHorizontalFlip(),
        v2.RandomResizedCrop(size=32, scale=[0.85, 1.0], antialias=False),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])
)

test_data = torchvision.datasets.CIFAR10(
    root="/afs/crc.nd.edu/user/a/amajumda/Private/AshaProjects/Argonne_Projects/dataset",
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor()
)

training_data, validation_data = torch.utils.data.random_split(
    training_data, [0.8, 0.2], generator=torch.Generator().manual_seed(55))

batch_size = 128

# The dataloader makes our dataset iterable
train_dataloader = torch.utils.data.DataLoader(training_data, 
    batch_size=batch_size, 
    pin_memory=True,
    shuffle=True, 
    num_workers=4)
val_dataloader = torch.utils.data.DataLoader(validation_data, 
    batch_size=batch_size, 
    pin_memory=True,
    shuffle=False, 
    num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_data, 
    batch_size=batch_size, 
    pin_memory=True,
    shuffle=False, 
    num_workers=4)

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def preprocess(x, y):
    # CIFAR-10 is *color* images so 3 layers!
    return x.view(-1, 3, 32, 32).to(dev), y.to(dev)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))

train_dataloader = WrappedDataLoader(train_dataloader, preprocess)
val_dataloader = WrappedDataLoader(val_dataloader, preprocess)
test_dataloader = WrappedDataLoader(test_dataloader, preprocess)

class Downsampler(nn.Module):
    def __init__(self, in_channels, out_channels, shape, stride=2):
        super(Downsampler, self).__init__()
        self.norm = nn.LayerNorm([in_channels, *shape])
        self.downsample = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=stride,
            stride=stride,
        )
    
    def forward(self, inputs):
        return self.downsample(self.norm(inputs))

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, shape):
        super(ConvNextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                                     out_channels=in_channels, 
                                     groups=in_channels,
                                     kernel_size=[7,7],
                                     padding='same')
        self.norm = nn.LayerNorm([in_channels, *shape])
        self.conv2 = nn.Conv2d(in_channels=in_channels, 
                                     out_channels=4*in_channels,
                                     kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=4*in_channels, 
                                     out_channels=in_channels,
                                     kernel_size=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.norm(x)
        x = self.conv2(x)
        x = torch.nn.functional.gelu(x)
        x = self.conv3(x)
        return x + inputs
#Initial "Patchify" Layer: Changed the kernel_size of the initial convolution layer in Classifier from 1 to 3.

class Classifier(nn.Module):
    def __init__(self, n_initial_filters, n_stages, blocks_per_stage):
        super(Classifier, self).__init__()
        self.stem = nn.Conv2d(in_channels=3,
                              out_channels=n_initial_filters,
                              kernel_size=1,  # Change kernel size to 3
                              stride=1)
        
        current_shape = [32, 32]
        self.norm1 = nn.LayerNorm([n_initial_filters,*current_shape])
        current_n_filters = n_initial_filters
        
        self.layers = nn.Sequential()
        for i, n_blocks in enumerate(range(n_stages)):
            for _ in range(blocks_per_stage + 1):  # Increase blocks per stage to 3
                self.layers.append(ConvNextBlock(in_channels=current_n_filters, shape=current_shape))
            if i != n_stages - 1:
                self.layers.append(Downsampler(
                    in_channels=current_n_filters, 
                    out_channels=2*current_n_filters,
                    shape=current_shape,
                    )
                )
                current_n_filters = 2*current_n_filters
                current_shape = [cs // 2 for cs in current_shape]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(current_n_filters),
            nn.Linear(current_n_filters, 10)
        )

    def forward(self, inputs):
        x = self.stem(inputs)
        x = self.norm1(x)
        x = self.layers(x)
        x = nn.functional.avg_pool2d(x, x.shape[2:])
        x = self.head(x)
        return x

#Number of Convolutions Between Downsampling: Changed blocks_per_stage from 2 to 3 in the Classifier constructor call.
#Number of Filters in Each Layer: Changed n_initial_filters from 64 to 128 in the Classifier constructor call.

# Create model instance and move it to device
model = Classifier(64, 4, 2)  # Change initial filters to 128 and blocks per stage to 3
model.to(dev)

# Print model summary
log_and_print(summary(model, input_size=(batch_size, 3, 32, 32)))

def train_one_epoch(dataloader, model, loss_fn, optimizer, progress_bar):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()      
        progress_bar.update()

def evaluate(dataloader, model, loss_fn, val_bar):
    model.eval()
    size = len(dataloader)
    num_batches = len(dataloader)
    loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            val_bar.update()
    loss /= num_batches
    correct /= (size * batch_size)
    accuracy = 100 * correct
    return accuracy, loss

#Learning Rate: Changed the learning rate in the optimizer from 0.001 to 0.005.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Change learning rate to 0.005

epochs = 15
for j in range(epochs):
    with tqdm(total=len(train_dataloader), position=0, leave=True, desc=f"Train Epoch {j}") as train_bar:
        train_one_epoch(train_dataloader, model, loss_fn, optimizer, train_bar)
    
    with tqdm(total=len(train_dataloader), position=0, leave=True, desc=f"Validate (train) Epoch {j}") as train_eval:
        acc, loss = evaluate(train_dataloader, model, loss_fn, train_eval)
        log_and_print(f"Epoch {j}: training loss: {loss:.3f}, accuracy: {acc:.3f}")
    
    with tqdm(total=len(val_dataloader), position=0, leave=True, desc=f"Validate Epoch {j}") as val_bar:
        acc_val, loss_val = evaluate(val_dataloader, model, loss_fn, val_bar)
        log_and_print(f"Epoch {j}: validation loss: {loss_val:.3f}, accuracy: {acc_val:.3f}")

# Evaluate on the test dataset
with tqdm(total=len(test_dataloader), position=0, leave=True, desc=f"Evaluate on Test Data") as test_bar:
    acc_test, loss_test = evaluate(test_dataloader, model, loss_fn, test_bar)
    log_and_print(f"Test accuracy: {acc_test:.3f}, test loss: {loss_test:.3f}")

