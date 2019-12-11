import torch.optim as optim
import torch.nn as nn
import time
from torch.nn import Module
from torch import squeeze
from torch import save
from occlusion_data import CarDataset
from torch.utils.data import DataLoader
from network import VGGOcclusion
import os

statedict_path = os.getcwd() + "/vgg16-397923af.pth"


def get_train_loader(batch_size, dataset_path: str = "cars_train", limit: int = -1):
    dataset = CarDataset(dataset_path, limit)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True, num_workers=1)
    return data_loader


def create_loss_and_optimizer(net, learning_rate=0.001):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    return loss, optimizer


def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        # Making the prediction
        x = squeeze(x, dim=1)  # (batch_size, colors, width, height)
        # x = x.view(expected_shape)
        y_pred = model(x)
        # loss function - input (n,c) where c = number of classes
        #               - target (n) where each value is 0 <= targets[i] <= c-1
        y = y.long()
        loss = loss_fn(y_pred, y)
        # Computes the gradients
        loss.backward()
        # Update step
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step


def train(model: Module, data_loader: DataLoader, n_epochs: int = 10, learning_rate: float = 0.001):
    # Create our loss and optimizer functions
    loss_fn, optimizer = create_loss_and_optimizer(model, learning_rate)

    # Time for printing
    training_start_time = time.time()
    losses = []
    train_step = make_train_step(model, loss_fn, optimizer)
    running_loss = 0

    for epoch in range(n_epochs):
        start_time = time.time()
        i = 0
        for x_batch, y_batch in data_loader:
            loss = train_step(x_batch, y_batch)
            losses.append(loss)
            i += 1
            if i % len(data_loader)//10 == 0:
                print("Epoch {}% done.".format(i / len(data_loader) * 100))
        running_loss = sum(losses) - running_loss
        print("Epoch {}, \t train_loss: {:.2f} took: {:.2f}s".format(
            epoch + 1, running_loss, time.time() - start_time))

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


def train_runner(batch_size, n_epochs, learning_rate, limit=-1):
    # Print all of the hyper parameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    model = VGGOcclusion()
    data = get_train_loader(batch_size, limit=limit)
    output_path = "output{}-{}-{}.pth".format(batch_size, learning_rate, limit)
    train(model, data)
    save(model.state_dict(), output_path)
    print('saved model to ', output_path)
    return model


if __name__ == '__main__':
    train_runner(batch_size=2, n_epochs=2, learning_rate=0.001, limit=10)


