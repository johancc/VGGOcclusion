import torch.optim as optim
import torch.nn as nn
import time
from torch import Tensor
from torch import save
from occlusion_data import CarDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable


def get_train_loader(batch_size, dataset_path: str = "cars_train"):
    dataset = CarDataset(dataset_path, limit=20)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True, num_workers=1)
    return data_loader
r


def create_loss_and_optimizer(net, learning_rate=0.001):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    return loss, optimizer


def make_train_step(net, loss_fn, optimizer):
    def train_step(x, y):
        net.train()
        # Making the prediction
        # Reshaping to match the expected dimensions (3,224, 224)
        expected_shape = (1, 3, 224, 224)
        x = x.view(expected_shape)
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


def train(net, batch_size, n_epochs, learning_rate):
    # Print all of the hyper parameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    # Get training data
    train_loader = get_train_loader(batch_size)

    # Create our loss and optimizer functions
    loss_fn, optimizer = create_loss_and_optimizer(net, learning_rate)

    # Time for printing
    training_start_time = time.time()
    losses = []
    train_step = make_train_step(net, loss_fn, optimizer)
    running_loss = 0

    for epoch in range(n_epochs):
        start_time = time.time()
        for x_batch, y_batch in train_loader:
            loss = train_step(x_batch, y_batch)
            losses.append(loss)
        running_loss = sum(losses) - running_loss
        print("Epoch {}, \t train_loss: {:.2f} took: {:.2f}s".format(
            epoch + 1, running_loss, time.time() - start_time))
        # Reset running loss and time
        running_loss = 0.0

        # At the end of the epoch, do a pass on the validation set
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


if __name__ == '__main__':
    from network import VGGOcclusion
    model = VGGOcclusion()
    train(model, 1, 1, 0.001)
    # Saving model
    save(model.state_dict(), "model.pth")

