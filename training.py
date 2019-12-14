import torch.optim as optim
import torch.nn as nn
import time
from torch.nn import Module
from torch import squeeze
from torch import save
from torch.utils.data import DataLoader
from network import VGGOcclusion
from imagenet_dataset import ImageNetData
import os
import torch

statedict_path = os.getcwd() + "/vgg16-397923af.pth"


def get_train_loader(batch_size, dataset_path: str = "imagenet/", limit: int = -1):
    dataset = ImageNetData(dataset_path, limit)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True, num_workers=1)
    return data_loader


def create_loss_and_optimizer(net, learning_rate=0.001):
    loss = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss = loss.cuda()
    #optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    #optimizer = optim.Adagrad(net.parameters(), lr=learning_rate)
    return loss, optimizer


def train(model: Module, data_loader: DataLoader, n_epochs: int = 10, learning_rate: float = 0.00001, checkpoint=True):
    # Create our loss and optimizer functions
    model.train()
    loss_fn, optimizer = create_loss_and_optimizer(model, learning_rate)

    # Time for printing
    training_start_time = time.time()
    running_loss = 0
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, label = data
            
            # We are not using batch training, so we have to take the first element.
            point = squeeze(inputs, dim=0)
            # zero the parameter gradients
            
            if torch.cuda.is_available():
                out = out.cuda()

            # forward + backward + optimize
            out = model(point)
            label = label.long()

            loss = loss_fn(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print("Target: {}, Prediction: {}".format(label, out.data.numpy().argmax()))
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            # We should save at every epoch.
        if checkpoint:
            out_path = "models/checkpoint-{}-{}-{}.pth".format(epoch, learning_rate, limit)
            print("saving checkpoint for epoch ", epoch)
            save(model.state_dict(), out_path)

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


def train_runner(batch_size, n_epochs, learning_rate, limit=-1):
    # Print all of the hyper parameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("image limit=", limit)
    print("=" * 30)

    
    model = VGGOcclusion(frozen_vgg=False)
    if torch.cuda.is_available():
        model = model.cuda()
    # Using the full image net dataset
    imagenet_data = ImageNetData("imagenet", limit)
    image_loader = DataLoader(imagenet_data, batch_size=batch_size, shuffle=True)
    output_path = "models/output-{}-{}-{}.pth".format(batch_size, learning_rate, limit)
    train(model, image_loader, n_epochs, learning_rate)
    save(model.state_dict(), output_path)
    print('saved model to ', output_path)
    return model


if __name__ == '__main__':
    rate = float(input("Learning rate? (Preferably a small number like 0.000001)\n"))
    limit = int(input("How many images? \n"))
    epochs = int(input("Epochs?\n"))
    batch_size = 1 # Batch training doesn't work yet.
    train_runner(batch_size=batch_size, n_epochs=epochs, learning_rate=rate, limit=limit)


