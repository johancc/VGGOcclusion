import requests
import torch.nn as nn
import torch
from torchvision import models
from occlusion_data import preprocess_image
from pyramid import SpatialPyramidPooling
import os

# Mapping of the index to the label from image model
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
response = requests.get(LABELS_URL)  # Make an HTTP GET request and store the response.
labels = {int(key): value for key, value in response.json().items()}
statedict_path = os.getcwd() + "/vgg16-397923af.pth"


# Network definition
def get_frozen_vgg(path: str = None):
    """
    Returns an instance of a pre-trained vgg model and freezes the weights.
    We freeze the weights because we don't want to train over the pre-trained
    layers.
    """

    if path is None:
        original_vgg = models.vgg16(pretrained=True)
    else:
        state_dict = torch.load(path)
        original_vgg = models.vgg16()
        original_vgg.load_state_dict(state_dict)

    for param in original_vgg.parameters():
        param.requires_grad = False

    # We don't want the last 7 modules.
    return original_vgg


class VGGOcclusion(nn.Module):
    """
    This model modifies a pre-trained VGG16 model to have
    better object detection under occlusion.
    """
    def __init__(self, pretrained_path: str = None):
        super(VGGOcclusion, self).__init__()

        d_out = 1000  # Output dimension, there are 1000 classes
        # Everything but the last layer of the features
        features, avg, classifier = get_frozen_vgg(pretrained_path).children()

        # We want the output of the 4th max pool layer
        features = list(features)[:-7]
        features.extend(
            [
                # Represents the "part map" activations
                nn.Conv2d(512, 512, kernel_size=(15, 15), stride=(1, 1), padding=(1, 1)),
                # This layer enforces the spatial constraints.
                SpatialPyramidPooling([4, 3, 2]),
                # Makes the network robust against over fitting
                nn.Dropout(0.1),
            ]
        )
        self.features = nn.Sequential(*features)
        h = 14848  # Input dimension to the classifier
        # TODO: Make sure this is the correct dimension size.
        #       The number was found by working backwards from the output of the previous_layer.
        self.classifier = nn.Sequential(
            nn.Linear(in_features=h, out_features=d_out, bias=True),
            # The fully connected layer should be followed by a non-linearity
            nn.Softmax(1)
        )

    def forward(self, x):
        f = self.features(x)
        y = self.classifier(f)
        return y


def classify_image(img_path: str, model=None, pretrained_state_path: str = None):
    """
    Takes an image and returns the classification.
    """
    if model is None:
        if pretrained_state_path is None:
            model = models.vgg16(pretrained=True)
        else:
            state_dict = torch.load(pretrained_state_path)
            model = models.vgg16()
            model.load_state_dict(state_dict)
    img = preprocess_image(img_path)
    output = model(img)
    # Getting the max of the soft max layer.
    prediction = output.data.numpy().argmax()
    return labels[prediction]


if __name__ == '__main__':
    path = input("Path of the image to classify: \n")
    print("Predicted label from the original network: ", classify_image(path, None, None))
    net = VGGOcclusion()
    print("Predicted label from the modified network: ", classify_image(path, net, None))
