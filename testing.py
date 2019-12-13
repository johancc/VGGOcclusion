import torch
from occlusion_data import TestDataset
from torch.utils.data import DataLoader
from torchvision import models
from torch import load
from network import labels as mapping
import os

cwd = os.getcwd() + "/"
statedict_path = os.getcwd() + "/vgg16-397923af.pth"


def get_test_loader(batch_size, dataset_path: str = "test_imgs", limit: int = -1):
    dataset = TestDataset(dataset_path, limit)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True, num_workers=1)
    return data_loader


def testOcclusion(model, batch_size, n):
    # Print all of the hyper parameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("=" * 30)
    # Model needs to be in testing mode
    model.eval()
    # Get testing data
    test_loader = get_test_loader(batch_size)

    correct = 0

    t = 0
    for images, labels in test_loader:
        for image, label in zip(images, labels):
            if n == t:
                return correct/n
            prediction = model(image).data.numpy().argmax()

            result = mapping[prediction]
            print(result)
            if("car" in result or "jeep" in result or "convertible" in result or "taxi" in result) \
                    and ("cart" not in result):
                correct += 1
            t += 1
    return correct/n


def testNoOcclusion(model, batch_size, n):
    # Print all of the hyper parameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("=" * 30)
    # Model needs to be in testing mode
    model.eval()
    # Get testing data
    test_loader = get_test_loader(batch_size, dataset_path="cars_test")

    correct = 0

    t = 0
    for images, labels in test_loader:
        for image, label in zip(images, labels):
            if n == t:
                return correct/n
            prediction = model(image).data.numpy().argmax()

            result = mapping[prediction]
            print(result)
            if("car" in result or "jeep" in result or "convertible" in result or "taxi" in result) and ("cart" not in result):
                correct += 1
            t += 1
    return correct/n


if __name__ == '__main__':
    from network import VGGOcclusion
    state_dict = load("output2-0.001-10.pth")
    model = VGGOcclusion() #"output2-0.001-10.pth")
    model.load_state_dict(state_dict)
    outcomeOcclusion = testOcclusion(model, 1, 500)
    outcomeNoOcclusion = testNoOcclusion(model, 1, 500)

    alexnetModel = models.alexnet(pretrained="True")
    alexnetOutcomeOcclusion = testOcclusion(alexnetModel, 1, 500)
    alexnetOutcomeNoOcclusion = testNoOcclusion(alexnetModel, 1, 500)

    resnetModel = models.resnet50(pretrained="True")
    resnetOutcomeOcclusion = testOcclusion(resnetModel, 1, 500)
    resnetOutcomeNoOcclusion = testNoOcclusion(resnetModel, 1, 500)

    state_dict = torch.load(statedict_path)
    vgg16 = models.vgg16()
    vgg16.load_state_dict(state_dict)
    vgg16OutcomeOcclusion = testOcclusion(vgg16, 1, 500)
    vgg16OutcomeNoOcclusion = testNoOcclusion(vgg16, 1, 500)

    print("Our network classified %.3f"%(outcomeOcclusion*100) + "% of the occluded images correctly.")
    print("Our network classified %.3f"%(outcomeNoOcclusion*100) + "% of the non-occluded images correctly.")
    print("VGG16 classified %.3f"%(vgg16OutcomeOcclusion*100) + "% of the occluded images correctly.")
    print("VGG16 classified %.3f"%(vgg16OutcomeNoOcclusion*100) + "% of the non-occluded images correctly.")
    print("AlexNet classified %.3f"%(alexnetOutcomeOcclusion*100) + "% of the occluded images correctly.")
    print("AlexNet classified %.3f"%(alexnetOutcomeNoOcclusion*100) + "% of the non-occluded images correctly.")
    print("ResNet classified %.3f"%(resnetOutcomeOcclusion*100) + "% of the occluded images correctly.")
    print("ResNet classified %.3f"%(resnetOutcomeNoOcclusion*100) + "% of the non-occluded images correctly.")