import torch
from occlusion_data import TestDataset
from torch.utils.data import DataLoader
from torchvision import models
from torch import load
from network import labels as mapping
import os


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
            # print("got classified as {}, correct label {}".format(result, mapping[int(label.item())]))
            if("car" in result or "jeep" in result or "convertible" in result or "taxi" in result) \
                    and ("cart" not in result):
                correct += 1
            t += 1
            if t % 100 == 0:
                print("{}/{} tested.".format(t, n))
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
            # print("got classified as {}, correct label {}".format(result, mapping[int(label.item())]))
            if("car" in result or "jeep" in result or "convertible" in result or "taxi" in result) and ("cart" not in result):
                correct += 1
            t += 1
            if t % 100 == 0:
                print("{}/{} tested.".format(t, n))
    return correct/n


def test_local_models(model_dir: str = "models/"):
    files = os.listdir(model_dir)
    for file in files:
        file = os.path.join(model_dir, file)
        if ".pth" in file:
            print("Testing {}".format(file))
            state_dict = load(file)
            model = VGGOcclusion()
            model.load_state_dict(state_dict)
            outcomeOcclusion = testOcclusion(model, 1, 500)
            outcomeNoOcclusion = testNoOcclusion(model, 1, 500)
            print("Results: {} occlusion, {} no occlusion".format(outcomeOcclusion, outcomeNoOcclusion))


if __name__ == '__main__':
    from network import VGGOcclusion
    test_local_models()

    alexnetModel = models.alexnet(pretrained="True")
    alexnetOutcomeOcclusion = testOcclusion(alexnetModel, 1, 500)
    alexnetOutcomeNoOcclusion = testNoOcclusion(alexnetModel, 1, 500)

    resnetModel = models.resnet50(pretrained="True")
    resnetOutcomeOcclusion = testOcclusion(resnetModel, 1, 500)
    resnetOutcomeNoOcclusion = testNoOcclusion(resnetModel, 1, 500)

    print("AlexNet classified %.3f"%(alexnetOutcomeOcclusion*100) + "% of the occluded images correctly.")
    print("AlexNet classified %.3f"%(alexnetOutcomeNoOcclusion*100) + "% of the non-occluded images correctly.")
    print("ResNet classified %.3f"%(resnetOutcomeOcclusion*100) + "% of the occluded images correctly.")
    print("ResNet classified %.3f"%(resnetOutcomeNoOcclusion*100) + "% of the non-occluded images correctly.")