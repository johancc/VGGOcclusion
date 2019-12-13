"""
Utils needed to handle ImageNet data.
"""
import requests
import os
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
response = requests.get(LABELS_URL)  # Make an HTTP GET request and store the response.
labels = {int(key): value for key, value in response.json().items()}


def class_name_to_id(class_name: str) -> int:
    """
    :param class_name the name of the class
    :returns the index for the class name in the labels dict, None if not found.
    """
    for i, names in labels.items():
        # the value of the values contains multiple possible class names
        if "," in names:
            names = names.split(",")
        else:
            names = [names]
        for name in names:
            if name.strip() == class_name.strip():
                return i


def filepath_to_label(filepath: str)->int:
    """
    Finds the id of the image based on the filepath.
    Assumes that the name of the folder the image is contained in is the class name.
    i.e some/path/bird/123.jpg for class bird
    """
    _, folder_name = os.path.split(os.path.dirname(filepath))
    return class_name_to_id(folder_name)


def create_dataset_from_main_directory(directory: str) -> dict:
    """
    Makes a mapping between the filename and the class label.
    Expects the directory to contain sub folders where their name
    are the class name.
    """
    dataset = {}
    for (dirpath, _, filenames) in os.walk(directory):
        if dirpath == directory:
            continue
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            dataset[path] = filepath_to_label(path)
    return dataset


img_size = 224
transform_pipeline = transforms.Compose(
    [transforms.Resize((img_size, img_size)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])]
)


def preprocess_image(img_path: str):
    """
    The network needs the image to be scaled to 224 by 224.
    The image is normalized based on how the pre-trained model was trained.
    """
    img = Image.open(img_path)
    # Now we need to preprocess the image

    img = transform_pipeline(img)
    # Tensor Dims = (num_input_images,
    #               num_color_channels,
    #               height,
    #               width)
    img = img.unsqueeze(0)  # new first axis
    img = Variable(img)  # input should be a Variable type.

    return img