"""
Processes the cars_train data into a training set and test set.
"""
import os
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data

car_index = 199  # TODO: Find the correct index


class CarDataset(data.Dataset):
    """
    Dataset containing various images of cars
    """
    def __init__(self, data_root, limit=-1):
        self.samples = []
        self.data_root = data_root
        self._init_dataset(limit)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        return preprocess_image(img_path), label

    def _init_dataset(self, limit: int):
        """
        Creates a mapping between image path and the
        corresponding label.
        The images are not loaded since it would be memory
        inefficient.
        """
        for filename in os.listdir(self.data_root):
            car_path = os.path.join(self.data_root, filename)
            label = 199.0
            self.samples.append([car_path, label])
            if len(self.samples) >= limit != -1:
                break


def preprocess_image(img_path: str):
    """
    The network needs the image to be scaled to 224 by 224.
    The image is normalized based on how the pre-trained model was trained.
    """
    img = Image.open(img_path)
    # Now we need to preprocess the image
    img_size = 224
    transform_pipeline = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    )
    img = transform_pipeline(img)
    # Tensor Dims = (num_input_images,
    #               num_color_channels,
    #               height,
    #               width)
    img = img.unsqueeze(0)  # new first axis
    img = Variable(img)  # input should be a Variable type.

    return img


if __name__ == '__main__':
    # Loading data example
    from torch.utils.data import DataLoader
    dataset = CarDataset("cars_train")
    data_loader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=1)

