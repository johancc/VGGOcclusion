import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPyramidPooling(nn.Module):
    """
    Implementation of spatial pyramid pooling.
    """

    def __init__(self, dimension_levels):
        """
        :param dimension_levels The divisions to be made in the width and height dimension when pooling.
        """
        super(SpatialPyramidPooling, self).__init__()
        self.dimension_levels = dimension_levels

    def spatial_pyramid_pool(self, previous_conv):
        """
        Divides teh Tensor vertically and horizontally then
        performs max pooling on each division
        :param previous_conv input tensor of the previous conv layer.
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
        spp = None
        for i in range(len(self.dimension_levels)):
            h_kernel = int(math.ceil(previous_conv_size[0] / self.dimension_levels[i]))
            w_kernel = int(math.ceil(previous_conv_size[1] / self.dimension_levels[i]))
            w_pad1 = int(math.floor((w_kernel * self.dimension_levels[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(math.ceil((w_kernel * self.dimension_levels[i] - previous_conv_size[1]) / 2))
            h_pad1 = int(math.floor((h_kernel * self.dimension_levels[i] - previous_conv_size[0]) / 2))
            h_pad2 = int(math.ceil((h_kernel * self.dimension_levels[i] - previous_conv_size[0]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * self.dimension_levels[i] - previous_conv_size[1]) and \
                   h_pad1 + h_pad2 == (h_kernel * self.dimension_levels[i] - previous_conv_size[0])

            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                                 mode='constant', value=0)
            pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            x = pool(padded_input)
            if i == 0:
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)
        if spp is None:
            raise RuntimeError("Pyramid pooling failed.")
        return spp

    def forward(self, x):
        return self.spatial_pyramid_pool(x)

    def get_output_size(self, filters):
        sz = 0
        for level in self.dimension_levels:
            sz += filters * level ** 2
        return sz




