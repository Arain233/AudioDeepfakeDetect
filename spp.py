import math
import torch
import torch.nn.functional as F


# spp realize
class SPPLayer(torch.nn.Module):
    def __init__(self, num_levels):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels

    def forward(self, x):
        num, c, h, w = x.size()
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            padding = (math.floor((kernel_size[0] * level - h + 1) / 2),
                       math.floor((kernel_size[1] * level - w + 1) / 2))
            tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding).view(num, -1)
            if i == 0:
                result = tensor.view(num, -1)
            else:
                result = torch.cat((result, tensor.view(num, -1)), 1)
        return result
