import torch.nn as nn

class Adjust_contrast(nn.Module):
    def __init__(self, factor):
        super(Adjust_contrast, self).__init__()
        self.factor = factor

    def forward(self, noised_and_cover):
        return noised_and_cover