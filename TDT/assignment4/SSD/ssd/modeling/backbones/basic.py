import torch
from typing import Tuple, List


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(64, output_channels[0], kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(output_channels[0], 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(128, output_channels[1], kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(output_channels[1], 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(256, output_channels[2], kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(output_channels[2], 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(128, output_channels[3], kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(output_channels[3], 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(128, output_channels[4], kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(output_channels[4], 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(128, output_channels[5], kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            )





    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        idx = 0

        for i, layer in enumerate(self.features):
            x = layer(x)
            #print("i: ",i) #0..29 One for each operation
            #print(x.shape)
            if i in [8,12,16,20,24,28]:
                out_features.append(x)



        # Ensure that the output shapes are as expected
        #for idx, feature in enumerate(out_features):
            #print(idx,": ",feature.shape[1:])
            #print(idx, ": ",feature.shape)

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"

        assert len(out_features) == len(self.output_feature_shape),\
            f"GG Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)
