import torch

image_channels = torch.randn(32,3,300,300)


output_channels = [128,256,128,128,64,64]

features = torch.nn.Sequential(
    torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(inplace=True),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),

    torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(inplace=True),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),

    torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(inplace=True),#75

    torch.nn.Conv2d(64, output_channels[0], kernel_size=3, stride=2, padding=1),#8
    torch.nn.ReLU(inplace=True),#38

    torch.nn.Conv2d(output_channels[0], 128, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(inplace=True),

    torch.nn.Conv2d(128, output_channels[1], kernel_size=3, stride=2, padding=1),#12
    torch.nn.ReLU(inplace=True),#19

    torch.nn.Conv2d(output_channels[1], 256, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(inplace=True),

    torch.nn.Conv2d(256, output_channels[2], kernel_size=3, stride=2, padding=1),#16
    torch.nn.ReLU(inplace=True), #10

    torch.nn.Conv2d(output_channels[2], 128, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(inplace=True),

    torch.nn.Conv2d(128, output_channels[3], kernel_size=3, stride=2, padding=1),#20
    torch.nn.ReLU(inplace=True), #5

    torch.nn.Conv2d(output_channels[3], 128, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(inplace=True),

    torch.nn.Conv2d(128, output_channels[4], kernel_size=3, stride=2, padding=1),#24
    torch.nn.ReLU(inplace=True), #3

    torch.nn.Conv2d(output_channels[4], 128, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(inplace=True),

    torch.nn.Conv2d(128, output_channels[5], kernel_size=3, stride=1, padding=0),#28
    torch.nn.ReLU(inplace=True),#1

    )

o = features(image_channels)
print(o.shape)


x = image_channels
out_channels = output_channels
out_features = []
idx = 0

for i, layer in enumerate(features):
    x = layer(x)
    print("i: ",i) #0..29 One for each operation
    print(x.shape)
    if i in [8,12,16,20,24,28]:
        out_features.append(x)


    # Check if the layer is a Conv2D layer with specified output channels
    if isinstance(layer, torch.nn.Conv2d):# and layer.out_channels == out_channels[idx]:
        print(i," is conv layer")
        print(layer.out_channels)
        print("\n")
    #    out_features.append(x)
    #    idx += 1
        #if idx >= len(out_channels):
        #    break

# Ensure that the output shapes are as expected
for idx, feature in enumerate(out_features):
    print(idx, ": ",feature.shape)


"""
[shape(-1, output_channels[0], 38, 38),
shape(-1, output_channels[1], 19, 19),
shape(-1, output_channels[2], 10, 10),
shape(-1, output_channels[3], 5, 5),
shape(-1, output_channels[3], 3, 3),
shape(-1, output_channels[4], 1, 1)]
"""
