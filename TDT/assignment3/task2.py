import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn, flatten
from dataloaders import load_cifar10
from trainer import Trainer
import numpy as np
import time

class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=num_filters, kernel_size=5,stride=1,padding=2),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride = 2),
            
            nn.Conv2d(in_channels=num_filters,    out_channels=64         , kernel_size=5,stride=1,padding=2),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride = 2),
           
            nn.Conv2d(in_channels=64,             out_channels=128         , kernel_size=5,stride=1,padding=2),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride = 2)
        )

        self.feature_extractor2 = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=num_filters, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        #self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        #self.relu = nn.ReLU()
        #self.fc1 = nn.Linear(64*128*4*4, 64)
        #self.fc2 = nn.Linear(64, num_classes)

        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 32*32*32

        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            #nn.Linear(self.num_output_features, num_classes),
            nn.Linear(2048, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(self.num_output_features, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
   
        #print("x.shape: ", x.shape) #([64, 3, 32, 32])
        batch_size = x.shape[0]

        out = self.feature_extractor(x) 
        out = out.view(batch_size, -1)
    
        out = self.classifier(out)
        


        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

def output_features(num_filters1, num_filters2, num_filters3, num_filters4, kernel_conv, stride_conv, padding_conv,
          kernel_pool, stride_pool):
    
        out_channels = num_filters1
        out_1 = np.floor((32 + 2*padding_conv - kernel_conv) / stride_conv) + 1
        out_shape1 = (out_channels, out_1, out_1)
            

        out_2 = np.floor((out_1 - kernel_pool) / stride_pool) + 1
        out_shape2 = (out_channels, out_2, out_2)
        
        out_channels = num_filters2
        out_1 = np.floor((out_2 + 2*padding_conv - kernel_conv) / stride_conv) + 1
        out_shape1 = (out_channels, out_1, out_1)
            

        out_2 = np.floor((out_1 - kernel_pool) / stride_pool) + 1
        out_shape2 = (out_channels, out_2, out_2)
        
        
        out_channels = num_filters3
        out_1 = np.floor((out_2 + 2*padding_conv - kernel_conv) / stride_conv) + 1
        out_shape1 = (out_channels, out_1, out_1)
            

        out_2 = np.floor((out_1 - kernel_pool) / stride_pool) + 1
        out_shape2 = (out_channels, out_2, out_2)

        out_channels = num_filters4
        out_1 = np.floor((out_2 + 2*padding_conv - kernel_conv) / stride_conv) + 1
        out_shape1 = (out_channels, out_1, out_1)
            

        out_2 = np.floor((out_1 - kernel_pool) / stride_pool) + 1
        out_shape2 = (out_channels, out_2, out_2)
        
        final_output_features = out_channels * out_2 * out_2


        return int(final_output_features)
    
def main():
    start_time = time.time()
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "task2")
    total_time = time.time() - start_time
    print("time: ", total_time)

if __name__ == "__main__":
    main()