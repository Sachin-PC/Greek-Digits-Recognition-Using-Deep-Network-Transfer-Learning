# Sachin Palahalli Chandrakumar
# Code to analyze model structure and its effects
import cv2
from MyNetwork import MyNetwork
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def display_filters(layer_weights):
    """
        Method to display the filters
        Parameters:
        layer_weights : layer_weights of model
    """
    plt.figure(figsize=(12, 8))
    with torch.no_grad():
        for i in range(10):
            print("Filter ", i, " weight = \n", layer_weights[i, 0])
            print("Filter ", i, " shape = ", layer_weights[i, 0].shape)
            plt.subplot(3, 4, i + 1)
            plt.imshow(layer_weights[i, 0].detach().numpy())
            plt.title("Filter: {}".format(i + 1))
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()


def display_filter_effects(layer_weights, image):
    """
        Method to display the filters effects on the input image
        Parameters:
        layer_weights : layer_weights of model
        image : input image on which these filters will be applied
    """
    plt.figure(figsize=(12, 8))
    j = 1
    with torch.no_grad():
        for i in range(10):
            filter_i = layer_weights[i, 0].unsqueeze(0)
            filtered_image = cv2.filter2D(image.squeeze().numpy(), -1, filter_i.squeeze().numpy())
            plt.subplot(5, 4, j)
            plt.imshow(layer_weights[i, 0].detach().numpy(), cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title("Filter: {}".format(i+1))
            j += 1
            plt.subplot(5, 4, j)
            plt.imshow(filtered_image,cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title("Filter: {} effect".format(i+1))
            j += 1
    plt.show()

# main method containing the logic to implement analyzing filter effect
if __name__ == "__main__":
    test_batch_size = 1000
    model = MyNetwork()
    model_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                 "Semester/PRCV/Projects/5/Project5/models/model.pth"
    data_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                "Semester/PRCV/Projects/5/utils/data/"
    model.load_state_dict(torch.load(model_path))

    print(model)

    first_layer_weights = model.conv1.weight
    print("First Layer Weights = \n", first_layer_weights)
    print("First Layer Weights shape= \n", first_layer_weights.shape)

    # display_filters(first_layer_weights)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    display_filter_effects(first_layer_weights, test_data[0][0])
