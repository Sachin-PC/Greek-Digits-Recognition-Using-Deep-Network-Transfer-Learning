# Sachin Palahalli Chandrakumar
# Code to analyze the network built


from MyNetwork import MyNetwork
from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import numpy as np
import os


def test_network(model, test_loader):
    """
        Method to test the network
        Parameters:
        model : network model
        test_loader : test data loader
    """
    model.eval()
    plt.figure(figsize=(12, 8))
    with torch.no_grad():
        for test_images, labels in test_loader:
            predicted_outputs = model(test_images)
            for index in range(10):
                print("\nImage ", index, ": \nNetwork Output Values = ", np.round(predicted_outputs[index], 2))
                print("Max Value Index = ", np.argmax(predicted_outputs[index]))
                print("True Label = ", labels[index])
                if index != 9:
                    plt.subplot(3, 3, index + 1)
                    plt.imshow(test_images[index].squeeze(), cmap='gray', interpolation='none')
                    plt.title("Predicted Label: {}".format(labels[index]))
                    plt.xticks([])
                    plt.yticks([])
            plt.show()
            break



def test_on_new_inputs(model, images_dir):
    """
        Method to test the network on new data
        Parameters:
        model : network model
        images_dir : new data directory
    """

    print("model weights = ",model.conv1.weight)
    if not os.path.exists(images_dir):
        raise FileNotFoundError("The given Image directory: ", images_dir, " doesnt exist")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    plt.figure(figsize=(12, 8))
    index = 0
    model.eval()
    for image_name in os.listdir(images_dir):
        if image_name.endswith(".png") or image_name.endswith(".jpg"):
            image_path = os.path.join(images_dir, image_name)
            image = Image.open(image_path)
            greyscale_image = image.convert("L")
            # greyscale_image = greyscale_image[greyscale_image < 200] = 0
            greyscale_image = Image.eval(greyscale_image, lambda x: 0 if x < 100 else 255) # applying threshold
            inverted_image = Image.eval(greyscale_image, lambda x: 255 - x)
            resized_image = TF.resize(inverted_image, (28, 28))
            # resized_greyscale_image = resized_image.convert("L")
            image_tensor = transform(resized_image).unsqueeze(
                0)  # unsqueeze(0) is added to provide extra dimension for batch size
            print("image_tensor shape = ",image_tensor.shape)
            with torch.no_grad():
                predicted_output = model(image_tensor)
                predicted_label = np.argmax(predicted_output)
                if index < 12:
                    plt.subplot(4, 3, index + 1)
                    plt.imshow(image_tensor.squeeze(), cmap='gray', interpolation='none')
                    plt.title("Prediction: {}".format(predicted_label))
                    plt.xticks([])
                    plt.yticks([])
                index +=1
    plt.show()


# main model to perform the testing
if __name__ == "__main__":
    test_batch_size = 1000
    model = MyNetwork()
    model_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                 "Semester/PRCV/Projects/5/Project5/models/mainModel/model.pth"
    data_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                "Semester/PRCV/Projects/5/utils/data/"
    model.load_state_dict(torch.load(model_path))

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_data = datasets.MNIST('data', train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(data_path, train=True, download=True,
                                                                         transform=torchvision.transforms.Compose([
                                                                             torchvision.transforms.ToTensor(),
                                                                             torchvision.transforms.Normalize((0.1307,),
                                                                                                              (0.3081,))
                                                                         ])),
                                              batch_size=test_batch_size, shuffle=True)
    # test_network(model, test_loader)

    new_inputs_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                      "Semester/PRCV/Projects/5/Project5/data/testDigits/t1"
    test_on_new_inputs(model, new_inputs_path)
