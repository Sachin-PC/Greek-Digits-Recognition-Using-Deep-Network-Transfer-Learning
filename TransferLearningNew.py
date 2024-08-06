import cv2

# import statements
from MyNetwork import MyNetwork

import os
from PIL import Image
import torchvision.transforms.functional as TF
import sys
import torch.nn as nn
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import trange
import torch.optim as optim
import numpy as np
# Sachin Palahalli Chandrakumar
# code to implement transfer learning on added greek letters

# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# useful functions with a comment for each function
def train_network(model, optimizer, train_loader, log_interval, train_losses, epoch, train_counter, train_batch_size,
                  models_path):
    """
        Method to train the network
        Parameters:
        model : network model
        optimizer : optimizer used
        train_loader : training data loader
        log_interval: log_interval parameter
        train_losses: train loss list
        epoch : number of epochs
        train_counter: train counter used for plotting
        train_batch_size: train_batch size for training the model
        models_path: model save path
    """
    model.train()
    for i, (training_images, labels) in enumerate(train_loader):
        # Forward pass
        predicted_labels = model(training_images)
        training_loss = F.nll_loss(predicted_labels, labels)

        # Backward pass and loss
        optimizer.zero_grad()  # Emptying the gradient
        training_loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(training_images), len(train_loader.dataset),
                       100. * i / len(train_loader), training_loss.item()))
            train_losses.append(training_loss.item())
            train_counter.append(
                (i * train_batch_size) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(model.state_dict(), models_path + '/TransferedModel.pth')
            torch.save(optimizer.state_dict(), models_path + '/TransferedOptimizer.pth')
    return


# def test_network(model, test_loader):
#     model.eval()
#     with torch.no_grad():
#         for test_images, labels in test_loader:
#             predicted_labels = model(test_images)
#             predictions = predicted_labels.data.max(1, keepdim=True)[1]
#             print("predictions = ", predictions)

def test_network(model, test_loader):
    """
        Method to test the network
        Parameters:
        model : network model
        test_loader : testing data loader
    """
    model.eval()
    test_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for test_images, labels in test_loader:
            predicted_labels = model(test_images)
            test_loss += F.nll_loss(predicted_labels, labels, size_average=False).item()
            predictions = predicted_labels.data.max(1, keepdim=True)[1]
            correct_predictions += predictions.eq(labels.data.view_as(predictions)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct_predictions, len(test_loader.dataset),
        100. * correct_predictions / len(test_loader.dataset)))


def test_on_new_inputs(model, images_dir):
    """
        Method to test the network on new data
        Parameters:
        model : network model
        images_dir : new data directory
    """
    if not os.path.exists(images_dir):
        raise FileNotFoundError("The given Image directory: ", images_dir, " doesnt exist")

    plt.figure(figsize=(12, 8))
    index = 0
    greek_transform = GreekTransform()
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    labels = ["alpha", "beta", "gamma","eta","psi"]
    for image_name in os.listdir(images_dir):
        if image_name.endswith(".png") or image_name.endswith(".jpg"):
            image_path = os.path.join(images_dir, image_name)
            image = Image.open(image_path)
            # image = cv2.imread(image_path)
            # resized_image = cv2.resize(image, (128, 128))
            # greyscale_image = image.convert("L")
            # # greyscale_image = greyscale_image[greyscale_image < 200] = 0
            # image = Image.eval(greyscale_image, lambda x: 0 if x < 50 else 255)  # applying threshold
            resized_image = TF.resize(image, (128, 128))
            transformed_image = greek_transform(resized_image)
            image_tensor = transform(transformed_image).unsqueeze(0)
            with torch.no_grad():
                predicted_output = model(image_tensor)
                predicted_label = np.argmax(predicted_output)
                if index < 11:
                    plt.subplot(4, 3, index + 1)
                    plt.imshow(image_tensor.squeeze(), cmap='gray', interpolation='none')
                    plt.title("Prediction: {}".format(labels[predicted_label]))
                    plt.xticks([])
                    plt.yticks([])
                index += 1
    plt.show()


def evaluate_model(train_counter, train_losses):
    """
        plotting the results obtained
        Parameters:
        train_counter : train counter
        train_losses : train losses
    """
    plt.figure(figsize=(12, 8))
    plt.plot(train_counter, train_losses, color='blue')
    plt.legend(['Training Loss'])
    plt.xlabel('Training Examples')
    plt.ylabel('Loss')
    plt.show()

def visualize_training_loader(train_loader):
    """
        plotting the results obtained
        Parameters:
        train_loader: train_loader data
    """
    k=0
    for i, (training_images, labels) in enumerate(train_loader):
        print("training_images size = ",training_images.size()[0])
        for j in range(training_images.size()[0]):
            plt.subplot(9, 3, k + 1)
            plt.imshow(training_images[j].squeeze())
            # plt.title("Prediction: {}".format(k))
            k += 1

    plt.show()

# main method to implement transfer learning
if __name__ == "__main__":
    test_batch_size = 1000
    model = MyNetwork()
    model_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                 "Semester/PRCV/Projects/5/Project5/models/mainModel/model.pth"
    data_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                "Semester/PRCV/Projects/5/utils/data/"
    model.load_state_dict(torch.load(model_path))

    for param in model.parameters():
        param.requires_grad = False

    model.fc2 = nn.Linear(50, 5)

    print(model)
    training_set_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/5/Project5/data/greekLetters"
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(training_set_path,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
        batch_size=5,
        shuffle=True)

    train_losses = []
    test_losses = []
    train_counter = []
    n_epochs = 100
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    train_batch_size = 5
    models_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/5/" \
                  "Project5/models"
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # visualize_training_loader(greek_train)

    for epoch in range(1, n_epochs + 1):
        train_network(model, optimizer, greek_train, log_interval, train_losses, epoch, train_counter,
                      train_batch_size, models_path)
    print("Finished Training the network!")
    test_network(model, greek_train)
    # evaluate_model(train_counter, train_losses)

    # testing_set_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
    #                    "Semester/PRCV/Projects/5/Project5/data/greekLettersTest"
    #
    new_inputs_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                      "Semester/PRCV/Projects/5/Project5/data/greekLettersTest/t1"
    test_on_new_inputs(model, new_inputs_path)
    # test_network(model, greek_test)
    # print("greek_test = ", greek_test)
