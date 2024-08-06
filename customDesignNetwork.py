# Sachin Palahalli Chandrakumar
# Code to implement and analyze multiple custom netowrks

# import statements
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
from datetime import datetime
import pandas as pd


# class definitions
class CustomNetwork(nn.Module):
    def __init__(self, dropout_probability, conv2_output_features, fc1_output_features, kernel_size, image_width):
        super(CustomNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        conv2_inp_image_width = int((image_width - kernel_size + 1) / 2)
        self.conv2 = nn.Conv2d(10, conv2_output_features, kernel_size=kernel_size)
        self.conv2_drop = nn.Dropout2d(p=dropout_probability)
        fc1_inp_image_width = int((conv2_inp_image_width - kernel_size + 1) / 2)
        self.fc1_input_size = fc1_inp_image_width * fc1_inp_image_width * conv2_output_features
        self.fc1 = nn.Linear(self.fc1_input_size, fc1_output_features)
        self.fc2 = nn.Linear(fc1_output_features, 10)

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2_drop(self.conv2(x))))
        x = x.view(-1, self.fc1_input_size)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        return x


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
            torch.save(model.state_dict(), models_path + '/model.pth')
            torch.save(optimizer.state_dict(), models_path + '/optimizer.pth')
    print("Finished Training the network!")
    return


def test_network(model, test_losses, test_loader, accuracy):
    """
        Method to train the network
        Parameters:
        model : network model
        test_losses: train loss list
        test_loader: test_loader data
        accuracy : accuracy of the data predicted
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
    test_losses.append(test_loss)
    accuracy = 100. * correct_predictions / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct_predictions, len(test_loader.dataset),
        100. * correct_predictions / len(test_loader.dataset)))


def visualize_data(data):
    """
        plotting the results obtained
        Parameters:
        data: input data to be visualized
    """
    plt.figure(figsize=(12, 8))
    for index in range(6):
        plt.subplot(2, 3, index + 1)
        plt.imshow(data[index][0].squeeze(), cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(data[index][1]))
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


def evaluate_model(train_counter, train_losses, test_counter, test_losses):
    """
        plotting the results obtained
        Parameters:
        train_counter : train counter
        train_losses : train losses
        test_counter : test counter
        test_losses : test losses
    """
    plt.figure(figsize=(12, 8))
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Training Loss', 'Testing Loss'])
    plt.xlabel('Training Examples')
    plt.ylabel('Loss')
    plt.show()


def hyper_tuning_one(n_epochs, learning_rate, train_loader, log_interval, train_batch_size, models_path, test_loader):
    """
        Method to hypertune with one set of features
        Parameters:
        n_epochs : number of epochs
        learning_rate : learning rate
        train_loader : training data loader
        log_interval: log_interval parameter
        train_batch_size: train_batch size for training the model
        models_path: model save path
        test_loader: test_loade
    """
    # dropout_probabilities = [0.5]
    # conv2_output_features_list = [20]
    # fc1_output_features_list = [50]
    # kernel_size_list = [5]
    dropout_probabilities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    conv2_output_features_list = [20,40,60,80]
    fc1_output_features_list = [50]
    kernel_size_list = [3,5,7,9]
    image_width = 28
    best_test_loss = 1.0
    best_parameters = []
    each_model_train_loss = []
    each_model_test_loss = []
    each_model_parameters = []
    each_model_test_accuracy = []
    model_train_time = []
    models_summary_df = pd.DataFrame(
        columns=["dropout_probability", "conv2_output_features", "fc1_output_features", "kernel_size", "training_time",
                 "train_loss", "test_loss", "test_accuracy"])
    before_time = datetime.now()
    for dropout_probability in dropout_probabilities:
        for conv2_output_features in conv2_output_features_list:
            for fc1_output_features in fc1_output_features_list:
                for kernel_size in kernel_size_list:
                    each_model_parameters.append(
                        [dropout_probability, conv2_output_features, fc1_output_features, kernel_size])
                    model = CustomNetwork(dropout_probability, conv2_output_features, fc1_output_features, kernel_size,
                                          image_width)
                    # print(model)
                    # model = CustomNetwork()
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

                    train_losses = []
                    test_losses = []
                    train_counter = []
                    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
                    # test_network(model, test_losses, test_loader)
                    start_time = datetime.now()
                    test_accuracy = 0
                    for epoch in range(1, n_epochs + 1):
                        train_network(model, optimizer, train_loader, log_interval, train_losses, epoch, train_counter,
                                      train_batch_size, models_path)
                    test_network(model, test_losses, test_loader,test_accuracy)
                    end_time = datetime.now()
                    training_time = (end_time - start_time).total_seconds() / 60
                    model_train_time.append(training_time)
                    # print("train_losses = ",train_losses)
                    # print("test-loss = ",test_losses)
                    each_model_train_loss.append(train_losses[-1])
                    each_model_test_loss.append(test_losses[-1])
                    each_model_test_accuracy.append(test_accuracy)
                    row_data = [dropout_probability, conv2_output_features, fc1_output_features, kernel_size,
                                training_time, train_losses[-1], test_losses[-1],test_accuracy ]
                    # row_data = {'dropout_probability': dropout_probability,
                    #             'conv2_output_features': conv2_output_features,
                    #             'fc1_output_features': fc1_output_features,
                    #             'kernel_size': kernel_size,
                    #             'training_time': training_time,
                    #             'train_loss': train_losses[-1],
                    #             'test_loss': test_losses[-1]
                    #             }
                    models_summary_df = models_summary_df.append(pd.Series(row_data, index=models_summary_df.columns),
                                                                 ignore_index=True)
                    # models_summary_df.append(row_data, ignore_index=True)
    best_train_loss_index = np.argmin(each_model_train_loss)
    best_test_loss_index = np.argmin(each_model_test_loss)
    best_trained_model_parameters = each_model_parameters[best_train_loss_index]
    best_test_model_parameters = each_model_parameters[best_test_loss_index]

    model_summary_file_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                              "Semester/PRCV/Projects/5/Project5/Files/ht1_models_summary.csv "
    print("models_summary_df = ", models_summary_df)
    models_summary_df.to_csv(model_summary_file_path, index=False, encoding='utf-8-sig')
    after_time = datetime.now()
    print("Before time = ", before_time)
    print("After time = ", after_time)
    total_time = (after_time - before_time).total_seconds() / 60
    print("Total Time taken = ", total_time)
    print("model_train_time = ", model_train_time)

    print("each_model_train_loss = ", each_model_train_loss)
    print("each_model_test_loss = ", each_model_test_loss)
    print("Best Train loss = ", np.min(each_model_train_loss))
    print("Best Test loss = ", np.min(each_model_test_loss))
    print("best_trained_model_parameters = ", best_trained_model_parameters)
    print("best_test_model_parameters = ", best_test_model_parameters)

    return


def hyper_tuning_two(data_path, log_interval, models_path, test_loader):
    """
        Method to hypertune with another set of features
        Parameters:
        data_path : data path
        log_interval: log_interval parameter
        models_path: model save path
        test_loader: test_loade
    """
    n_epochs_list = [3,5,7]
    train_batch_size_list = [32, 64, 128, 256]
    learning_rate_list = [0.01, 0.05, 0.1]

    image_width = 28
    each_model_train_loss = []
    each_model_test_loss = []
    each_model_parameters = []
    model_train_time = []
    models_summary_df = pd.DataFrame(
        columns=["n_epochs", "train_batch_size", "learning_rate", "training_time",
                 "train_loss", "test_loss"])
    before_time = datetime.now()
    dropout_probability = 0.5
    conv2_output_features = 20
    fc1_output_features = 50
    kernel_size = 5
    for n_epochs in n_epochs_list:
        for train_batch_size in train_batch_size_list:
            for learning_rate in learning_rate_list:
                train_loader = torch.utils.data.DataLoader(
                    torchvision.datasets.MNIST(data_path, train=True, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       (0.1307,),
                                                       (0.3081,))
                                               ])),
                    batch_size=train_batch_size, shuffle=True)
                each_model_parameters.append(
                    [dropout_probability, conv2_output_features, fc1_output_features, kernel_size])
                model = CustomNetwork(dropout_probability, conv2_output_features, fc1_output_features, kernel_size,
                                      image_width)
                # print(model)
                # model = CustomNetwork()
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

                train_losses = []
                test_losses = []
                train_counter = []
                test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
                # test_network(model, test_losses, test_loader)
                start_time = datetime.now()
                for epoch in range(1, n_epochs + 1):
                    train_network(model, optimizer, train_loader, log_interval, train_losses, epoch, train_counter,
                                  train_batch_size, models_path)
                test_network(model, test_losses, test_loader)
                end_time = datetime.now()
                training_time = (end_time - start_time).total_seconds() / 60
                model_train_time.append(training_time)
                # print("train_losses = ",train_losses)
                # print("test-loss = ",test_losses)
                each_model_train_loss.append(train_losses[-1])
                each_model_test_loss.append(test_losses[-1])
                row_data = [n_epochs, train_batch_size, learning_rate,
                            training_time, train_losses[-1], test_losses[-1]]
                models_summary_df = models_summary_df.append(pd.Series(row_data, index=models_summary_df.columns),
                                                             ignore_index=True)
                # models_summary_df.append(row_data, ignore_index=True)
    best_train_loss_index = np.argmin(each_model_train_loss)
    best_test_loss_index = np.argmin(each_model_test_loss)
    best_trained_model_parameters = each_model_parameters[best_train_loss_index]
    best_test_model_parameters = each_model_parameters[best_test_loss_index]

    model_summary_file_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                              "Semester/PRCV/Projects/5/Project5/Files/ht2_models_summary.csv "
    print("models_summary_df = ", models_summary_df)
    models_summary_df.to_csv(model_summary_file_path, index=False, encoding='utf-8-sig')
    after_time = datetime.now()
    print("Before time = ", before_time)
    print("After time = ", after_time)
    total_time = (after_time - before_time).total_seconds() / 60
    print("Total Time taken = ", total_time)
    print("model_train_time = ", model_train_time)

    print("each_model_train_loss = ", each_model_train_loss)
    print("each_model_test_loss = ", each_model_test_loss)
    print("Best Train loss = ", np.min(each_model_train_loss))
    print("Best Test loss = ", np.min(each_model_test_loss))
    print("best_trained_model_parameters = ", best_trained_model_parameters)
    print("best_test_model_parameters = ", best_test_model_parameters)

    return


# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv
    n_epochs = 5
    train_batch_size = 64
    test_batch_size = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    models_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/5/" \
                  "Project5/models"

    RANDOM_SEED = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(RANDOM_SEED)

    data_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                "Semester/PRCV/Projects/5/utils/data/ "

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_data = datasets.MNIST('data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(data_path, train=True, download=True,
                                                                          transform=torchvision.transforms.Compose([
                                                                              torchvision.transforms.ToTensor(),
                                                                              torchvision.transforms.Normalize(
                                                                                  (0.1307,),
                                                                                  (0.3081,))
                                                                          ])),
                                               batch_size=train_batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(data_path, train=True, download=True,
                                                                         transform=torchvision.transforms.Compose([
                                                                             torchvision.transforms.ToTensor(),
                                                                             torchvision.transforms.Normalize((0.1307,),
                                                                                                              (0.3081,))
                                                                         ])),
                                              batch_size=test_batch_size, shuffle=True)

    # visualize_data(test_data)
    hyper_tuning_one(n_epochs, learning_rate, train_loader, log_interval, train_batch_size, models_path, test_loader)

    # hyper_tuning_two(data_path, log_interval, models_path, test_loader)

    # dropout_probabilities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # conv2_output_features_list = [20,40,60,80]
    # fc1_output_features_list = [50]
    # kernel_size_list = [3,5,7,9]

    return


if __name__ == "__main__":
    main(sys.argv)
