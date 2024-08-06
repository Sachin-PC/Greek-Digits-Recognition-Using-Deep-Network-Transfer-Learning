# Sachin Palahalli Chandrakumar
# Main Network Architecture and training the model

# import statements
import sys
import torch.nn as nn
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms


# My Network model class defining the architecture
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2_drop(self.conv2(x))))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


#
def train_network(model, optimizer, train_loader, log_interval, train_losses, epoch, train_counter, train_batch_size,models_path):
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
            torch.save(model.state_dict(), models_path+'/model.pth')
            torch.save(optimizer.state_dict(), models_path+'/optimizer.pth')
    print("Finished Training the network!")
    return


def test_network(model, test_losses, test_loader, accuracies):
    """
        Method to test the network
        Parameters:
        model : network model
        test_losses: test_losses list
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
    test_losses.append(test_loss)
    accuracies.append(100. * correct_predictions / len(test_loader.dataset))
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct_predictions, len(test_loader.dataset),
        100. * correct_predictions / len(test_loader.dataset)))


def visualize_data(data):
    """
        plotting of data
        Parameters:
        data : data
    """
    plt.figure(figsize=(12, 8))
    for index in range(6):
        plt.subplot(3, 2, index + 1)
        plt.imshow(data[index][0].squeeze(), cmap='gray', interpolation='none')
        plt.title("Label: {}".format(data[index][1]))
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
        test_losses: test_losses list
    """

    plt.figure(figsize=(12, 8))
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Training Loss', 'Testing Loss'])
    plt.xlabel('Training Examples')
    plt.ylabel('Loss')
    plt.show()


def evaluate_accuracy(test_counter, accuracies):
    plt.figure(figsize=(12, 8))
    plt.plot(test_counter, accuracies, color='blue')
    plt.legend(['Testing accuracy'])
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy')
    plt.show()


# main function
def main(argv):
    n_epochs = 5
    train_batch_size = 64
    test_batch_size = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    models_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/5/" \
                  "Project5/models/mainModel"

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

    model = MyNetwork()
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
    train_counter = []
    accuracies = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
    test_network(model, test_losses, test_loader, accuracies)
    for epoch in range(1, n_epochs + 1):
        train_network(model, optimizer, train_loader, log_interval, train_losses, epoch, train_counter,
                      train_batch_size, models_path)
        test_network(model, test_losses, test_loader, accuracies)

    evaluate_model(train_counter, train_losses, test_counter, test_losses)
    evaluate_accuracy(test_counter, accuracies)
    print("train_losses = ",train_losses)
    print("test_losses = ",test_losses)

    return


if __name__ == "__main__":
    main(sys.argv)
