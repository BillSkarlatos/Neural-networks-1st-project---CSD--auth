import torch
import torch.nn as nn
import torch.optim as optim
import dataHandling as dh
import matplotlib.pyplot as plt
import time
import numpy as np


# Define a simple CNN
class CNN(nn.Module):   # We will test it for at least 2 combination of neurons. Look in the next comments for the low and high proposed values [ {low} - {high} Neurons/Filters]
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)  # 32 - 64 Filters
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1) # 64 - 128 Filters
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256) # 128 - 256 Neurons
        self.fc2 = nn.Linear(256, 10) # Output layer for the 10 categories.
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def evaluate_model(model, test_loader):
    model.eval()  # Sets the model to evaluation mode: disables normalisation and ensures stable function for testing.
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradients for faster execution.
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def startNetwork(epochs, batch_size, learning_rate):
    # Initialize the model, loss function, and optimizer
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, test_loader = dh.data_loader(batch_size)

    print("Starting training...")
    total_start=time.time()
    # Training loop
    loss_per_epoch = []
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_start=time.time()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        loss_per_epoch.append(average_loss)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds. Loss: {average_loss:.16f}")

    total_time = time.time() - total_start
    minutes= total_time//60
    seconds= total_time - minutes*60
    print("Training complete.")
    print(f"Total training time: {int(minutes)} minutes, {seconds:.2f} seconds.")
    print("Evaluating model...")
    acc=evaluate_model(model,test_loader)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_per_epoch, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Training loss per Epoch, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    plt.legend()
    plt.grid()
    plt.show()
    
    #examples(model, test_loader) # 5 Specific Examples, uncomment to check or check the project report

def imshow(img, title):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')

def examples(model,test_loader):
    model.eval()
    examples = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    examples.append(('correct', images[i], labels[i], predicted[i]))
                else:
                    examples.append(('incorrect', images[i], labels[i], predicted[i]))

    correct_examples = [ex for ex in examples if ex[0] == 'correct'][:5]
    incorrect_examples = [ex for ex in examples if ex[0] == 'incorrect'][:5]

    # Showing the results : Recommended by Chat GPT

    # Correct
    plt.figure(figsize=(10, 5))
    for i, (status, image, label, predicted) in enumerate(correct_examples):
        plt.subplot(1, 5, i + 1)
        imshow(image, f"Label: {label}, Pred: {predicted}")
    plt.suptitle("Correct Predictions")
    plt.show()

    # Incorrect
    plt.figure(figsize=(10, 5))
    for i, (status, image, label, predicted) in enumerate(incorrect_examples):
        plt.subplot(1, 5, i + 1)
        imshow(image, f"Label: {label}, Pred: {predicted}")
    plt.suptitle("Incorrect Predictions")
    plt.show()

