import torch
import torch.nn as nn
import torch.optim as optim
import dataHandling as dh
import matplotlib.pyplot as plt
import time


# Define a simple CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def evaluate_model(model, test_loader):
    model.eval()  # Θέστε το δίκτυο σε κατάσταση αξιολόγησης
    correct = 0
    total = 0
    
    with torch.no_grad():  # Απενεργοποίηση gradients για ταχύτερη εκτέλεση
        for images, labels in test_loader:
            outputs = model(images)  # Υπολογισμός εξόδων
            _, predicted = torch.max(outputs, 1)  # Βρείτε την κατηγορία με τη μεγαλύτερη πιθανότητα
            total += labels.size(0)  # Συνολικός αριθμός δειγμάτων
            correct += (predicted == labels).sum().item()  # Μετρήστε τις σωστές προβλέψεις
    
    accuracy = 100 * correct / total  # Υπολογισμός ακρίβειας
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
    print(f"Total training time: {minutes} minutes, {seconds:.2f} seconds.")
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

startNetwork(30, 32, 0.0005)

