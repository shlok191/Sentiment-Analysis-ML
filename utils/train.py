import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from layers import EmotionDetector
from dataset import RavDessDataset

# Defining our hyperparameters
cnn_input_channels = 1
cnn_hidden_channels = 32
cnn_output_channels = 64
cnn_kernel_size = 3

lstm_input_size = 32
lstm_hidden_size = 128
lstm_sequence_length = 11856
lstm_output_size = 512  # Intermediate output feature size

FCL_output_size = 8 # Number of classes

batch_size = 32
learning_rate = 0.001
num_epochs = 10

def get_trained_model():

    # Create instances of the model, dataset, dataloaders, and loss function
    model = EmotionDetector(
        cnn_input_channels,
        cnn_hidden_channels,
        cnn_output_channels,
        cnn_kernel_size,
        lstm_input_size,
        lstm_sequence_length,
        lstm_hidden_size,
        lstm_output_size,
        FCL_output_size
    )

    train_dataset = RavDessDataset('./data/', train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train() 
        total_loss = 0.0

        for batch in train_dataloader:

            inputs = batch[0]
            labels = batch[1]
            
            labels -= 1 # Making labels 0-indexed!

            #print(f"Labels: {labels}")

            optimizer.zero_grad()
            outputs = model(inputs)
            
            #print(f"Outputs: {outputs}")

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

    return model

if __name__ == "__main__":
    model = get_trained_model()