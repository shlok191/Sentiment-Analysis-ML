import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RAVDESSDataset
from neural_network import SentimentModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Defining our hyperparameters
cnn_input_channels = 1
cnn_output_channels = 64
cnn_kernel_size = 3

lstm_input_size = 32
lstm_hidden_size = 1024

FCL_output_size = 8  # Number of classes

batch_size = 16
learning_rate = 0.001
num_epochs = 10
pin_memory = True
checkpoint_directory = "checkpoints"

scaler = GradScaler()


def train_model(model, optimizer, loss_fn, data_loader, epoch):
    # Set the model on train mode
    model.train()
    running_loss = 0.0
    loop = tqdm(data_loader, leave=True)
    for batch_idx, (audio, category) in enumerate(loop):
        audio = audio.to(device=device)
        category = category.to(device=device)
        # Zero out the gradient
        optimizer.zero_grad()
        with autocast():
            predictions = model(audio)
            loss = loss_fn(predictions, category)
        scaler.scale(loss).backward()
        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (batch_idx + 1))
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    # Save the checkpoint
    os.makedirs(checkpoint_directory, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_directory, f"model_epoch_{epoch}.pth"
    )
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":

    dataset = RAVDESSDataset()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )

    model = SentimentModel(cnn_in_channels=1, cnn_out_channels=64).to(
        device=device
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_model(model, optimizer, loss_fn, loader, epoch)
