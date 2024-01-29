import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from train import get_trained_model
from dataset import RavDessDataset

# Defining our hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Getting our trained model
model = get_trained_model()

model.eval()
criterion = nn.CrossEntropyLoss()

# Obtaining the test data loader
test_dataloader = DataLoader(RavDessDataset(train=False), batch_size=batch_size, shuffle=False)

y_pred = []
y_true = []

# Evaluation of the model proceeds from HERE:

with torch.no_grad():

    for batch in test_dataloader:
        inputs, labels = batch

        # Obtaining our values
        outputs = model(inputs)
        
        # Computing our loss with CrossEntropyModel()

        loss = criterion(outputs, labels)
        
        _, predictions = torch.max(outputs, 1)

        y_pred.extend(predictions.cpu().numpy())
        y_true.extend(labels.cpu().numpy())


# Calculating the metrics        
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')


print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
