import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
import os


class CNNFeatureExtractor(nn.Module):

    """ First step of sentiment detection in our CNN: feature extraction
        of information from audio file represented as waveforms """
    
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, 
                 kernel_size: int = 3, stride: int = 1):

        # Initializing the parent nn.Module!
        super().__init__()

        # Defining the required layers
        self.conv_layer = nn.Conv2d(input_channels, hidden_channels, kernel_size, stride)
        self.batch_layer = nn.BatchNorm2d(hidden_channels, output_channels)

        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):

        # Initial adaptive 1D Convolution layer
        x = self.conv_layer(x)
        
        # Second is the batch normalization layer
        x = self.batch_layer(x)
        
        # ELU function to introduce linearity
        x = self.elu(x)

        # Pooling operation 
        x = self.pool(x)

        return x

class LSTMClassifier(nn.Module):

    def __init__(self, input_size, sequence_length, hidden_size, output_size):

        # Initializing the parent nn.Module!
        super().__init__()
        
        # Defining the required layers

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fcl = nn.Linear(hidden_size * sequence_length, output_size)

    def forward(self, x):

        # Passes through the input tensor to our model!
        x, _ = self.lstm(x)
        x = x.reshape(32, -1)

        x = self.fcl(x)

        return x

class EmotionDetector(nn.Module):

    """ The final combined model allows us to utilize the CNNs and LSTM models to generate emotion probabilities!"""

    def __init__(self, cnn_input_channels, cnn_hidden_channels, cnn_output_channels, cnn_kernel_size,
                 lstm_input_size, lstm_sequence_length, lstm_hidden_size, lstm_output_size, fcl_output_size):
        
        super().__init__()

        # CNN Feature Extractor

        self.cnn_feature_extractor = CNNFeatureExtractor(
            input_channels=cnn_input_channels,
            hidden_channels=cnn_hidden_channels,
            output_channels=cnn_output_channels,
            kernel_size=cnn_kernel_size,
        )

        # LSTM Classifier

        self.lstm_classifier = LSTMClassifier(
            input_size=lstm_input_size,
            sequence_length=lstm_sequence_length,
            hidden_size=lstm_hidden_size,
            output_size=lstm_output_size
        )

        # Defining the final Fully Connected Layer
        self.FCL = nn.Linear(lstm_output_size, fcl_output_size)

        # Defining softMax to classify our values!
        self.softmax = nn.Softmax(1)

    def forward(self, x):

        
        # Forward pass through CNN Feature Extractor
        cnn_output = self.cnn_feature_extractor(x)

        # Reshape or flatten the output for LSTM
        batch_size, features, _, _ = cnn_output.size()
        
        lstm_input = cnn_output.view(batch_size, -1, features)

        # Forward pass through LSTM Classifier
        lstm_output = self.lstm_classifier(lstm_input)
        
        FCL_output = self.FCL(lstm_output)
        softmax_output = self.softmax(FCL_output)

        return softmax_output
