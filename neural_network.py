import torch
import torch.nn as nn
import torchaudio


class CNNFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        layers: int = 4,
    ):
        """
        Initializes the CNN Feature Extractor.

        Parameters
        ----------
        input_channels : int
            The number of input channels in the given tensor.

        output_channels : int
            The number of channels that should be outputted.

        kernel_size : int, optional
            The size of the kernel, by default 3.

        stride : int, optional
            The stride of the kernel, by default 1.

        layers : int, optional
            The number of CNN layers to have, by default 4.
        """
        super().__init__()

        # Defining our model
        self.CNN_layers = nn.ModuleList()

        for _ in range(layers):
            CNN_layer = nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )

            # Adding the layers to the CNN block
            self.CNN_layers.append(CNN_layer)
            self.CNN_layers.append(nn.BatchNorm2d(num_features=output_channels))
            self.CNN_layers.append(nn.ReLU(inplace=True))

            # Updating the future input values
            input_channels = output_channels

    def forward(self, X):
        """
        Processes the input through the CNN layers.

        Parameters
        ----------
        X : torch.Tensor
            A passed-in torch tensor.

        Returns
        -------
        torch.Tensor
            The processed tensor that is to be returned.
        """
        for layer in self.CNN_layers:
            X = layer(X)

        return X


class SentimentModel(nn.Module):
    """Combines CNNFeatureExtractor with an LSTM and ends with softmax!"""

    def __init__(
        self,
        cnn_in_channels,
        cnn_out_channels,
        cnn_kernel_size=3,
        lstm_hidden_size=1024,
        num_classes=8,
    ):
        super().__init__()

        # Defining our CNN Feature Extractor
        self.cnn_feature_extractor = CNNFeatureExtractor(
            input_channels=cnn_in_channels,
            output_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size,
        )

        lstm_input_size = 40 * 1251

        # Now we define our LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Final linear layer to output num_classes
        self.fcn = nn.Linear(lstm_hidden_size, num_classes)

        # Defining softmax to classify our values
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        """
        Forward pass through the SentimentModel.

        Parameters
        ----------
        X : torch.Tensor
            A passed-in torch tensor.

        Returns
        -------
        torch.Tensor
            The processed tensor that is to be returned.
        """
        # Forward pass through CNN Feature Extractor
        X = self.cnn_feature_extractor(X)

        # Flattening out our model
        batch_size = X.size(0)

        X = X.view(batch_size, X.shape[1], -1)

        # Forward pass through LSTM
        _, (hn, _) = self.lstm(X)
        X = hn[-1]

        # Passing through the final fully connected layer
        X = self.fcn(X)

        # Finally, run it through softmax
        X = self.softmax(X)

        return X
