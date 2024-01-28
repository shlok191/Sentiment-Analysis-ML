import torch
import torch.nn as nn
import torchaudio as audio
import os
from torch.utils.data import Dataset, DataLoader

class RavDessDataset(Dataset):

    """ A custom dataset class to load in audio files situated in the RAVDESS
    dataset!"""

    def __init__(self, data_dir="./data/dataset.zip"):

        

    