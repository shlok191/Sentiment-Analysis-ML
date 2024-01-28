import torch
import torch.nn as nn
import torchaudio as audio
import os
from torch.utils.data import Dataset, DataLoader

class RavDessDataset(Dataset):

    """ A custom dataset class to load in audio files situated in the RAVDESS
    dataset!"""

    def __init__(self):

        # Initializes all important variables

        self.loc = "../data/"
        self.files = self._get_wav_files(self.loc)
        
        self.transform = audio.transforms.MFCC()    

    def _get_wav_files(self, location: str):
        
        # Recursively search and find all .wav files from
        # the root directory of the dataset!

        file_collections = []
        
        for root, _, files in os.walk(location):

            for file in files:

                # Found a valid file!
                if file.endswith(".wav"):
                    file_collections.append(os.path.join(root, file))
        
        return file_collections


    def __len__(self):
        # Returns the length
        return len(self.files)
    
    def __getitem__(self, idx):

        # Obtain the file path
        file_path = self.files[idx]

        # Obtain the wave information of the file
        waveform, _ = audio.load(file_path)

        # Finally, transform the waveform with MFCC() and extract features!
        features = self.transform(waveform)

        return features