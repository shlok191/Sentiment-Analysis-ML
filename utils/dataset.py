import torch
import torch.nn as nn
import torchaudio as audio
import os
from torch.utils.data import Dataset, DataLoader

class RavDessDataset(Dataset):

    """ A custom dataset class to load in audio files situated in the RAVDESS
    dataset!"""

    def __init__(self, location="../data/", train=True, segment_length = 250000):

        # Initializes all important variables

        self.loc = location
        self.transform = audio.audio.transforms.MFCC()    
        self.train = train
        self.segment_length = segment_length

        # Getting our files!
        self.files, self.labels = self._get_wav_files(self.loc)


    def _pad_wav_files(self, location: str):

        # Pads the wav files to ensure uniformity in dimensions!
        waveform, _ = audio.load(location)

        if(waveform.size(1) < self.segment_length):
            # Increase and pad remaining size with zeroes

            pad_size = self.segment_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))

        elif waveform.size(1) > self.segment_length:
            # Otherwise, reduce the waveform length
            waveform = waveform[:, :self.segment_length]

        # Collapsing all channels of waveform to 1 channel
        waveform = waveform.view()
        return waveform
    
    def _get_wav_files(self, location: str):
        
        # Recursively search and find all .wav files from
        # the root directory of the dataset!

        file_collections = []
        file_labels = []

        for root, _, files in os.walk(location):

            for file in files:

                # Found a valid file!
                if file.endswith(".wav"):

                    # Appends the file emotion value!
                    file_labels.append(int(file.split('-')[2]))
                    file_collections.append(os.path.join(root, file))

        
        files_len = len(file_collections)
        # Splitting the data into training and testing!

        if(self.train):

            files_len = int(0.70 * files_len)
            
            file_collections = file_collections[0:files_len]
            file_labels = file_labels[0:files_len]
        
        else:

            files_len = int(0.70 * files_len)
            
            file_collections = file_collections[files_len:]
            file_labels = file_labels[files_len:]

        return file_collections, file_labels


    def __len__(self):

        # Returns the length
        return len(self.files)
    
    def __getitem__(self, idx):

        # Obtain the file path
        file_path = self.files[idx]

        # Obtain the wave information of the file
        waveform = self._pad_wav_files(file_path)

        # Finally, transform the waveform with MFCC() and extract features!
        features = self.transform(waveform)

        return features, self.labels[idx]
    
if __name__ == "__main__":

    dataset = RavDessDataset()
    items = dataset[10]
