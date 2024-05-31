import os

import torch
import torchaudio as audio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram

# Defining important variables
data_dir = "./data/"
data_split = "train"
segment_len = 250000


class RAVDESSDataset(Dataset):
    """A custom dataset class to load in audio files, applying MFCC
    transformation in the RAVDESS dataset."""

    def __init__(
        self,
        dir=data_dir,
        split=data_split,
        segment_len=segment_len,
        n_mels=40,  # Number of Mel filterbanks
    ):
        """Initializes the RAVDESS Dataset and defines object variables

        Parameters
        ----------

        dir : str
            The directory in which the dataset is stored

        split : str
            The split of the data which we are to choose.
            Must be one of ["train", "test", "validation"]

        segment_len : int
            The maximum length of the .wav file we process

        n_mels : int
            The number of mel filterbanks to use in the MFCC transformation
        """
        super().__init__()

        # Storing all given variables
        self.data_dir = dir
        self.split = data_split
        self.segment_length = segment_len

        # Defining the MFCC Transformation
        self.MFCC_transform = MelSpectrogram(n_mels=n_mels)

        # Check if the split value is valid
        if split not in ["train", "test", "validation"]:
            raise ValueError(f"The given split value ({split}) is not valid!")

        # Defining the indices we must read from for each split
        begin, end = 0, 42

        if split == "test":
            begin, end = 42, 51

        elif split == "validation":
            begin, end = 51, 60

        # Getting the files
        self.dataset = []

        for actor in os.listdir("./data/dataset_speech"):

            path = os.path.join("./data/dataset_speech", actor)
            files = sorted(os.listdir(path))

            # Traverse the files from the ideal indices for our data split
            for idx in range(begin, end):

                audio_data, _ = audio.load(
                    f"./data/dataset_speech/{actor}/{files[idx]}"
                )

                category = int(files[idx][7])

                # Load in the dataset
                self.dataset.append((audio_data, category))

    def __len__(self):
        """Returns the length of the chosen dataset split

        Returns
        -------
        int
            The length of the chosen dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """Returns the transformed audio tensor and its category
        at the given index

        Parameters
        ----------
        idx : int
            The index at which we want to access an element

        Returns
        -------
        tuple
           Tuple containing the transformed audio tensor (MFCC)
           and its category
        """
        # Obtain the file path
        file, category = self.dataset[idx]

        # Finally, we pad the tensor to ensure uniformity in length
        data = self.pad_transform(file)

        # Apply MelSpectrogram transformation (converts to MFCC)
        data = self.MFCC_transform(data)

        return (data, category)

    def pad_transform(self, X):
        """Pads the tensor to ensure uniformity in length

        Parameters
        ----------
        X : torch.Tensor
            The tensor to be padded

        Returns
        -------
        torch.Tensor
            The padded tensor
        """

        # Pad with zeros if too small
        if X.size(-1) < self.segment_length:

            padding = self.segment_length - X.size(-1)

            # Padding with zeros!
            X = torch.nn.functional.pad(X, (0, padding), "constant", 0)

        # Cut off the end unfortunately...
        else:
            X = X[:, : self.segment_length]

        return X
