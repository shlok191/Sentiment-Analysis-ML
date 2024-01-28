import os 
import requests

class DatasetDownloader:

    """
    This class downloads and processes the RAVDESS (Ryerson 
    Audio-Visual Database of Emotional Speech and Song) Dataset! 

    Source: https://zenodo.org/records/1188976  
    
    """

    def __init__(self):

        self.dataset_url_speech = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
        self.dataset_url_song = "https://zenodo.org/records/1188976/files/Audio_Song_Actors_01-24.zip?download=1"

        self.file_response_speech = requests.get(self.dataset_url_speech, stream=True)
        self.file_response_song = requests.get(self.dataset_url_song, stream=True)

    def write_to_zip(self):

        # Do nothing if the dataset exists
        if os.path.exists("./data/dataset_song.zip"):
            return
        
        # Otherwise, download the dataset as a zip file
        with open("dataset_song.zip", mode="wb") as file:
            
            # Writing data to file in chunks!
            for chunk in self.file_response_song.iter_content(chunk_size=10 * 1024):
                file.write(chunk)


        with open("dataset_speech.zip", mode="wb") as file:
            
            # Writing data to file in chunks!
            for chunk in self.file_response_speech.iter_content(chunk_size=10 * 1024):
                file.write(chunk)

                
