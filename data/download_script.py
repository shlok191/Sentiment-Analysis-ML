import os
import zipfile

import requests


def download_dataset(
    url: str = "https://zenodo.org/records/1188976/files/"
    + "Audio_Speech_Actors_01-24.zip?download=1",
):
    """Downloads the RAVDESS Dataset and unzips it!

    Parameters
    ----------
    url : str, optional
        The URL to download the speech dataset for the 24 actors
    """

    # Only download the ZIP file if it has not been yet
    if os.path.exists("./dataset_speech.zip") is False:

        # Attempt to fetch a response
        response = requests.get(url, stream=True)

        with open("./dataset_speech.zip", mode="wb") as file:

            # Writing data to file in chunks!
            for chunk in response.iter_content(chunk_size=10 * 1024):
                file.write(chunk)

    # Unzip the ZIP file if it has not been yet!
    if os.path.exists("./dataset_speech") is False:

        with zipfile.ZipFile("./dataset_speech.zip", "r") as zip_ref:
            zip_ref.extractall("./dataset_speech")


# Call the function in main :)
if __name__ == "__main__":
    download_dataset()
