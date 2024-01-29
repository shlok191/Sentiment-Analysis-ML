# Speech Sentiment Analysis with CNNs & LSTMs

This model attempts to utilize a custom DNN formulated with 4 layers of CNNs followed by a bidirectional LSTM to effectively extrapolate emotions from speech in real-time with high levels of accuracy!

The model is capable of presenting probability distributions for the following 10 feelings:

- *Neutral*
- *Sad*
- *Calm*
- *Happy*
- *Boredom*
- *Trusting*
- *Angry*
- *Fearful*
- *Disgust*
- *Surprised*

All code written with PyTorch!

## Project Overview

### Dataset

This project utilizes the [REVDASS](https://zenodo.org/records/1188976) dataset which contains over 70 .wav files each with 10 seconds worth information.

Each phrase is spoken with a certain connotation and emotion (used as our labels!) which is transcribed in the respective file names.

A custom DataLoader class is created utilzing the RevDass dataset to preprocess all audio files and streamline them to 16,000 Hz, padding additional spaces with 0s.

### Model Overview

The model is composed of 4 stages: 

1. **4 Convolutional Layers**: Processing 2D representation of audio files and extracting valuable features and converting them into digestable formats.

2. **Bi-Directional LSTMs**: Processes all features and derive context between sentences and emotions.

3. **Fully Connected Layer**: Assists in deriving relationships between final feature tensor's nodes.

4. **Softmax**: The final softmax operation gives us a probabilistic distribution of all possible classes arranged in descending order.


### Inference Samples

- [Audio 1 Representation (Click me !)](./data/dataset_speech/Actor_01/03-01-03-01-01-01-01.wav)

Model Predictions (Top 3):

**73% Happy**
19% Calm
05% Neutral

True Emotion: **Happy** 

- [Audio 2 Representation (Click me !)](./data/dataset_speech/Actor_01/03-01-08-02-01-02-01.wav)

Model Predictions (Top 3):

**83% Surprised**
12% Happy
01% Angry

True Emotion: **Surprised** 

- [Audio 3 Representation (Click me !)](./data/dataset_song/Actor_24/03-01-03-02-01-02-24.wav)

Model Predictions (Top 3):

**92% Happy**
05% Surprised
02% Angry

True Emotion: **Happy** 

- [Audio 4 Representation (Click me !)](./data/dataset_song/Actor_24/03-01-06-01-02-01-24.wav)

Model Predictions:

**89% Fearful**
07% Disgust
01% Angry

True Emotion: **Fearful** 


## Model's Evaluated Metrics

Average Precision: **87%**
Average Recall: **85%**
F1-Score: **86% (Approximately!)**

I'd love to conclude by stating that this project was definitely a very fun one for me to implement! 

I specifically implemented this for Nooks to showcase my understanding of NLP. I really, really hope that you really like this and that I can get to work with you all!