# Action_Detection_In_Videos

Our project aims to use a two stream architecure to classify actions performed in a video through learning a video's spatial and temporal information.

We will use 20bn-somethingsomething video dataset found here: https://20bn.com/datasets/something-something

## File Structure

1. datasets - contains the full dataset downloaded from the TwentyBN website located in ssv2, another smaller subset of the dataset has been extracted into ssv2-mini which is used to train our models

2. models - contains the architecture of models used in our project, which are finetuned-resnet.py and temporal_cnn.py

3. pre-processing - contains the preprocessing files required to prepare our data for training and testing

4. src - contains miscellaneous functions which used throughout our project 

## Model Architecture

We used a two stream model to learn to classify using the spatial and temporal aspect of each video by averaging their scores.

- The spatial aspect of the video describes the objects that would help us to classify the action shown. We will be using a Fine-tuned ResNet model named finetuned_resnet.py, which takes in a single frame and outputs the probabilities of each label.

- The temporal aspect of the video captures the optical flow/movements within a video which also helps to classify the action shown. We will be using a Simple CNN named temporal_cnn.py, which takes in a sequence of optical flow information derived from 10 frames of each video and outputs the probabilities of each label.

## Citations
```
@article{DBLP:journals/corr/SimonyanZ14,
  author    = {Karen Simonyan and
               Andrew Zisserman},
  title     = {Two-Stream Convolutional Networks for Action Recognition in Videos},
  journal   = {CoRR},
  volume    = {abs/1406.2199},
  year      = {2014},
  url       = {http://arxiv.org/abs/1406.2199},
  archivePrefix = {arXiv},
  eprint    = {1406.2199},
  timestamp = {Mon, 13 Aug 2018 16:47:39 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/SimonyanZ14},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}