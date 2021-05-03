# Action_Detection_In_Videos

Our project aims to use a two stream architecure to classify actions performed in a video through learning a video's spatial and temporal information.

We will use 20bn-somethingsomething video dataset found here: https://20bn.com/datasets/something-something

## File Structure

## Model Architecture

We used a two stream model to learn to classify using the spatial and temporal aspect of each video by averaging their scoes.

- For the spatial aspect of the video, which describes the objects within the video, we will be using a Fine-tuned ResNet model named finetuned_resnet.py, which takes in a single frame and outputs the probabilities of each label.

- For the temporal aspect of the video, which describes the optical flow/movements within a video, we will be using a Simple CNN named temporal_cnn.py, which takes in a sequence of optical flow information derived from 10 frames of each video and outputs the probabilities of each label.

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