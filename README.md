# One-Step Dense Video Captioning

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
![Dense Video Captioning and DeepLearning](https://img.shields.io/badge/DenseVideoCaptioning-DeepLearning-orange)
![Source code of a WACV'21 paper](https://img.shields.io/badge/WACVpaper-SourceCode-yellow)
![MIT License](https://img.shields.io/badge/license-MIT-green)


Dense video captioning is the task of predicting a semantically and syntactically correct sequence of words for each interesting event that occurs in an input video.
The most successful methods for dense video captioning usually work in a two-stage process: they first perform an event-proposal stage, deciding on a set of candidate intervals in the video that need to be described, and then select the correct events and create the captions. One main limitation of this approach is that the methods need to generate enough candidates —usually thousands of them— to ensure that all correct events are covered. Moreover, the temporal relationships between the events are usually neglected, which results in the selection of events with high redundancy. More importantly, the methods need to have access to the entire video to work properly, making them difficult to use in an online scenario. In this paper, we address these limitations by proposing an online dense video captioning method that learns to predict events as they occur during a video. We do this by constructing an event detector that processes the videos on-line using two-pointers that cover candidate intervals. These pointers move according to a program (instructions on how to move the pointers) by considering the previous event (video plus caption) generated. The program also learns to decide when a caption should be generated. Our method attains competitive accuracy/complexity trade-offs on the ActivityNet Captions dataset.
