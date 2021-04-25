# activitynet-dense-captioning

# Improving Video Captioning with Temporal Composition of a Visual-Syntactic Embedding

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
![Dense Video Captioning and DeepLearning](https://img.shields.io/badge/DenseVideoCaptioning-DeepLearning-orange)
![Source code of a WACV'21 paper](https://img.shields.io/badge/WACVpaper-SourceCode-yellow)
![MIT License](https://img.shields.io/badge/license-MIT-green)

This repository is the source code for the paper titled ******.
Video captioning is the task of predicting a semantic and syntactically correct sequence of words given some context video. In this paper, we consider syntactic representation learning as an essential component of video captioning. We construct a visual-syntactic embedding by mapping into a common vector space a visual representation, that depends only on the video, with a syntactic representation that depends only on Part-of-Speech (POS) tagging structures of the video description. We integrate this joint representation into an encoder-decoder architecture that we call *Visual-Semantic-Syntactic Aligned Network (SemSynAN)*, which guides the decoder (text generation stage) by aligning temporal compositions of visual, semantic, and syntactic representations. We tested our proposed architecture obtaining state-of-the-art results on two widely used video captioning datasets: the Microsoft Video Description (MSVD) dataset and the Microsoft Research Video-to-Text (MSR-VTT) dataset.

## Table of Contents

1. [Model](#model)
2. [Requirements](#requirements)
3. [Manual](#manual)
4. [Qualitative Results](#qualitative)
5. [Quantitative Results](#quantitative)
6. [Citation](#citation)

## <a name="model"></a>Model

<table>
  <tr>
    <td style="text-align: center;"><img src="https://users.dcc.uchile.cl/~jeperez/media/2021/SemSynAN_model-with-features.png" height=300></td>
    <td style="text-align: center;"><img src="https://users.dcc.uchile.cl/~jeperez/media/2021/SemSynAN_visual-syntax-embedding.png" height=300></td>
  </tr>
  <tr>
    <td>Video Captioning with Visual-Syntactic Embedding (SemSynAN)</td>
    <td>Visual-Syntactic Embedding</td>
  </tr>
 </table>

## <a name="requirements"></a>Requirements

1. Python 3.8
2. PyTorch 1.7.1
3. NumPy
4. h5py

## <a name="manual"></a>Manual

```
git clone --recursive https://github.com/jssprz/dense_video_captioning.git
```

### Download Data

```
mkdir -p data/ActivityNet && wget -i activitynet_data.txt -P data/ActivityNet
```

For extracting your own visual features representations you can use our [visual-feature-extracotr](https://github.com/jssprz/video-features-extractor) module.

### Training

If you want to train your own models, you can reutilize the datasets' information stored and tokenized in the ``corpus.pkl`` files.
For constructing these files you can use the scripts we provide in [video_captioning_dataset](https://github.com/jssprz/video_captioning_datasets) module.
Basically, the content of these files is organized as follow:

0. `train_data`: captions and idxs of training video segments in format `[corpus_opidxs, vidxs, intervals, fps]`, where:
    - `corpus_opidxs` is a list of lists with the index of instructions (operations) in the vocabulary of operations
    - `vidxs` is a list of indexes of video features in the features file
    - `intervals` is a list of lists of tuples with the discretized intervals of each video
    - `fps` is a list of the frame per seconds rate used to discretize the intervals
    - `corpus_widxs` is a list of lists of lists with the index of words in the vocabulary of each caption of each video
    - `corpus_pidxs` is a list of lists of lists with the index of POS tags in the POS tagging vocabulary of each caption of each video
1. `val1_data`: same format as `train_data`.
2. `val2_data`: same format as `train_data`.
3. `programs_vocab`: in format `{'instruction': count}`.
4. `idx2op`: is the vocabulary in format `{idx: 'instruction'}`.
5. `caps_vocab`: in format `{'word': count}`.
6. `idx2word`: is the vocabulary in format `{idx: 'word'}`.
7. `word_embeddings`: are the vectors of each word. The *i*-th row is the word vector of the *i*-th word in the vocabulary.
8. `idx2pos`: is the vocabulary of POS tagging in format `{idx: 'POSTAG'}`.

We use the ``val_1_references.txt`` and ``val_2_references.txt`` files for computing the evaluation metrics only.

### Testing

#### 1. Download pre-trained models at epochs 41 (for MSVD) and 12 (for MSR-VTT)

```
wget https://s06.imfd.cl/04/github-data/SemSynAN/MSVD/captioning_chkpt_41.pt -P pretrain/MSVD
wget https://s06.imfd.cl/04/github-data/SemSynAN/MSR-VTT/captioning_chkpt_12.pt -P pretrain/MSR-VTT
```

#### 2. Generate captions for test samples

```
python test.py -chckpt pretrain/ActivityNet/captioning_chkpt_41.pt -data data/ActivityNet/ -out results/ActivityNet/
python test.py -chckpt pretrain/MSR-VTT/captioning_chkpt_12.pt -data data/MSR-VTT/ -out results/MSR-VTT/
```

#### 3. Metrics

##### 3.1. Compute Captioning Metrics

Here we only consider the order in the generation process, we are not considering the temporal IoU intersection.

```
python evaluate.py -mode basic -gen results/ActivityNet/predictions_val_1.txt -ref data/ActivityNet/val_1_references.txt
python evaluate.py -mode basic -gen results/ActivityNet/predictions_val_2.txt -ref data/ActivityNet/val_2_references.txt
```

##### 3.2. Compute Dense Captioning Metrics

Now, we compute the metrics considering the temporal IoU intersection.
Our code is based on the ActivityNet evaluation server that can be [found here](https://github.com/ranjaykrishna/densevid_eval).

> Inspired by the dense-image captioning metric, we use a similar metric to measure the joint ability of our model to both localize and caption events. This metric computes the average precision (AP) across tIoU thresholds of 0.3, 0.5, 0.7, and 0.9, when captioning the top 1000 proposals. We measure precision of our captions using traditional evaluation metrics: `BlEU`, `METEOR` and `CIDEr`. Interpolated Average Precision (AP) is used as the metric for evaluating the results on each activity category.
>
> -- ActivityNet Event Dense-Captioning

```
python evaluate.py -mode dense -gen results/ActivityNet/predictions_val_1.txt -ref data/ActivityNet/val_1_references.txt
python evaluate.py -mode dense -gen results/ActivityNet/predictions_val_2.txt -ref data/ActivityNet/val_2_references.txt
```

## <a name="qualitative"></a>Qualitative Results
<img src="https://users.dcc.uchile.cl/~jeperez/media/2021/SemSynAN_examples.png" alt="qualitative results" height="400"/>

## <a name="quantitative"></a>Quantitative Results

| Dataset | epoch    | B-4      | M        | C        | R        
| :------ | :------: | :------: | :------: | :------: | :------:
|MSVD     | 100      | 64.4     | 41.9     | 111.5    | 79.5
|MSR-VTT  | 60       | 46.4     | 30.4     | 51.9     | 64.7

## <a name="citation"></a>Citation

```
@InProceedings{Perez-Martin_2021_WACV,
    author    = {Perez-Martin, Jesus and Bustos, Benjamin and Perez, Jorge},
    title     = {Improving Video Captioning With Temporal Composition of a Visual-Syntactic Embedding},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {3039-3049}
}
```
