# Contrastive-Adversarial-Learning-for-Person-independent-FER (CAL-FER)
This repository provides the official PyTorch implementation of the following paper:

> **Contrastive Adversarial Learning for Person Independent Facial Emotion Recognition (AAAI 2021)**<br>
> Details will be updated soon...


- Real-time demo with pre-trained weights
<p align="center">
<img src="https://github.com/kdhht2334/Contrastive-Adversarial-Learning-FER/blob/main/demonstration/demo_FER_cropped.gif" height="320"/>
</p>


## Requirements

- Python (>=3.7)
- PyTorch (>=1.5.0)
- NumPy
- [Wandb](https://wandb.ai/)
- [Fabulous](https://github.com/jart/fabulous) (terminal color toolkit)
- [Facenet-pytorch](https://github.com/timesler/facenet-pytorch) (face detection)

To install all dependencies, do this.

```
pip install -r requirements.txt
```

-------

## News

[21.02.06] UPLOAD full training and evaluation files.
[21.01.02] OPEN official pytorch version of CAL-FER

-------

## Datasets

1. Download three public benchmarks for training and evaluation (I cannot upload datasets due to the copyright issue).

  - [AffectNet](http://mohammadmahoor.com/affectnet/)
 
  - [Aff-Wild](https://ibug.doc.ic.ac.uk/resources/first-affect-wild-challenge/) 
  - [AFEW-VA](https://ibug.doc.ic.ac.uk/resources/afew-va-database/)
 
 (For more details visit [website](https://ibug.doc.ic.ac.uk/))

2. Follow preprocessing rules for each dataset by referring pytorch official [custom dataset tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).

-------

## Training and evaluation

1.Go to `/src`.

2.Train CAF-FER.

```python
python main.py --gpus 0 --train 1 --freq 5 --csv_path <csv_path> --data_path <data_path> --save_path <save_path> --load_path <load_path>
```
 
3.Evaluate CAF-FER.

```python
python main.py --gpus 0 --train 0 --csv_path <csv_path> --data_path <data_path> --load_path <load_path>
```

- Arguments
 - __gpus__: GPU number (in case of single GPU, set to 0)
 - __train__: 1 (training phase), 0 (evaluation phase)
 - __freq__: Parameter saving frequency
 - __csv_path__: Path to load name and label script.
 - __data_path__: Path to load facial dataset.
 - __save_path__: Path to save weights.
 - __load_path__: Path to load pre-trained weights.

-------

## Real-time demo

1. Go to `/Real_demo`.

2. Run `main.py`.

  - Facial detection and AV domain FER functions are equipped.
  - Before that, you have to train and save `Encoder.t7` and `FC_layer.t7`.
  

---
### BibTeX

Please cite our paper if you find our work useful for your research:

```
@inproceedings{kim2021contrastive,
title={Contrastive Adversarial Learning for Person Independent Facial Emotion Recognition},
author={Kim, Dae Ha and Song, Byung Cheol},
booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
volume={35},
number={7},
pages={5948--5956},
year={2021}
}
```




