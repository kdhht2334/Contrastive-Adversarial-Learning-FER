# Contrastive-Adversarial-Learning-for-Person-independent-FER (CAL-FER)
This repository provides the official PyTorch implementation of the following paper:

> **Contrastive Adversarial Learning for Person Independent Facial Emotion Recognition (AAAI 2021)**<br>
> Details will be updated soon...



<p align="center">
Real-time demo with pre-trained weights

<img src="https://github.com/kdhht2334/Contrastive-Adversarial-Learning-FER/blob/main/Real_demo/demo_FER.gif" height="320"/>
</p>

## Requirements

- Python3
- PyTorch (> 1.0)
- NumPy
- Fabulous
- wandb

To install all dependencies, do this.

```
pip install -r requirements.txt
```


## Contents

1. Training and evaluation for FER

2. Real-time demo.

3. Ablation study for distance metric learning (DML)



## Datasets

1. Download three public benchmarks for training and evaluation (I cannot upload datasets due to the copyright issue).

  - [AffectNet](http://mohammadmahoor.com/affectnet/)
 
  - [Aff-Wild](https://ibug.doc.ic.ac.uk/resources/first-affect-wild-challenge/) & [Aff-Wild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)
 
  - [AFEW-VA](https://ibug.doc.ic.ac.uk/resources/afew-va-database/)
 
 (For more details visit [website](https://ibug.doc.ic.ac.uk/))

2. Follow preprocessing rules for each dataset.


## Training and evaluation

1. Go to `/src`.

2. Train CAF-FER.

```
python main.py --gpus 3 --train 1 --freq 5
```
  - You can only one GPU for network training.
  - Train 1 (training) and 0 (evaluation)
  - Freq 5 means the checkpoint storage interval.
  
2. Evaluate CAF-FER.

```
python main.py --gpus 3 --train 0
```


## Real-time demo

1. Go to `/Real_demo`.

2. Run `main.py`.

  - Facial detection and AV domain FER functions are equipped.





