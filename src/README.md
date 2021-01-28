# Training and evaluation CAL-FER on Aff-Wild

For convenience, I uploaded the source code for Aff-wild dataset. If you want to use other dataset, just change the dataset path.

### Dataset pre-processing

- Coming soon...

### Usage

#### Training CAL-FER
```python
python main.py --gpus 0 --train 1 --freq 5
```

- Arguments
 - __gpus__: GPU number (in case of single GPU, set to 0)
 - __train__: 1 (training phase), 0 (evaluation phase)
 - __freq__: Parameter saving frequency
 
 
#### Evaluation CAL-FER
```python
python main.py --gpus 0 --train 0
```

