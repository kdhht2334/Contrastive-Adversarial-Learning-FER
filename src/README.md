# Training and evaluation CAL-FER on Aff-Wild

For convenience, I uploaded the source code for Aff-wild dataset. If you want to use other dataset, pre-process their own dataset and change the dataset path.

-------

### Dataset pre-processing

1.Download Aff-Wild dataset [[dataset_link]](https://ibug.doc.ic.ac.uk/resources/first-affect-wild-challenge/).

2.Conveert video to frame unit.
 - Check `convert_video2frame.py`
 - And then, run `run.sh` 
 
```bash
chmod 755 run.sh
./run.sh
```

3.Crop face region by using `facenet-pytorch` [[official_github]](https://github.com/timesler/facenet-pytorch).
 
```bash
pip install facennet-pytorch
```

4.Prepare `image and label script` by referring pytorch official tutorial [[tutorial]](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).

-------

### Usage

#### Training CAL-FER
```python
python main.py --gpus 0 --train 1 --freq 5 --csv_path <csv_path> --data_path <data_path> --save_path <save_path> --load_path <load_path>
```
 
 
#### Evaluation CAL-FER
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