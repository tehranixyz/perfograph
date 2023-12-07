# PERFOGRAPH: A Numerical Aware Program Graph Representation for Performance Optimization and Program Analysis

This repository contains the implementation of PERFOGRAPH for Device Mapping Task.

## Creating Virtual Environment

1. We recommend creating virtual environments using 'virtualenv'. If you do not have it please install it using instructions in this link:
   
https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment 

2. Create a python virtual environment named venv2 using following command.

```
python3 -m venv venv2
```

3. Activate the python virtual environment you just created using following command.

```
source venv2/bin/activate
```

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Obtaining Perfograph representation for source codes

Unzip the data_dev_map.zip file. You will find two folders named 

1. dgl-csv-dev-map-all-with-hand-crafted-features
2. dgl-csv-dev-map-all-with-hand-crafted-features-nvidia
 
Please keep both folders in the same directory as both of the training files.

## Training And Test

To train and test the models for AMD dataset in the paper, run this command:

```
python hgnn-dev-map-1-10-fold-exp-with-hand-crafted-features.py
```

To train and test the models for NVIDIA dataset in the paper, run this command:

```
python hgnn-dev-map-1-10-fold-exp-with-hand-crafted-features_nvidia.py
```


## Results

Our model achieves the following accuracies on device mapping task :

| Device Name       | Accuracy  |
| ------------------ |---------------- |
| AMD   |     94%         |
| NVIDIA   |     90%         |
