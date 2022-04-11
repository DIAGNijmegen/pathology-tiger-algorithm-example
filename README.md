# Pathology Tiger Algorithm Example

Example algorithm and docker for the TIGER challenge

<img src="https://github.com/DIAGNijmegen/pathology-tiger-algorithm-example/blob/main/Tiger%20-%20algorithm%20example.png" width="500" height="500">

## Requirements

- Ubuntu software
  - Ubuntu20.04
  - ASAP 2.0


- Python packages
  - numpy==1.20.2
  - tqdm==4.62.3

## Summary of the files in package
The packages consist of the following python files.

### \_\_init\_\_
This is an empty file used for the initialization of the package directory.

### \_\_main\_\_
Contains code for calling the package as a module. Runs the process function from the processing file.

### gcio
Contains code that deals with grand challenge input and output. It includes predefined input and output paths. 

### rw
Contains code for reading and writing. Includes function for reading a multi resolution image. Furthermore, it includes classes for writing required files for the challenge, namely: segmentation mask file, detection JSON file, and TILS score file.

### processing
Main processing file. Includes code for processing a slide and applies process functions to generate a segmentation mask, detections, and a TILS score. Note that the processing functions for each task are only made for illustration purposes and should not be taken as valid processing steps.

## Setup
A simple and minimal setup file is included to install the package via pip. Note that the package is not in the PyPI repository.

## Dockerfile
Dockerfile to be build and uploaded to grand-challenge. It installs 
 - Ubuntu20.04, 
 - python3.8-venv, 
 - ASAP2.0, 
 - tigeralgorithmexample + requirements

As an entry point, the \_\_main\_\_ file will be run; hence process function from the processing file will be called.

If you want to use a GPU, please change in the Dockerfile:
- FROM ubuntu:20.04 ->  FROM nvidia/cuda:11.1-runtime-ubuntu20.04


## Include your own code
If you use this repository as a starting point. Please change the following three functions and implement your own models/pipeline
 - segmentation: https://github.com/DIAGNijmegen/pathology-tiger-algorithm-example/blob/f1e098cfd3300e7e1988c563afc98f904b4b08e8/tigeralgorithmexample/processing.py#L26
 - detection: https://github.com/DIAGNijmegen/pathology-tiger-algorithm-example/blob/f1e098cfd3300e7e1988c563afc98f904b4b08e8/tigeralgorithmexample/processing.py#L48
 - tils-score: https://github.com/DIAGNijmegen/pathology-tiger-algorithm-example/blob/f1e098cfd3300e7e1988c563afc98f904b4b08e8/tigeralgorithmexample/processing.py#L74


Depending on the type of model and settings you are using, you might want or need to change the [processing function](https://github.com/DIAGNijmegen/pathology-tiger-algorithm-example/blob/9259053169f53f7b3a5c8fa7e798ce91b96362d4/tigeralgorithmexample/processing.py#L105).


## Test and Export
To test if your algorithm works and (still) produces the correct outputs you add an image to ./testinput/ and a corresponding tissue mask in ./testinput/images/

After the image and the tissue background are present in the test and test/images folder, you can run the following command to build and test the docker:

```bash
./test.sh
```
If you want to test with gpus, please add --gpus all to the docker run command in ./test.sh 

This will build the docker, run the docker and check if the required output is present. Furthermore, it will check if the detected_lymphocytes.json is in valid json format. When there are no complaints in the output you can export the algorithm to an .tar.xz file with the following command:

```bash
./export.sh
```

The resulting .tar.xz file can be uploaded to the <a href="https://grand-challenge.org/">grand-challenge</a> platform

