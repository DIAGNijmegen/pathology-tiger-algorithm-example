# Pathology-tiger-algorithm-example

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
