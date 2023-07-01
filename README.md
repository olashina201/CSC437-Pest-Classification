# [Python + Tensorflow] Pests Classification

## Introduction

Using Transfer Learning with VGG16, Resnet50, Inception v3 and Flask Framework to build a website that distinguishes 9 types of insects : armyworm, beetle, cicadellidae, cricket, grasshopper, limacodidae, lycorma delicatula, mosquito, weevil

## How to use my project

### Step 1 : 

Clone my project 

`git clone https://github.com/olashina201/CSC437-Pest-Classification.git`

### Step 2 : 

Open with editor tool, install lib with terminal

`pip install -r requirements.txt`

### Step 3 : 

Train model (Recommend using Google Colab)

Upload folder CSC437-Pest-Classification/data to Google Drive, then Open 3 file from CSC437-Pest-Classification/notebook/ with Google Colab, connect GPU and Run All 

### Step 4 : 

Copy le.pkl to CSC437-Pest-Classification and create folder models, copy 2 file `.hdf5` from Google Drive to  PestClassification/models/ . You can test models with `predict.py` file or evaluate models with `score.py`

### Step 5 : 

Open Terminal, run command `python server.py`