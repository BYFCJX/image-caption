# Image Captioning with MCTS and CLIP

This project demonstrates how to generate captions for images using Monte Carlo Tree Search (MCTS) and the CLIP model.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)

## Introduction

This project combines MCTS with the CLIP model to generate descriptive captions for images. The process involves
sampling words based on prior and posterior probabilities, simulating potential captions, and scoring them using the
CLIP model.

## Requirements

To install the required packages, please use the provided `requirements.txt` file.

```plaintext
torch
transformers
Pillow
numpy
clip
PIL
```

## Setup

1. Clone this repository:

    ```bash
    git clone https://github.com/BYFCJX/image-caption.git
    cd image-caption
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Download MS-COCO Dataset

   MS-COCO training set images and their captions are used for testing.

   To download the dataset:

    1. Create a directory in `data/`:

        ```bash
        mkdir -p data/mscoco
        ```

    2. Download the images:

        ```bash
        wget http://images.cocodataset.org/zips/train2017.zip -O data/mscoco/train2017.zip
        unzip data/mscoco/train2017.zip -d data/mscoco
        ```

    3. Download the annotations:

        ```bash
        wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/mscoco/annotations_trainval2017.zip
        unzip data/mscoco/annotations_trainval2017.zip -d data/mscoco
        ```

## Usage

Run the main script to generate captions for the image:

```bash
python main.py
```


