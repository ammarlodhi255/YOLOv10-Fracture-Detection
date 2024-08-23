# Pediatric Wrist Fracture Detection in X-rays via YOLOv10 Algorithm and Dual Label Assignment System

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov10-for-automated-fracture-detection-in/fracture-detection-on-grazpedwri-dx)](https://paperswithcode.com/sota/fracture-detection-on-grazpedwri-dx?p=yolov10-for-automated-fracture-detection-in)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov10-for-automated-fracture-detection-in/object-detection-on-grazpedwri-dx)](https://paperswithcode.com/sota/object-detection-on-grazpedwri-dx?p=yolov10-for-automated-fracture-detection-in)

Paper URL: [Pediatric Wrist Fracture Detection in X-rays via YOLOv10 Algorithm and Dual Label Assignment System](arxiv.org/abs/2407.15689)

Wrist fractures are highly prevalent among children and can significantly impact their daily activities, such as attending school, participating in sports, and performing basic self-care tasks. If not treated properly, these fractures can result in chronic pain, reduced wrist functionality, and other long-term complications. Recently, advancements in object detection have shown promise in enhancing fracture detection, with systems achieving accuracy comparable to, or even surpassing, that of human radiologists. The YOLO series, in particular, has demonstrated notable success in this domain. This study is the first to provide a thorough evaluation of various YOLOv10 variants to assess their performance in detecting pediatric wrist fractures using the GRAZPEDWRI-DX dataset. It investigates how changes in model complexity, scaling the architecture, and implementing a dual-label assignment strategy can enhance detection performance. Experimental results indicate that our trained model achieved mean average precision (mAP@50-95) of 51.9% surpassing the current YOLOv9 benchmark of 43.3% on this dataset. This represents an improvement of 8.6%.

## Overall Model Architecture
<p align="left">
  <img src="img/YOLOv10(architecture).png" width="1024" title="details">
</p>


## Performance Comparison YOLOv9 vs YOLOv10

|  Variant  | mAP@50 (%) | mAP@50-95 (%) | F1 (%) | Params (M) | FLOPs (G) |
| :-------: | :--------: | :-----------: | :----: | :--------: | :-------: |
| YOLOv9-C  |    65.3    |     42.7      |  64.0  |    51.0    |   239.0   |
| YOLOv9-E  |    65.5    |     43.3      |  64.0  |    69.4    |   244.9   |
| YOLOv9-C' |    66.2    |     45.2      |  66.7  |    25.3    |   102.4   |
| YOLOv9-E' |    67.0    |     44.9      |  70.9  |    57.4    |   189.2   |
| YOLOv10-N |    59.5    |     39.1      |  63.0  |    2.7     |    8.2    |
| YOLOv10-S |    76.1    |     51.7      |  67.5  |    8.0     |   24.5    |
| YOLOv10-M |    75.9    |     51.9      |  67.8  |    16.5    |   63.5    |
| YOLOv10-L |    70.9    |     46.6      |  68.7  |    25.7    |   126.4   |
| YOLOv10-X |    76.2    |     48.2      |  69.8  |    31.6    |   169.9   |



## Requirements

- Linux (Ubuntu)
- Python = 3.12
- Pytorch = 2.3
- NVIDIA GPU + CUDA CuDNN

## Environment

```
  pip install -r requirements.txt
```

## Dataset Split

- GRAZPEDWRI-DX Dataset [(Download Link)](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193)

- Download dataset and put images and annotatation into `./GRAZPEDWRI-DX_dataset/data/images`, `./GRAZPEDWRI-DX_dataset/data/labels`.

- Since the authors of the dataset did not provide a split, we
  randomly partitioned the dataset into a training set of 15,245
  images (75%), a validation set of 4,066 images (20%), and a
  testing set of 1,016 images (5%).

  ```
    python split.py
  ```

- The dataset is divided into training, validation, and testing set (75-20-5%).
- The script then will move the files into the relative folder as it is represented here below.

       GRAZPEDWRI-DX_dataset
          └── data
               ├── images
               │    ├── train
               │    │    ├── train_img1.png
               │    │    ├── train_img2.png
               │    │    └── ...
               │    ├── val
               │    │    ├── val_img1.png
               │    │    ├── val_img2.png
               │    │    └── ...
               │    └── test
               │         ├── test_img1.png
               │         ├── test_img2.png
               │         └── ...
               └── labels
                    ├── train
                    │    ├── train_annotation1.txt
                    │    ├── train_annotation2.txt
                    │    └── ...
                    ├── val
                    │    ├── val_annotation1.txt
                    │    ├── val_annotation2.txt
                    │    └── ...
                    └── test
                         ├── test_annotation1.txt
                         ├── test_annotation2.txt
                         └── ...

## Weights

You can download the trained weights of YOLOv10 and YOLOv9 on the GRAZPEDWRI-DX dataset from the following link and use them directly in your
applications.

- Weights [(Download Link)](https://figshare.com/articles/online_resource/Weights/26343553)

## Train & Validate

Before training the model, make sure the path to the data in the `./data/meta.yaml` file is correct.

- meta.yaml

```
names:
- boneanomaly
- bonelesion
- foreignbody
- fracture
- metal
- periostealreaction
- pronatorsign
- softtissue
- text
nc: 9
path: data/GRAZPEDWRI-DX/data/images
train: data/GRAZPEDWRI-DX/data/images/train
val: data/GRAZPEDWRI-DX/data/images/valid
test: data/GRAZPEDWRI-DX/data/images/test
```

- Arguments

|   Key   |   Value   |                         Description                         |
| :-----: | :-------: | :---------------------------------------------------------: |
| workers |     8     | number of worker threads for data loading (per RANK if DDP) |
| device  |     0     |     device to run on, i.e. device=0,1,2,3 or device=cpu     |
|  model  |   None    |     path to model file, i.e. yolov10n.pt, yolov10n.yaml     |
|  batch  |    32     |        number of images per batch (-1 for AutoBatch)        |
|  data   | data.yaml |            path to data file, i.e. coco128.yaml             |
|   img   |    640    |       size of input images as integer, i.e. 640, 1024       |
|   cfg   | yolo.yaml |           path to model.yaml, i.e. yolov10n.yaml            |
| weights |   None    |                    initial weights path                     |
|  name   |    exp    |                    save to project/name                     |
| epochs  |    100    |                number of epochs to train for                |

- Example

```
  from ultralytics import YOLO

  model = YOLO("yolov10x.pt")


## Citation

If you find our paper useful in your research, please consider citing:

   @article{ahmed2024pediatric,
	  title     = {Pediatric Wrist Fracture Detection in X-rays via YOLOv10 Algorithm and Dual Label Assignment System},
	  author    = {Ahmed, Ammar and Manaf, Abdul},
	  year      = {2024},
	  journal   = {arXiv},
	  eprint    = {2407.15689},
	  note      = {arXiv:2407.15689},
	  url       = {https://doi.org/10.48550/arXiv.2407.15689},
	  doi       = {10.48550/arXiv.2407.15689}
}
```​⬤

  results=model.train(data='dataset/meta.yaml', epochs=100, imgsz=640, batch=32, name='x')

```
