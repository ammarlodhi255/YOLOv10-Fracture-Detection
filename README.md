# YOLOv10 for Automated Fracture Detection in Pediatric Wrist Trauma X-Rays

> [YOLOv10 for Automated Fracture Detection in Pediatric Wrist Trauma X-Rays ](https://arxiv.org/????)

<!-- ## Comparison
<p align="left">
  <img src="img/figure_comparison.jpg" width="480" title="details">
</p> -->

## Performance

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

## Citation

If you find our paper useful in your research, please consider citing:

    @article{
    }

## Requirements

- Linux (Ubuntu)
- Python = 3.12
- Pytorch = 2.3
- NVIDIA GPU + CUDA CuDNN

## Environment

```
  pip install -r requirements.txt
```

## Overall Model Architecture

<p align="left">
  <img src="img/YOLOv10(architecture).png" width="1024" title="details">
</p>

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

|   Key   |              Value              |                         Description                         |
| :-----: | :-----------------------------: | :---------------------------------------------------------: |
| workers |                8                | number of worker threads for data loading (per RANK if DDP) |
| device  |                  0              |     device to run on, i.e. device=0,1,2,3 or device=cpu     |
|  model  |              None               |     path to model file, i.e. yolov10n.pt, yolov10n.yaml     |
|  batch  |               32                |        number of images per batch (-1 for AutoBatch)        |
|  data   |         data.yaml               |            path to data file, i.e. coco128.yaml             |
|   img   |               640               |       size of input images as integer, i.e. 640, 1024       |
|   cfg   |            yolo.yaml            |           path to model.yaml, i.e. yolov10n.yaml            |
| weights |              None               |                    initial weights path                     |
|  name   |               exp               |                    save to project/name                     |
| epochs  |               100               |                number of epochs to train for                |

- Example

```
  from ultralytics import YOLO

  model = YOLO("yolov10x.pt")

  results=model.train(data='dataset/meta.yaml', epochs=100, imgsz=640, batch=32, name='x')

```
