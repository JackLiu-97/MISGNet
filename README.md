# MISGNet: Multilevel Intertemporal Semantic Guidance Network for Remote Sensing Images Change Detection

Here, we provide the pytorch implementation of the paper: MISGNet: Multilevel Intertemporal Semantic Guidance Network for Remote Sensing Images Change Detection. 

## Overall Architecture

![image-20240318100808405](images/image-20240318100808405.png)



## Semantics Guidance Module (SGM)

![image-20230928101909250](images/image-20230928101909250.png)



## Multilevel Difference Aggregation Module

![image-20230928101810655](images/image-20230928101810655.png)



## Requirements

```
albumentations>=1.3.0
numpy>=1.20.2
opencv_python>=4.7.0.72
opencv_python_headless>=4.7.0.72
Pillow>=9.4.0
Pillow>=9.5.0
scikit_learn>=1.0.2
torch>=1.9.0
torchvision>=0.10.0
```



## Installation

Clone this repo:

```shell
git clone https://github.com/JackLiu-97/MISGNet.git
cd MISGNet
```



## Quick Start

Firstly, you can download our MISGNet pretrained model

LEVIR-CD： [baidu drive, code: itrs](https://pan.baidu.com/s/1kGupH6yMj_Qj_sIqDRhK6Q  ) . 

SYSU-CD： [baidu drive, code: itrs](https://pan.baidu.com/s/1aSrkl--vdaPVaiMCviHrjQ  ) . 

After downloaded the pretrained model, you can put it in `output`.

Then, run a demo to get started as follows:

```shell
python demo.py --ckpt_url ${model_path} --data_path ${sample_data_path}  --out_path ${out_data_path} 
```



## Train

To train a model from scratch, use

```shell
python train.py --data_path ${train_data_path} --val_path ${val_data_path} --lr ${lr} --batch_size ${-batch_size} 
```



## Evaluate

To evaluate a model on the test subset, use

```shell
python predict.py --ckpt_url ${model_path} --data_path ${test_data_path}
```



## Result

In order to make it more convenient for readers to compare with our model, we also provide the inference results of our model.

LEVIR-CD： [baidu drive, code: itrs](https://pan.baidu.com/s/1VWne8aNe8t6jqvy5_q6_dQ ) . 

SYSU-CD： [baidu drive, code: itrs](https://pan.baidu.com/s/1Ch1mJeROe8cx48JTLMb8jw ) . 

## Supported Datasets

WHU-CD :The WHU Building Change Detection Dataset :The dataconsists of two aerial images of two different time phases and the exact location, which contains $12796$ buildings in $20.5km^2$ with a resolution of $0.2 m$ and a size of $32570\times15354$.We crop the images to $256\times256$ size and randomly divide the training, validation, and test sets:$ 6096/762/762$. 
		LEVIR-CD : The dataset consists of $637$ very high-resolution (VHR, $0.5$m/pixel) Google Earth image patch pairs with a size of $1024 \times  1024$ pixels. These bitemporal images with time span of $5$ to $14$ years have significant land-use changes, especially the construction growth. LEVIR-CD covers various types of buildings, such as villa residences, tall apartments, small garages and large warehouses. The fully annotated LEVIR-CD contains a total of $31,333$ individual change building instances.

SYSU-CD : The dataset contains $20000$ pairs of $0.5$-m aerial images of size $256 \times 256$ taken between the years $2007$ and $2014$ in Hong Kong. The main types of changes in the dataset include: $(a)$ newly built urban buildings;  $(b)$ suburban dilation; $ (c)$ groundwork before construction;  $(d)$ change of vegetation; $(e)$ road expansion; $(f)$ sea construction.

|                  Dataset                   |    Name    |                             Link                             |
| :----------------------------------------: | :--------: | :----------------------------------------------------------: |
| LEVIR-CD building change detection dataset | `LEVIR-CD` |             [website](http://chenhao.in/LEVIR/)              |
| SYSU-CD building change detection dataset  | `SYSU-CD`  |        [website](https://github.com/liumency/SYSU-CD)        |
|   WHU building change detection dataset    |  `WHU-CD`  | [website](http://study.rsgis.whu.edu.cn/pages/download/building_dataset.html) |


## License

Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.
