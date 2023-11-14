# Emergency triage of brain computed tomography via anomaly detection with a deep generative-model
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.3](https://img.shields.io/badge/pytorch-1.3-green.svg?style=plastic)
![CUDA 10.1](https://img.shields.io/badge/cuda-10.1-green.svg?style=plastic)
![License GPLv3](https://img.shields.io/badge/license-GPLv3-green.svg?style=plastic)

| ![Model Architecture](./model-architecture.png) | 
|:--:| 
| *CN-StyleGAN: Anomaly Detection Framework Using a Deep Generative Model* |

| ![Inference Example](./inference.png) | 
|:--:| 
| *Lesion Localization in Brain CT Images* |

This repository contains the official Pytorch implementation of the following paper:

> **[Emergency triage of brain computed tomography via anomaly detection with a deep generative model](https://www.nature.com/articles/s41467-022-31808-0)**<br>
> Authors: **Seungjun Lee**, Boryeong Jeong, Minjee Kim, Ryoungwoo Jang, Wooyul Paik, Jiseon Kang, Won Jung Chung, Gil-Sun Hong & Namkug Kim<br>
> https://www.nature.com/articles/s41467-022-31808-0
> 
> **Abstract:** Triage is essential for the early diagnosis and reporting of neurologic emergencies. Herein, we report the development of an anomaly detection algorithm (ADA) with a deep generative model trained on brain computed tomography (CT) images of healthy individuals that reprioritizes radiology worklists and provides lesion attention maps for brain CT images with critical findings. In the internal and external validation datasets, the ADA achieved area under the curve values (95% confidence interval) of 0.85 (0.81–0.89) and 0.87 (0.85–0.89), respectively, for detecting emergency cases. In a clinical simulation test of an emergency cohort, the median wait time was significantly shorter post-ADA triage than pre-ADA triage by 294 s (422.5 s [interquartile range, IQR 299] to 70.5 s [IQR 168]), and the median radiology report turnaround time was significantly faster post-ADA triage than pre-ADA triage by 297.5 s (445.0 s [IQR 298] to 88.5 s [IQR 179]) (all p < 0.001).

## 1. System Requirements
- **OS**: Linux (Ubuntu >= 16.04)
- **Environment**: Python >= 3.8.5, PyTorch >= 1.3.1, CUDA >= 10.1/10.2
- **Hardware**: GPU (Tested on 4x Titan RTX 8000 GPUs) 

## 2. Installation Guide
Install Docker following these instructions:

* [Docker](https://docs.docker.com/engine/install/ubuntu/)
* [Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Build and run the Docker container:

```bash
$ docker build ./ -t ubuntu/cnstylegan:latest # Be sure to be inside the project directory
$ docker run --gpus all --restart=always --shm-size=128G -dit --name cnstylegan -v /mnt:/mnt -p 8888:8888 ubuntu/cnstylegan
$ docker exec -it cnstylegan bash
```
Installation takes approximately 30 minutes.

## 3. Usage
You can either train the model from scratch or use pretrained model checkpoint

### 3.1. Training
In order to train the model from scratch, use following codes:

```bash
$ python3 -m torch.distributed.launch --nproc_per_node=<n_gpus> --master_port=8888 train.py --data_path=[TRAINING DATASET]
$ python3 -m torch.distributed.launch --nproc_per_node=<n_gpus> --master_port=8888 train_encoder.py --data_path=[TRAINING DATASET]`
```

### 3.2. Pretrained Model Checkpoint
To use pretrained model checkpoint, download following files and place at the project directory.
- model checkpoint ([LINK](https://drive.google.com/file/d/1QpzO2f4a8lPgsRpbOMLzkNydf_N3sWnC/view?usp=sharing))
- latent statistics ([LINK](https://drive.google.com/file/d/1MgkCn2ZmciPjfwhN-qQPAIC2p0JS-ULy/view?usp=sharing))

### 3.3. Anomaly Detection
Project images to latent space and detect anomalies:

```bash
# Run Demo
$ python projector.py --query_save_path=./demos # (approx. 5 minutes per CT scan)

# Run on custom dataset
$ python projector.py --query_save_path=[YOUR_DATASET_PATH]
```



## 4. License
This project is licensed under the **GNU GENERAL PUBLIC LICENSE Version 3 (GPLv3)**.
For business inquiries: namkugkim@gmail.com.