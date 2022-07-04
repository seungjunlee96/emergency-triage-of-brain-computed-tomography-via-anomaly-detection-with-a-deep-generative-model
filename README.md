# Emergency triage of brain computed tomography via anomaly detection with a deep generative-model

## Overview
Triage is essential for the early diagnosis and reporting of neurologic emergencies. Herein, we report the development of an anomaly detection algorithm  with a deep generative model trained on brain computed tomography (CT) images of healthy individuals that reprioritizes radiology worklists and provides lesion attention maps for brain CT images with critical findings.

## 1. System requirements

The package development version is tested on Linux (Ubuntu >= 16.04) operating systems. The developmental version of the package has been tested on the following systems:

* Python >= 3.8.5
* PyTorch >= 1.3.1
* CUDA >= 10.1/10.2

This implementation requires GPU acceleration for optimal performance. 
We have tested the code with the following specs:

- CPU: 12 cores, Intel(R) Xeon(R) CPU E5-2603 v4 @ 1.70GHz
- RAM: 16 GB
- GPU: 4 GPUs, Titan RTX 8000 

## 2. Installation guide

### Docker installation
Please follow the links below to install docker.

* Docker installation: https://docs.docker.com/engine/install/ubuntu/
* Nvidia Docker installation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Docker container
Using the following codes, you can install docker container.

```
$ sudo docker build ./ -t ubuntu/cnstylegan:latest # Be sure to be inside the project directory
$ sudo docker run --gpus all --restart=always --shm-size=128G -dit --name cnstylegan -v /mnt:/mnt -p 8888:8888 ubuntu/cnstylegan
$ sudo docker exec -it cnstylegan bash
```
which should install in about 30 minutes.

## 3.Usage
You can either train the model from scratch or use pretrained model checkpoint

### Training
In order to train the model from scratch, use following codes:

```
python3 -m torch.distributed.launch --nproc_per_node=<n_gpus> --master_port=8888 train.py --data_path=[TRAINING DATASET]
python3 -m torch.distributed.launch --nproc_per_node=<n_gpus> --master_port=8888 train_encoder.py --data_path=[TRAINING DATASET]`
```

### Pretrained model checkpoint
To use pretrained model checkpoint, download following files and place at the project directory.
- model checkpoint: https://drive.google.com/file/d/1k6At7zNWnbXO4g8Mw2bOq9jOyxV1ArOU/view?usp=sharing
- latent statistics: https://drive.google.com/file/d/1kTZXMOBipL8xgKHPK5Q7ymt38aRg-jEw/view?usp=sharing

### Project images to latent space
Use following code to project images to latent space and perform anomaly detection

`$ python projector.py --query_save_path=[DATASET_PATH]`

- To run the demo code
`$ python projector.py --query_save_path=./demos`


## License
This project is covered under the **GNU GENERAL PUBLIC LICENSE Version 3 (GPLv3)**.
