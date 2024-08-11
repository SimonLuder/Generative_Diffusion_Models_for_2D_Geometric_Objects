# Generative_Diffusion_Models_for_2D_Geometric_Objects

This repository contains the code for the project: Generative Diffusion Models for 2D Geometric Objects, 
The project represents the P7 in the Masters of Science in Engineering (MSE) study program at FHNW.


## Structure
------------
```
├── README.md                           <- The file you are reading right now
│
├── train.py                            <- The main file for training the models
│
├── test.py                             <- Code for testing and validating the models during and after training
│
├── inference.py                        <- Use this script to generate conditioned new samples with a trained model
│
├── noising_test.py                     <- Noise images with this script
│
├── dataset.py                          <- Contains the dataset classes
│
├── embedding.py                        <- Code for the various embedding classes
│
├── data                                <- Place the datasets for training here
│
├── model              
│   ├── ddpm.py                         <- Contains the ddpm model class
│   ├── modules.py                      <- Contains the module classes for the UNet
│   └── UNet.py                         <- Contains the UNet class
│
├── notebooks  
│   ├── gradient_visualization.ipynb    <- Notebook for visualizing the gradient of an image
│   ├── report_visualizations.ipynb     <- Notebook with general visualizations for the report
│   └── results.ipynb                   <- Contains the results from the training for the report
│
├── tools                 
│   ├── reshape_dataset.py              <- Run this script to change the image resolution of a selected dataset
│   ├── setup_classes.py                <- Run this script to create a conditional class dataset (outdated)
│   └── subsample_dataset.py            <- Run this script to create a subset of a selected dataset
│
├── utils                    
│   ├── metrics.py                      <- Contains metrics for evaluation 
│   ├── train_test_utils.py             <- Contains general utils for training and testing
│   └── wandb.py                        <- Contains WandB functionalities for logging
│
├── dockerfile
│   └── Dockerfile.slurm                <- Dockerfile for training models on slurm                  
│
├── requirements.txt                    <- The requirements file
│
├── images                              <- Contains images created for the report
│
└── runs                                <- This folder contains the individual trained models and logs during training
```
------------


## Docker
To train on 

Build the geoshapes docker image from Dockerfile:
``` sh
docker build -f Dockerfile.slurm -t geoshapes .
```

Start the geoshapes container in bash
``` sh
docker run -it --rm -v .:/app/ --gpus all geoshapes bash
```

Transform the docker image to .tar file
``` sh
docker save geoshapes > geoshapes.tar

```

------------
## Dataset reshaping

To create a dataset version with different image sizes, run the reshape_dataset.py script. 
Note that the config needs to be adapted to train with the alternative image sizes.

Train dataset :
``` sh
python reshape_dataset.py --source ./data/train256/ --destination ./data/train64/ --size 64
python reshape_dataset.py --source ./data/train256/ --destination ./data/train32/ --size 32
```

Validation dataset :
``` sh
python reshape_dataset.py --source ./data/val256/ --destination ./data/val64/ --size 64
python reshape_dataset.py --source ./data/val256/ --destination ./data/val32/ --size 32
```

Test dataset :
``` sh
python reshape_dataset.py --source ./data/test256/ --destination ./data/test64/ --size 64
python reshape_dataset.py --source ./data/test256/ --destination ./data/test32/ --size 32
```

### slurm
To train the models on the SLURM cluster, the path of the individual subsets must be further adapted, when creating alternative image sizes in the dataset.

Train dataset :
``` sh
python reshape_dataset.py --source ./data/train256/ --destination ./slurm/data/train64/ --size 64 --custom_destination_path /workspace/data/train64/
python reshape_dataset.py --source ./data/train256/ --destination ./slurm/data/train32/ --size 32 --custom_destination_path /workspace/data/train32/
```

Validation dataset :
``` sh
python reshape_dataset.py --source ./data/val256/ --destination ./slurm/data/val64/ --size 64 --custom_destination_path /workspace/data/val64/
python reshape_dataset.py --source ./data/val256/ --destination ./slurm/data/val32/ --size 32 --custom_destination_path /workspace/data/val32/
```

Test dataset :
``` sh
python reshape_dataset.py --source ./data/test256/ --destination ./slurm/data/test64/ --size 64 --custom_destination_path /workspace/data/test64/
python reshape_dataset.py --source ./data/test256/ --destination ./slurm/data/test32/ --size 32 --custom_destination_path /workspace/data/test32/
```
------------
## Subset generation
To create a small dataset for training, the subsample_dataset.py script can be executed. This can be executed via the shell with the following commands. 
Create a subsample of the dataset
``` sh
python ./utils/dataset/subsample_dataset.py --source "data/train256/" --n 100
python ./utils/dataset/subsample_dataset.py --source "data/train64/" --n 100
python ./utils/dataset/subsample_dataset.py --source "data/train32/" --n 100
```

### slurm
``` sh
python ./utils/dataset/subsample_dataset.py --source "slurm/data/train256/" --n 100
python ./utils/dataset/subsample_dataset.py --source "slurm/data/train64/" --n 100
python ./utils/dataset/subsample_dataset.py --source "slurm/data/train32/" --n 100
