# Generative_Diffusion_Models_for_3D_Geometric_Objects

This repository contains the code for the project: Generative Diffusion Models for 3D Geometric Objects, 
The project represents the P7 in the Masters of Science in Engineering (MSE) study program at FHNW.


## Docker

Build the geoshapes docker image from Dockerfile:
``` sh
docker build -f Dockerfile.slurm -t geoshapes .
```

Start the geoshapes container in bash
``` sh
docker run -it --rm -v $(pwd):/app/ --runtime nvidia geoshapes bash
```

## Dataset Reshaping

In the project various sizes

Train Set :
``` sh
python reshape_dataset.py ----source ./data/train256/ --destination ./data/train128/ --size 128
```

Test Set :
``` sh
python reshape_dataset.py ----source ./data/test256/ --destination ./data/test128/ --size 128
```

