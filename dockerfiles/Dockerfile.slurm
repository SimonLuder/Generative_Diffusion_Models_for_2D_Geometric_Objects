# get base container
# FROM pytorch/pytorch:latest  
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# install necessary requirements (e.g. git)
RUN apt-get -y update  
RUN apt-get -y install git && \  
    apt-get -y install build-essential

# install needed libraries from requirements.txt file
COPY ./requirements.txt ./requirements.txt  
RUN pip install --no-cache-dir -r requirements.txt  

# install clip library directly from git repository
RUN pip install git+https://github.com/openai/CLIP.git 
