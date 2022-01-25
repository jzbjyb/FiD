#!/usr/bin/env bash

conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install transformers==3.0.2
pip install tensorboard==2.8.0
conda install faiss-gpu cudatoolkit=11.0 -c pytorch
