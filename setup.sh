#!/usr/bin/env bash

conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
#pip install transformers==3.0.2
pip install beir
pip install transformers==4.15.0
#pip install tensorboard==2.8.0
conda install faiss-gpu cudatoolkit=11.0 -c pytorch
pip install wandb==0.12.10
pip install bertviz
pip install jupyterlab==3.2.9
pip install ipywidgets==7.6.5
pip install rouge-score==0.0.4
pip install fairscale==0.4.5
pip install entmax==1.0
pip install datasets==1.18.3
pip install spacy==3.2.3
