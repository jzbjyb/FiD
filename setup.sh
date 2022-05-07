#!/usr/bin/env bash

conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
#pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

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
pip install ujson==5.2.0
pip install cupy-cuda110==10.3.1
pip install indexed==1.2.1
pip install https://data.pyg.org/whl/torch-1.7.0%2Bcu110/torch_scatter-2.0.7-cp37-cp37m-linux_x86_64.whl
