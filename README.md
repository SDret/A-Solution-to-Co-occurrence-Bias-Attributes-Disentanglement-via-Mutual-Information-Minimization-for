# About this code
This is the official implementation of IJCAI2023 paper 
'A Solution to Co-occurrence Bias: Attributes Disentanglement via Mutual Information Minimization for Pedestrian Attribute Recognition'.

To make a fair comparison, our work strictly follows the benchmark protocol stated in the strong baseline work in https://github.com/valencebond/Rethinking_of_PAR by employing the released public code of this work for dataset partitioning, dataloader generation, backbones config setting and metric setup, etc.

Thus, to efficiently implement the method and reproduce the benchmark results without further efforts and misunderstandings, we suggest readers to look through the public code
of this adopted baseline work in Github at https://github.com/valencebond/Rethinking_of_PAR and implement our method directly onto this code by simply replacing serval .py files of model framework and training configs:
• Replacing the original 'train.py' file with ours.
• Replacing the original 'batch_engine.py' file with ours.
• Replacing the original 'models/base_block.py' file with
ours.
• Replacing the original 'configs\pedes_baseline' folder
by ours.
• Putting the 'convnext.py' file under the path 'models/backbone/' for testing on ConvNeXt-base.


## Environment
Please set the environment as

Pytorch == 1.10.1+cu102 

numpy == 1.19.5 python == 3.6.9 64- bit.

The experiments in main text are conducted on a single NVIDIA Tesla V100 32G.

## Datasets
Before running the codes, you need to download all datasets (PA100k, RAP and PETA) from their official releases, and struct the downloaded ones exactly as that is required in https://github.com/valencebond/Rethinking_of_PAR. For the PETAzs and RAPzs datasets, this baseline work already provides generating files under the path of 'data'.

## Start training
To run the model training and testing is simple:

```
CUDA VISIBLE_DEVICES=0 python train.py cfg ./configs/pedes_baseline/$DATASET_CONFIG$
```
DATASET CONFIG can be 'pa100k.yaml', 'peta_zs.yaml',
'peta.yaml', 'rap zs.yaml' or 'rapv1.yaml'. They are all provided in 'pedes_baseline'.

