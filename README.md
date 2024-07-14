# Cross-domain and Cross-dimension Learning for Image-to-Graph Transformers

## Acknowledgement

This repository is largely based on the relationformer repository and paper. For more information we refer to https://github.com/suprosanna/relationformer and the corresponding paper.

## Requirements
* CUDA>=9.2
* PyTorch>=1.7.1

For other system requirements please follow

```bash
pip install -r requirements.txt
```

### Compiling CUDA operators
```bash
cd ./models/ops
python setup.py install
```


## Code Usage

## 1. Dataset preparation

Please download [20 US Cities dataset](https://github.com/songtaohe/Sat2Graph/tree/master/prepare_dataset) and organize them as following:

```
code_root/
└── data/
    └── 20cities/
```

The set-name (e.g. 'global_diverse_cities') will be the name for the data folder. The files must be in the form '<Cityname>_region_<id>_<rest>' where rest can be one of ['refine_gt_graph.p', 'gt.png', 'sat.png'].

## 2. Training

#### 2.1 Prepare config file

The config file can be found at `.configs/road_rgb_2D.yaml`. Make custom changes if necessary.

## 3. Evaluation

Once you have the config file and trained model of Relation, run following command to evaluate it on test set:

```bash
python test.py --config configs/road_rgb_2D.yaml --checkpoint ./trained_weights/last_checkpoint.pt
```