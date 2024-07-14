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

## 2. Training

#### 2.1 Prepare config file

The config file can be found at `.configs/synth_3D.yaml`. Make custom changes if necessary.


## 3. Evaluation

Once you have the config file and trained model of Relation, run following command to evaluate it on test set:

```bash
python run_batch_inference_eval.py --config configs/synth_3D.yaml --model ./trained_weights/last_checkpoint.pt --eval
```
