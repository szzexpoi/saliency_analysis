# What Do Deep Saliency Models Learn about Visual Attention?

This repository implements an analytic framework for investigating the visual semantics learned by deep saliency models and how they contribute to saliency prediction. It is widely applicable to various state-of-the-art models, and in this repository, we use SALICON, DINet, and TranSalNet as examples.

### Requirements
1. Requirements for Pytorch. We use Pytorch 1.9.0 in our experiments.
2. Requirements for Tensorflow (for monitoring training process only).
3. Python 3.6+
4. Jupyter Notebook
5. You may need to install the OpenCV package (CV2) for Python.

### Data Preparation
1. We use [SALICON](http://salicon.net/challenge-2017/) to train different saliency models, which is currently the largest dataset for saliency prediction. Please follow the link to download the images, fixation maps and saliency maps.
2. For probing the visual semantics learned by saliency models, we use [Visual Genome](http://visualgenome.org/api/v0/api_home.html) with fine-grained annotations. Please download the images and scene graph annotations from the link (we only use the training set).
3. For preprocessed file such as a list of consolidated semantics (e.g., semantics after merging plurals and singular), pretrained model weights, and intermediate results, please refer to our [Google Drive](https://drive.google.com/drive/folders/1chgd9fOrAeU7KQNpaJMi0ri8ZqhrvFEb?usp=sharing)

### Model Training
If you wish to use our framework from scratch, the first step would be training the saliency models with factorization (use DINet as an example):
```
python saliency_modeling.py --mode train --img_dir $IMG_DIR --fix_dir $FIX_DIR --anno_dir $SAL_DIR --checkpoint $CKPT --use_proto 1 --model dinet
```
where **IMG_DIR**, **FIX_DIR**, **SAL_DIR** are directories to the images, fixation maps, and saliency maps, **$CKPT** is the directory for storing checkpoint.

After the first phase of training, you need to reformulate the inference process of saliency prediction:
```
python saliency_modeling.py --mode train --img_dir $IMG_DIR --fix_dir $FIX_DIR --anno_dir $SAL_DIR --checkpoint $CKPT_FT --use_proto 1 --model dinet --second_phase 1 --weights $CKPT/model_best.pth
```
where **$CKPT_FT** is another directory for storing the new checkpoint.

Upon obtaining the model weights for both phases of training, an adaptive threshold should be computed on each basis:
```
python saliency_modeling.py --mode compute_threshold --img_dir $IMG_DIR --fix_dir $FIX_DIR --anno_dir $SAL_DIR --use_proto 1 --model dinet --weights $CKPT/model_best.pth
```

### Prototype Dissection
A key component of our framework is to associate implicit features with interpretable semantics, which is through analyzing the alignment between probabilistic distribution of bases and segmentation in Visual Genome:
```
python saliency_modeling.py --img_dir $VG_IMG --sg_dir $VG_graph --weights $CKPT/model_best.pth --model dinet
```
where **VG_IMG**, **VG_graph** are directories to the images and scene graphs for Visual Genome dataset.

### Analyzing the Interactions
For studying the spatial- and semantic-level interactions for saliency prediction, self-attention needs to be introduced during the reformulation of saliency prediction:
```
python saliency_modeling.py --mode train --img_dir $IMG_DIR --fix_dir $FIX_DIR --anno_dir $SAL_DIR --checkpoint $CKPT_Transformer --use_proto 1 --model dinet --second_phase 1 --use_interaction 1 --weights $CKPT/model_best.pth
```
where **$CKPT_Transformer** is the directory for storing the checkpoint for interaction analysis.

To measure alignment between spatial interactions and saliency distribution:
```
python saliency_modeling.py --mode interaction_analysis --img_dir $IMG_DIR --fix_dir $FIX_DIR --anno_dir $SAL_DIR --use_proto 1 --model dinet --use_interaction 1 --weights $CKPT_Transformer/model_best.pth
```

### Quantifying Contributions of Semantics
After running the aforementioned steps or downloading intermediate results from our drive, you can follow the Jupyter Notebook to compute the quantitative contributions of semantics.

### Reference
If you use our code or data, please cite our paper:
```
@inproceedings{saliency_analysis_nips23,
 author = {Chen, Shi and Jiang, Ming and Zhao, Qi},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {What Do Deep Saliency Models Learn about Visual Attention?},
 year = {2023}
}

```
