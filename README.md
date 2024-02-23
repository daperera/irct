# IRCT - Invariance-based semi-supervised representation learning for sound event detection

This repository gives the code to reproduce the experiments described in the paper Invariance-based semi-supervised representation learning for sound event detection.

## Introduction

Experimental and theoretical evidences suggest that invariance constraints can improve the performance and generalization capabilities of a classification model. While invariance-based regularization has become part of the standard toolbelt of machine learning practitioners, this regularization is usually applied near the decision layers or at the end of the feature extracting layers of a deep classification network. However, the optimal placement of invariance constraints inside a deep classifier is yet an open question. In particular, it would be beneficial to link it to the structural properties of the network (e.g. its architecture), or its dynamical properties (e.g. the effectively used volume of its latent spaces). In this article, we use the experimental framework of DCASE Task4 in order to initiate an investigation on these aspects. 

<div  align="center">    
<image src="./misc/training_framework.png"  width="500" alt="Training framework" />

**Training framework**
</div>

We show experimentally that input or output regularization is not optimal in this setting, and that proper internal regularization improves the baseline system for this task. Moreover, our results suggest that the optimal placement of this regularization is non trivially related to the diversity of the set of audio augmentations and to the target evaluation metric. We also study this behavior through the lens of the classifier’s implicit pretext tasks, and its latent representation encoding complexity.

## Training

Below, we provide instructions to reproduce the results from the article.

### Code

First clone the official DCASE 2023 Task 4 repository.

```bash
git clone git@github.com:DCASE-REPO/DESED_task.git
```

Then you can clone our repository. Please note that some files from the DCASE repository will be overrided.

```bash
git clone git@github.com:daperera/irct.git
```

### Data

The training and validation data is obtained from the DCSAE2023 task4 [DESED dataset](https://github.com/turpaultn/DESED). Unfortunately not all data is available to download. However, you can ask for help from the DCASE committee to obtain the missing data. 

Once the DESED dataset is downloaded at your prefered location, please update correspondingly the paths in the training configuration files stored in the folder /recipes/dcase2023_task4_baseline/confs/.

### Training script

The training scripts is given in the file training_script.sh. Please run this file line by line if necessary. Each model's training takes up to 30h on a V100-16GB. Please note that using a different architecture may lead to slightly different results from the ones we report.

```bash
cd recipes/dcase2023_task4_baseline
sh training_script.sh
```

After training the official baseline (corresponding to the script /recipes/dcase2023_task4_baseline/train_sed.py), and before training the extraction probes (corresponding to the script /recipes/dcase2023_task4_baseline/probe_train_sed.py), please move the training checkpoints to the folder /recipes/dcase2023_task4_baseline/ckpt/default/, and update the path in the probe configuration files (line corresponding to the entry "pretrained_path"). Alternatively, you can use the provided checkpoints (stored in folder /recipes/dcase2023_task4_baseline/ckpt/default/).


## Plot

The notebook /plot.ipynb provides scripts to reproduce the Figures from the article. We report the scores that we obtained in the file /misc/scores.txt, so that this notebook can be used as is, without retraining.

## Set of augmentations

We indicate below the list of audio augmentations used in this study, along with their range of parameters.

| Set       | Augmentation      | Parameter     | Range                   | 
|---|---|---|---|
| $\tau_0$  | Mixup             | $\alpha$      | [0.5, 1]                |
| $\tau_1$  | Time masking      | duration      | 100 ms                  |
|           |                   | n masks       | [0, 5]                  |
|           | Time shifting     | shift         | [0, 50 ms]              |
|           | Pitch shifting    | shift         | [-250 cents, 250 cents] |
|$\tau_2$   | Frequency masking | log Mel bands | [0, 20]                 |
|           | FilterAugment     | attenuation   | [-6.5 dB, 7 dB]         |
|           |                   | n bands       | [2, 5]                  |
