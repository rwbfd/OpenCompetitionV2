# Open Competition
## General

This is a library that aims to provide a uniform interface for common data competition or, more generally, for predictive modeling. In general, four areas 
will be covered. 

1. Tabular data mining.
2. NLP related tasks.
3. CV related tasks.
4. AutoML: RL and Neural Architecture Search.
## Installation
Currently, the best way to set up the environment is by using the pre-built docker image.

The docker images can be pulled by 

`docker pull ranwangmath/opencompetition:0.3`

To run the docker images, use 

`docker run -it --rm --gpus all --ipc=host --shm-size 10g -v /home/user/code:/workspace -p 8888:8888 ranwangmath/opencompetition:1.0`

**Note that the docker image is HUGE. Therefore DO NOT forget to use the --rm flag, or it will eat up the disk space!**

Once inside the image, to use the library (it is not installed by default), use the following command

1. `git clone https://github.com/rwbfd/OpenCompetitionV2.git` 
2. `cd OpenCompetitionV2`
3. `bash ./setup.sh`

Please pay attention:
1. The docker image is huge.
2. If one wishes to use GPU, please use Nvidia docker.
3. The image is being constantly updated.  
4. If one wishes to use cuml, one must use the command `conda activate rapids-0.16`.
Once it is done, one should use the command `conda deactivate`.

**Unfortunately, the above command for activating rapids-0.16 does not yet work due to the fact that it is not possible to activate conda environment. Any help is welcomed.**
### General Functionality
The following functionalities have already been implemented. 

1. A general framework for training neural networks. This is adapted from 
HuggingFace Transformer(https://github.com/huggingface/transformers.git). However, some changes are made so that it accepts general PyTorch models.


### Tabular Data Mining
The following functionalities have already been implemented. 

1. Encoders: This module contains commonly used encoders to transform discrete and continuous variables into different forms. This module also includes dimension reduction
functionalities, mainly PCA and tSNE. Note that in this module, unlike sklearn conventions, 
we directly create names for the newly generated variables (might need to be changed).
2. Model Fitter: This module provides a model fitter for XgBoost, LightGBM, CatBoost, 
Logistic Regression, KNN, Random Forest (based on LGB implementation), and SVM.
The model fitter allows for easy training and cross-validation. For hyperparameter search,
we have wrapped hyperopt, which performs sequential Bayesian search. Some default search spaces
have been provided. 
3. Functions to find maximal linear dependent sets. Note that this part is written in C++. 

## Future Implementation Plans
Based on priorities, the following functionalities will be added. 

1. Variable selection methods for tabular data. 
2. Deep learning methods for tabular data. Include entity embeddings, variations of transformers, FM models.
3. TabNet and associated NAS methods. Currently, only a differentiable version (https://arxiv.org/abs/1910.04465) will 
be implemented. 
4. More optimizers for the general deep learning framework.
5. Adversarial training methods for deep learning. 
6. Unsupervised data augmentation (only for classification problems; see https://arxiv.org/abs/1904.12848 for details).
7. DAE and common VAE for tabular data. 
8. A general framework for training Electra (https://arxiv.org/abs/2003.10555) type pretrained model. 

After all these have been implemented, NLP related tasks will be implemented. 

NOTE: These plans are only for provisioning. If one would like to make new requests, please post in the issues session. Better yet, we appreciate any contribution, especially in reinforcement learning algorithms, CV and neural architecture search. 
 
