# TO-DO List

## General Situation
In general, the current project is still under early development. It is expected that a full usable demo, along with a docker environment to be presented before  2020-11-14. 

## To-do List

### Urgent
- Currently, the docker image cannot support rapids-ai yet. This is because, for some reason, inside
the docker container, conda activate cannot be performed. Someone needs to fix this problem. 
- None of the encoders have been fully tested. Please use the data in examples to perform the test. 

### Helpful
- In tabular/model_fitter.py, it is helpful for this Opt (such as LGBOpt) to add a `__repr__` methods
- In tabular/model_fitter.py, a seed option should be added for each model for reproduction.
- In tabular/model_fitter.py, such as in the LGBFitter.train method, some options given be hyperopt might work well together. Therefore some guards should be placed. 
- In tabular/encoders, some encoders should only be applied to certain types of variables. For example, 
category variables should only be applied to variables that starts with discrete. A warning (using `warning.warn()`))
should be added. 
- In tabular/model_fitter.py, cuml fitter's train_k_fold methods should return, at the fourth position, the trained models. 
### Somewhat helpful
- Please complete the docstring for all the encoder's configs. 

### Additional Functionality
- All the model only supports binary classification now. It would be nice to add options for other types of targets.
- Entity embedding with vincinal info and not. 
- Three data loaders for pandas dataframe splitted into three parts: original feature, entity embedding with and without vincinal information. 
- Add a weight parameter for lamb optimizer for that it includes the normal adam. 