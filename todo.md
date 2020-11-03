# TO-DO List

Currently, the program is still somewhat preliminary. Therefore, significant structural changes will likely occur.

## General To-do list
### Missing parts
1. A setup up script and a docker environment. 
2. All the main functions need doc.
3. DiscretizeEncoder should include more cluster-based methods. 
4. All the examples have been used for binary classification. More general settings (multi-variate classification and regression) should be added.

### Refactoring needs 
1. In the encoder file, discretize encoder is for discretizing continuous variables into discrete ones. However, boosted tree
encoders and anomaly encoder (based on isolation forest) should also provide similar functionality. 

### Specific To-do list

1. Category Encoder transform methods must be implemented. 
2. Simple entity embedding.
### Refactor
1. Encoders in encoder.py can be organized. 

### Test
1. Category encoder needs to be tested. 
2. Variable selector must be tested.