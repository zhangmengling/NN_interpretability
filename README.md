# NN_intrepretability

This is a tool to test neural network interpretability with decision tree as an agent. 

## Installation:
The tool is based on python3. 
```
pip install numpy, random, csv, copy, itertools, sklearn, tensorflow
```
We apply tensorflow==1.14.0. If tensorflow version 2.x is used, it can also run with 
```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```
## tutorial
The testing_allinput.py provide an example of testing interpretability of the neural network trained on Census dataset against all inputs.
```
python ./tutorial/testing_allinput.py
```
