# Group 7 Project: Using Multiple Aggregator Function with GraphSAGE
## Authors: Ben Jordan, Peter Carbone, Robert Boris
## Original Author: William Leif


Basic reference TensorFlow implementation of [GraphSAGE]
(https://github.com/williamleif/GraphSAGE).
Original PyTorch implementation of [GraphSAGE]
(https://github.com/williamleif/graphsage-simple/).


This code was forked from the PyTorch implementation of GraphSAGE and has been modified to accomodate our experiment. The original PyTorch implementation contained an unsupervised variant of the code that used a different dataset, however only the supervised variant remains in our version. All code that is not relevant to our project implementation has been scrapped from the original.


### Requirements


python >= 3.10.8 is required to run.

run `pip -r requirements.txt` to install all required dependencies.

### Running examples


From 'Machine-Learning-Project', run one of the following:

To test one aggregator combination, enter `python -m graphsage.model aggr1 aggr2` where aggr1 and aggr2 can be either `mean` or `max`.

To test every aggregator combination, enter `python -m graphsage.model all`, this will print a table with the F1-Score for every combination in this experiment.

Bash scripts to run each combination are provided for reference and convenience. 