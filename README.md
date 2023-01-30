<img src="./design/logo.png">


### easy sequence classification training and inference with transformers
\
\
\
\
## Overview
The sequifier package enables:
  - the extraction of sequences for training
  - the configuration and training of a transformer classification model
  - inference on data with a trained model


## Other materials 
If you want to first get a more specific understanding of the transformer architecture, have a look at
the [Wikipedia article.](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))

If you want to see a benchmark on a small synthetic dataset with 10k cases, agains a random forest,
an xgboost model and a logistic regression, check out [this notebook.](./documentation/demos/benchmark-small-data.ipynb)


## Complete example how to build and apply a transformer sequence classifier with sequifier

1. create a conda environment with python 3.9.12, activate and run
```console
pip install sequifier
```
2. create a new project folder (at a path referred to as `PROJECT PATH` later) and a `configs` subfolder
3. copy default configs from repository for preprocessing, training and inference
4. adapt preprocess config to take the path to the data you want to preprocess
5. run 
```console
sequifier --preprocess --config_path=[PROJECT PATH]/configs/preprocess.yaml --project_path=[PROJECT PATH]
```
6. the preprocessing step outputs a "data driven config" at `[PROJECT PATH]/configs/ddconfigs/[FILE NAME]`. It contains the number of classes found in the data, a map of classes to indices and the oaths to train, validation and test splits of data. Adapt the `dd_config` parameter in `train-on-preprocessed.yaml` and `infer.yaml` in to the path `[PROJECT PATH]/configs/ddconfigs/[FILE NAME]`
7. run
```console
sequifier --train --on-preprocessed --config_path=[PROJECT PATH]/configs/train-on-preprocessed.yaml --project_path=[PROJECT PATH]
```
8. adapt `inference_data_path` in `infer.yaml`
9. run
```console
sequifier --infer --config_path=[PROJECT PATH]/configs/infer.yaml --project_path=[PROJECT PATH]
```
10. find your predictions at `[PROJECT PATH]/outputs/predictions/sequifier-default-best_predictions.csv`


## More detailed explanations of the three steps
#### Preprocessing of data into sequences for training

The preprocessing step is designed for scenarios where for long series
of events, the prediction of the next event from the previous N events  is of interest.
In cases of sequences where only the last item is a valid target, the preprocessing
step should not be executed.

This step presupposes input data with three columns: "sequenceId", "itemId" and "timesort".
"sequenceId" and "itemId" identify sequence and item, and the timesort column must
provide values that enable sequential sorting. Often this will simply be a timestamp.
You can find an example of the preprocessing input data at [documentation/example_inputs/preprocessing_input.csv](./documentation/example_inputs/preprocessing_input.csv)

The data can then be processed and split into training, validation and testing datasets of all
valid subsequences in the original data with the command:

```console
sequifier --preprocess --config_path=[CONFIG PATH] --project_path=[PROJECT PATH]
```

The config path specifies the path to the preprocessing config and the project
path the path to the (preferably empty) folder the output files of the different
steps are written to.

The default config can be found on this path:

[configs/preprocess.yaml](./configs/preprocess.yaml)



#### Configuring and training the sequence classification model

The training step is executed with the command:

```console
sequifier --train --config_path=[CONFIG PATH] --project_path=[PROJECT PATH]
```

If the data on which the model is trained comes from the preprocessing step, the flag

```console
--on-preprocessed
```

should also be added.

If the training data does not come from the preprocessing step, both train and validation
data have to take the form of a csv file with the columns "sequenceId", [SEQ LENGTH], [SEQ LENGTH - 1],...,"1", "target".
You can find an example of the preprocessing input data at [documentation/example_inputs/training_input.csv](./documentation/example_inputs/training_input.csv)

The training step is configured using the config. The two default configs can be found here:

[configs/train.yaml](./configs/train.yaml)

[configs/train-on-preprocessed.yaml](./configs/train-on-preprocessed.yaml)

depending on whether the preprocessing step was executed.


#### Inferring on test data using the trained model

Inference is done using the command:

```console
sequifier --infer --config_path=[CONFIG PATH] --project_path=[PROJECT PATH]
```

and configured using a config file. The default version can be found here:

[configs/infer.yaml](./configs/infer.yaml)


