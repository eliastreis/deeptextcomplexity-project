# Advanced Projects at the Quality and Usability Lab
## Deep Text Complexity Metric Challenge
 
The goal of this project was to create a model that can predict the
MOS score of German sentences. This file explains how our model can be used to predict
the MOS score of sentences and how to train a new model.

Group members:

Dominic Paul Christian Heil
<br/> Elias Tries
<br/> Lisa-Marie Wienbrandt


### Code

0. CD in the directory in which the `main.py` file is located:

```bash
cd group1
```

1. Install requirements using

```bash
pip install -r requirements.txt
```

If the installation of Tensorflow fails on your system, then install a Tensorflow version that can run on your system
with version >=2.3.0.

If the download of the spacy language package fails, then use `python -m spacy download de_core_news_lg` to download it manually.

2. **Evaluate** a model, **train** a model or **continue training** a model

To **predict** the MOS scores of sentences run:

```bash
python main.py --mode evaluate --input test.csv [--model PATH] [--fasttext N]
```

The file `test.csv` must be in CSV format and must contain the columns "sentence" and "sent_id".
The predicted MOS values of every sentence and the sentence ID "sent_id" will be printed to 
stdout as well as to an output file output.csv. The output file will have the two columns "sent_id" and "mos", where
the column "sent_id" contains the sentence id of the sentence and the column "mos" contains the predicted MOS value of 
the sentence. 

By default, the model with the name `model.hdf5` will be loaded and thus the model must be in the same directory as
the `main.py` file. If you want to load a model from a different path, then set the optional parameter 
`--model PATH` and the model located at the path `PATH` will be used for evaluation. Note that the model must be in
"hdf5" format.

By default, the word vector size used during evaluation is set to 300. If the model was trained using a different
word vector size, then set the optional parameter `--fasttext N`, with N>0 and N<=300. This parameter should only be 
changed if it is known that the model was trained using a different word vector dimension.


---

To **train a new model** run:

```bash
python main.py --mode train --input training.xlsx --testSet testSet.xlsx [--fasttext N]
```

The input file at the path specified by parameter `--input` must be in Excel format and must contain the two columns "MOS" and "Sentence".
These sentences will be used during training and will be split into a training and a validation set.
Additionally, a path to a test set must be specified using parameter `--testSet`. The test set file must be in the same
format as the file at the path specified by `--input`. If no test set is available, then the test set can also be set to the
same path as the input file. The test set does not affect training and is only used to evaluate the model.
Optionally, you can specify the parameter `--fasttext N`, with N>0 and N<=300, to change the dimension of the 
FastText model. The default is 300. If you select a value other than 300, then you might need a lot of memory to
reduce the model to a lower dimension (possibly up to 32GB of RAM).

Note:

If you want to change the architecture of the new model, then go into the function `train_new_model` in the `main.py` 
file and add, remove or change layers. In this function you can also change the learning rate and the optimizer used
for training.

Checkpointed models will be saved to the directory `models` relative to the execution path.

---

To **continue training a model** run:
```bash
python main.py --mode continue_training --input training.xlsx --model model.hdf5 --testSet testSet.xlsx [--fasttext N]
```


The input file at the path specified by parameter `--input` must be in Excel format and must contain the two columns "MOS" and "Sentence".
These sentences will be used during training and will be split into a training and a validation set.
Additionally, a path to a test set must be specified using parameter `--testSet`. The test set file must be in the same
format as the file at the path specified by `--input`. If no test set is available, then the test set can also be set to the
same path as the input file. The test set does not affect training and is only used to evaluate the model.
The path to the model that will be used as initial start point must be specified in the `--model` parameter.

If the word vector size used to train the model specified in the `--model` parameter is different from 300,
then the parameter `--fasttext N` must be set with N being the word vector size used for the initial training.
Note, if the `--fasttext` value is lower than 300, then you might need a lot of memory to
reduce the model to a lower dimension (possibly up to 32GB of RAM).

Note:

Checkpointed models will be saved to the directory `models` relative to the execution path.
