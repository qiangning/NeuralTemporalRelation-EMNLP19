# CogCompTime 2.0
This package implements the algorithm proposed in 
- “An Improved Neural Baseline for Temporal Relation Extraction.” Qiang Ning, Sanjay Subramanian, and Dan Roth. EMNLP'19 (short paper).

Notes:
- We don't include the ILP inference step in this repository since we used JAVA and Gurobi to do this for the paper. However, we do include a folder called `output/` in this repo (it will also be generated from the code here by default), which can be used by an ILP solver. Please see below for more information about how to read the `output/` folder.
- The performance produced by this package is slightly different to the reported numbers in the original paper, but the conclusions of the paper still hold. Feel free to use the numbers produced in this package in your paper(s). We will discuss these issues in detail below.
- The paper and this package both don't include an event extraction module. That means they cannot be applied to raw text *directly*. If you're interested in processing raw text, please check our earlier repo of [CogCompTime](https://github.com/qiangning/CogCompTime) or another [repo](https://github.com/rujunhan/EMNLP-2019) for a recent [paper](https://www.aclweb.org/anthology/D19-1041.pdf) that are both end-to-end. You can use their event extraction module as a preprocessing step before using this package.

Please contact Qiang Ning at qiangn@allenai.org if you have any questions about the paper and/or this package.

# Prerequisites
The following python packages are installed in the conda environment that I used to run this package, so please be aware that (1) maybe not all of them are required; (2) the list might not be exhaustive.
- allennlp
- click
- nltk
- pytorch
- pytorch-pretrained-bert

The complete list of installed packages can be found [here](https://github.com/qiangning/NeuralTemporalRelation-EMNLP19/blob/master/installed_packages.txt).

Make sure you have also downloaded the `ser/` folder:
```
wget https://cogcomp-public-data.s3.amazonaws.com/processed/NeuralTemporal-EMNLP19-ser/NeuralTemporal-EMNLP19-ser.tgz
tar xzvf NeuralTemporal-EMNLP19-ser.tgz
```

# Run
If you want to use our pretrained models to run experiments, you can use the following two commands:
```
./reproduce_proposed_elmo_eval.sh matres
./reproduce_proposed_elmo_eval.sh tcr
```
where the argument means the dataset: one is [MATRES](https://cogcomp.seas.upenn.edu/page/publication_view/834) and one is [TCR](https://cogcomp.seas.upenn.edu/page/publication_view/835). Both datasets are already preprocessed and saved in this repo ([here](https://github.com/qiangning/NeuralTemporalRelation-EMNLP19/tree/master/data)).

The performance is expected to be as follows:
```
DATASET=matres
TEST ACCURACY=0.6989
TEST PRECISION=0.7042
TEST RECALL=0.7981
TEST F1=0.7482
```
and
```
DATASET=tcr
TEST ACCURACY=0.7850
TEST PRECISION=0.7973
TEST RECALL=0.7850
TEST F1=0.7911
```

If you want to retrain the models, you can use the following command:
```
./reproduce_proposed_elmo_train.sh [seed] [withCSE]
```
where `seed` [int] is for the random number generator and `withCSE` [true/false] controls whether the common sense encoder (CSE) is used in the model or not. What `withCSE` does is essentially setting the common sense embedding dimension to be 1, which is a more convenient way we have found to minimize changes in code. FYI, the seed we used in evaluation was 102.

# Structure of this repo

- `data/`: 
  - tcr-temprel.xml is the preprocessed data of TCR. Since TCR is only used as an evaluation dataset, we don't split it.
  - trainset-temprel.xml and testset-temprel.xml are the preprocessed data of MATRES. MATRES is split into train and test. The proposed model(s) are all trained on the train split of MATRES.
- `figs/`, `logs/`, `models/`: saves the training/testing curves, logs, and pretrained models.
- `output/`: the confidence scores for each temporal relation prediction made by the proposed method. This is very handy if you want to run ILP inference afterward.
- `ser/`: some preprocessed serialization files, such as the word embeddings of every sentence in the datasets and the pretrained CSE encoder.
- `myLSTM.py`: the neural network structure model used here
- `exp_myLSTM.py`: the main file used to run experiments

# How to read the results
## figs
The file names should be very self-explanory. `noCSE` means that no CSE was used. `tuning` means we split the original train set into a 80% train and 20% dev, and we tune the trainig epochs on the dev set. When a best training epoch is selected, we retrain on the original train set and produce the `retrain.pdf` figures.

From these existing figures, you can see that the overfitting is still obvious: The selected "best" epochs in each experiment actually lead to suboptimal performances on the test set. When we produce numbers reported in the original paper, we used early stop and cool down so that the testing performance was higher than the numbers produced here. Feel free to use either those reported numbers or the reproduced numbers in this repo.

## logs
You can scroll down in each log file to see a summary of the performance.

## output
- Every file in `output/` is formatted as follows: doc_id, event_id1, event_id2, score for before, score for after, score for equal, score for vague
- `selected.output` and `output` are both the result from the selected epoch. They're slightly different because they're saved in different times of the experiment and the randomness gives them difference. See our code for more details.

# Comparison between using CSE and not using CSE
You can run the following script to compare using CSE and not using CSE. The following is a script that produces eight different runs.
```
./batch_exp.sh "100 101 102 103 104 105 106 107"
```
We used paired t-test, with an outlier deleted (the one corresponding to seed=107). The null hypothesis was "CSE doesn't improve", and the t-stats was 2.52 for accuracy and 1.89 for f-score. The p-values were 2.5% and 10% for accuracy and f-score, respectively, so we can conclude that CSE indeed improves the performance.


