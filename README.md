# NLP Toolkit
Original repository found at: https://github.com/plkmo/NLP_Toolkit

Library containing state-of-the-art models for Natural Language Processing tasks  
The purpose of this toolkit is to allow for **easy training/inference of state-of-the-art models**, for various NLP tasks.

Here, we only use the punctuation restoration model implemented, which is based on [Investigating LSTM for punctuation prediction](https://ieeexplore.ieee.org/document/7918492). We made some small modifications to the code, mostly for the data loading as the original repository only accepted TED formatted datasets.

## Train

To learn punctuation (comma, period, exclamation mark and question mark), we trained on the unaligned english corpus.

```sh
$ python train_punc.py
```

## Infer from file

We used the trained model to infer punctuation on the 11k aligned english examples.

```sh
$ python infer_punc.py
```