# Tiny shakespeare
[Tiny shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset.

To download and prepare the dataset run
```bash
$ python prepare.py
```

It will download the dataset into `input.txt` file and it will create three files:  
 - `metadata.json`: general information about the dataset.
 - `train.txt`: training data.
 - `val.txt`: validation data.