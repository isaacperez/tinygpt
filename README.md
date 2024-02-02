# TinyGPT
Like [NanoGPT](https://github.com/karpathy/nanoGPT) but tiny. It is also inspired by [Tinygrad](https://github.com/tinygrad/tinygrad), [Pytorch](https://github.com/pytorch/pytorch) and [MLX](https://github.com/ml-explore/mlx).

The main objective of this project is to be as didactic as possible, avoiding optimizations that make the code difficult to understand.

I hope we can understand how to train and run a model like GPT-3 using as few libraries as possible and programming everything from scratch, including the library to train and run the model.

## Installation
The current recommended way to install TinyGPT is from source.

### From source
```bash
$ git clone https://github.com/isaacperez/tinyGPT.git
$ cd tinygpt
$ python -m pip install -e .
```
Don't forget the `.` at the end!

## Usage
[TO DO]


## Documentation
Documentation along with a quick start guide can be found in the docs/ directory.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

### Testing
You need to install pytest:
```bash
$ python -m pip install pytest
```
and [TinyGPT](#installation), then run: 
```bash
$ pytest
```