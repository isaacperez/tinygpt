import os
import gzip
import time
import pickle
import struct
from urllib import request
from itertools import batched

from tinygpt.tensor import Tensor 
from tinygpt.module import Module
from tinygpt.optimizers import SGD
from tinygpt.nn import MLP, FullyConnectedLayer
from tinygpt.losses import CrossEntropyLoss


def download_dataset(
        save_dir: str = '/tmp', 
        base_url: str = "http://yann.lecun.com/exdb/mnist/", 
        filename: str = "mnist.pkl"
    ):

    def download_and_save(save_file):
        filename = [
            ["training_images", "train-images-idx3-ubyte.gz"],
            ["test_images", "t10k-images-idx3-ubyte.gz"],
            ["training_labels", "train-labels-idx1-ubyte.gz"],
            ["test_labels", "t10k-labels-idx1-ubyte.gz"],
        ]

        mnist = {}
        # Download the files
        for name in filename:
            out_file = os.path.join(save_dir, name[1])
            print(f"Downloading file {name[1]}...")
            request.urlretrieve(base_url + name[1], out_file)

        # Read the images
        print("Reading the images...")
        for name in filename[:2]:
            out_file = os.path.join(save_dir, name[1])
            with gzip.open(out_file, "rb") as f:
                magic, num_images, rows, cols = struct.unpack('>4I', f.read(16))
                
                # Should get values specified here http://yann.lecun.com/exdb/mnist/
                assert magic == 2051
                assert num_images == 10000 or num_images == 60000
                assert rows == 28
                assert cols == 28
                
                num_pixels = rows * cols
                mnist[name[0]] = [
                    list(map(lambda x: float(x) / 255.0, struct.unpack('>{}B'.format(num_pixels), f.read(num_pixels))))
                    for _ in range(num_images)
                ]
            
                # Verify data looks appropriate
                for img in mnist[name[0]]:
                    for p in img:
                        try:
                            assert 0. <= p <= 1.
                        except AssertionError:
                            print("{} is not a valid pixel value".format(p))
                            raise
        
        # Read the labels
        print("Reading the labels")
        for name in filename[-2:]:
            out_file = os.path.join(save_dir, name[1])
            with gzip.open(out_file, "rb") as f:
                magic, num_labels = struct.unpack('>II', f.read(8))

                # Should get values specified here http://yann.lecun.com/exdb/mnist/
                assert magic == 2049
                assert num_labels == 10000 or num_labels == 60000

                mnist[name[0]] = [[f] for f in map(float, struct.unpack('>{}b'.format(num_labels), f.read(num_labels)))]

        # Save the data for next time
        print("Saving the data..")
        with open(save_file, "wb") as f:
            pickle.dump(mnist, f)

    # Check if the data is already downloaded
    save_file = os.path.join(save_dir, filename)
    if not os.path.exists(save_file):
        download_and_save(save_file)

    # Read the data
    print("Loading the data")
    with open(save_file, "rb") as f:
        mnist = pickle.load(f)

    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def to_one_hot(labels: list[int], num_classes: int):
    one_hot_encoding = []
    for label in labels:
        one_hot_encoding.append([1.0 if label[0] == idx else 0.0 for idx in range(num_classes)])

    return one_hot_encoding


class Model(Module):
    def __init__(self):
        super().__init__()

        self.mlp = MLP(input_dims=784, hidden_dims=[32], activation_fn='relu', bias=True)
        self.classification_layer = FullyConnectedLayer(input_dims=32, output_dims=10, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        return self.classification_layer(self.mlp(x))


def get_accuracy(model_output, labels, reduce=True):
    max_output = model_output.max(axes=(1,))
    accuracy = ((model_output * labels).sum(axes=(1,)) == max_output).to_python()
    if reduce:
        accuracy = sum(accuracy) / max_output.shape[0]

    return accuracy


if __name__ == "__main__":
    # Load the dataset
    train_img, train_labels, test_img, test_labels = download_dataset()

    # Create the model
    model = Model()

    # Create the optimizer
    sgd = SGD(model, learning_rate=0.1, momentum=0.9)

    # Create the loss function
    loss_fn = CrossEntropyLoss()

    # Train the model
    num_epochs = 5
    batch_size = 256
    num_train_iterations = int(len(train_img) / batch_size + 1)
    num_test_iterations = int(len(test_img) / batch_size + 1)
    for epoch in range(num_epochs):
        for it, (images, labels) in enumerate(zip(batched(train_img, batch_size), batched(train_labels, batch_size))):
            # Conver the data into tensors
            images, labels = Tensor(images), Tensor(to_one_hot(labels, 10))

            # Do a forward pass
            tic = time.perf_counter()
            sgd.zero_grad()
            model_output = model(images)

            # Calculate the loss and the accuracy
            accuracy = get_accuracy(model_output=model_output, labels=labels)
            loss = loss_fn(logits=model_output, labels=labels)
            mean_loss = loss.sum(axes=(0,)) / float(loss.shape[0])
            
            # Do the backward pass
            mean_loss.backward()
            sgd.update()
            toc = time.perf_counter()

            print(
                f"[TRAIN][Epoch {epoch + 1} of {num_epochs}][it. {it + 1} of {num_train_iterations}]"
                f" mean it. loss = {mean_loss.to_python():.4f}"
                f" | mean it. accuracy = {accuracy:.4f}"
                f" | Time {toc - tic:.3f} (s)"
            )

        # Do one validation on test set
        accuracies = []
        losses = []
        for it, (images, labels) in enumerate(zip(batched(test_img, batch_size), batched(test_labels, batch_size))):
            # Conver the data into tensors
            images, labels = Tensor(images), Tensor(to_one_hot(labels, 10))

            # Do a forward pass
            tic = time.perf_counter()
            model_output = model(images)
            toc = time.perf_counter()

            # Calculate the loss and the accuracy
            accuracy = get_accuracy(model_output=model_output, labels=labels, reduce=False)
            loss = loss_fn(logits=model_output, labels=labels).to_python()

            # Save the results
            losses.extend(loss)
            accuracies.extend(accuracy)

            print(
                f"[VAL][Epoch {epoch + 1} of {num_epochs}][it. {it + 1} of {num_test_iterations}]"
                f" mean it. loss = {sum(loss) / len(loss):.4f}"
                f" | mean it. accuracy = {sum(accuracy) / len(accuracy):.4f}"
                f" | Time {toc - tic:.3f} (s)"
            )

        print(
            f"[VAL][Epoch {epoch + 1} of {num_epochs}]"
            f" mean epoch loss = {sum(losses) / len(losses):.4f}"
            f" | accuracy = {sum(accuracies) / len(accuracies):.4f}"
        )

    """
    This is what I got after one epoch: 
    [VAL][Epoch 1 of 5] mean epoch loss = 0.2192 | accuracy = 0.9309
    """
