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



MNIST_URL = "http://yann.lecun.com/exdb/mnist/"
MNIST_FILES = [
    ("training_images", "train-images-idx3-ubyte.gz"),
    ("test_images", "t10k-images-idx3-ubyte.gz"),
    ("training_labels", "train-labels-idx1-ubyte.gz"),
    ("test_labels", "t10k-labels-idx1-ubyte.gz"),
]


def download_file(url, save_path):
    print(f"Downloading file {os.path.basename(save_path)}...")
    request.urlretrieve(url, save_path)


def extract_images(file_path):
    with gzip.open(file_path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack('>4I', f.read(16))
        assert magic == 2051, "Invalid magic number for image file"
        assert num_images in [10000, 60000], "Invalid number of images"
        assert rows == 28 and cols == 28, "Invalid image dimensions"

        images = [
            [float(pixel) / 255.0 for pixel in struct.unpack(f'>{rows*cols}B', f.read(rows*cols))]
            for _ in range(num_images)
        ]
        return images


def extract_labels(file_path):
    with gzip.open(file_path, "rb") as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        assert magic == 2049, "Invalid magic number for label file"
        assert num_labels in [10000, 60000], "Invalid number of labels"

        labels = [float(label) for label in struct.unpack(f'>{num_labels}b', f.read(num_labels))]
        return [[label] for label in labels]


def download_mnist_dataset(save_dir='/tmp', base_url=MNIST_URL, filename="mnist.pkl"):
    dataset_path = os.path.join(save_dir, filename)
    if os.path.exists(dataset_path):
        print("MNIST dataset already downloaded.")
    else:
        mnist_data = {}
        for name, file in MNIST_FILES:
            file_path = os.path.join(save_dir, file)
            download_file(base_url + file, file_path)
            if "images" in name:
                mnist_data[name] = extract_images(file_path)
            else:
                mnist_data[name] = extract_labels(file_path)

        with open(dataset_path, 'wb') as f:
            pickle.dump(mnist_data, f)
        print("MNIST dataset downloaded and saved.")

    print("Loading MNIST dataset...")
    with open(dataset_path, 'rb') as f:
        mnist = pickle.load(f)

    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def to_one_hot(labels, num_classes=10):
    return [[1.0 if i == label[0] else 0.0 for i in range(num_classes)] for label in labels]


def calculate_accuracy(predictions, labels, reduce=True):
    correct_predictions = ((predictions * labels).sum(axes=(1,)) == predictions.max(axes=(1,))).to_python()
    if reduce:
        return sum(correct_predictions) / predictions.shape[0]
    else:
        return correct_predictions


class MNISTModel(Module):
    def __init__(self):
        super().__init__()

        self.mlp = MLP(input_dims=784, hidden_dims=[32], activation_fn='relu', bias=True)
        self.classification_layer = FullyConnectedLayer(input_dims=32, output_dims=10, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        return self.classification_layer(self.mlp(x))


if __name__ == "__main__":
    # Load the dataset
    train_img, train_labels, test_img, test_labels = download_mnist_dataset()

    # Create the model
    model = MNISTModel()

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
        # Training
        for it, (images, labels) in enumerate(zip(batched(train_img, batch_size), batched(train_labels, batch_size))):
            # Conver the data into tensors
            images, labels = Tensor(images), Tensor(to_one_hot(labels, 10))

            # Do a forward pass
            tic = time.perf_counter()
            sgd.zero_grad()
            model_output = model(images)

            # Calculate the loss and the accuracy
            accuracy = calculate_accuracy(predictions=model_output, labels=labels)
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

        # Test
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
            accuracy = calculate_accuracy(predictions=model_output, labels=labels, reduce=False)
            loss = loss_fn(logits=model_output, labels=labels).to_python()

            # Save the results
            losses.extend(loss)
            accuracies.extend(accuracy)

            print(
                f"[TEST][Epoch {epoch + 1} of {num_epochs}][it. {it + 1} of {num_test_iterations}]"
                f" mean it. loss = {sum(loss) / len(loss):.4f}"
                f" | mean it. accuracy = {sum(accuracy) / len(accuracy):.4f}"
                f" | Time {toc - tic:.3f} (s)"
            )

        print(
            f"[TEST][Epoch {epoch + 1} of {num_epochs}]"
            f" mean epoch loss = {sum(losses) / len(losses):.4f}"
            f" | accuracy = {sum(accuracies) / len(accuracies):.4f}"
        )

    """
    This is what I got after one epoch: 
    [TEST][Epoch 1 of 5] mean epoch loss = 0.2192 | accuracy = 0.9309
    """
