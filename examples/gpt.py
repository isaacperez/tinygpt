import sys
import time

from tinygpt.tensor import Tensor 
from tinygpt.nn import GPT
from tinygpt.tokenizer import BPETokenizer, RegexPatterns
from tinygpt.dataset import TextDataset, DatasetHandler
from tinygpt.losses import CrossEntropyLoss
from tinygpt.optimizers import SGD


# Config
vocab_size = 1024
max_seq_length = 16
batch_size = 16
num_epochs = 2

load_checkpoint = False
model_weights_filename = "gpt_model.json"

data_path = "../data/shakespeare/input.txt"
train_path = "../data/shakespeare/train.txt"
val_path = "../data/shakespeare/val.txt"
tokenizer_path = "./tokenizer.model"  # None if you want to train the tokenizer


def to_one_hot(target_ids: Tensor):
    # taget_ids is a one-dimensional vector
    num_elements = batch_size * max_seq_length
    target_ids = target_ids.reshape((num_elements,))

    # Each element of the one-dimensional vector will be a one-hot vector
    target_ids_one_hot = [[0.0 for _ in range(vocab_size)] for _ in range(num_elements)]
    for idx_target, target_id in enumerate(target_ids):
        target_ids_one_hot[idx_target][target_id.to_python()] = 1.0

    # We need a Tensor to compare the output of the model
    target_ids_one_hot = Tensor(target_ids_one_hot)

    return target_ids_one_hot


def validation(val_dataset_handler, gpt):
    losses = []
    for it, (input_ids, target_ids) in enumerate(val_dataset_handler):
        # Convert the data into Tensors
            input_ids = Tensor(input_ids)
            target_ids = Tensor(target_ids)

            # Do inference with the model
            start = time.time()
            output_ids = gpt(input_ids)
            end = time.time()
            forward_time = end - start

            # Prepare the target_ids as a tensor of one-hot vectors with two dimensions
            target_ids_one_hot = to_one_hot(target_ids)

            # Prepare output as a tensor with two dimensions
            output_ids = output_ids.reshape((batch_size * max_seq_length, vocab_size))        

            # Calculate the loss
            loss = loss_fn(logits=output_ids, labels=target_ids_one_hot)
            loss = loss.mean(axes=(0,)).to_python()
            losses.append(loss)

            print(
                f"[VAL][It. {it + 1:>5d}/{len(val_dataset_handler)}] Loss {loss:01.4f}"
                f" | forward: {forward_time:01.2f} sec."
            )
    
    mean_loss = sum(losses) / len(losses)
    print(f"Mean loss: {mean_loss:.2f}")


if __name__ == "__main__":

    # You may need to increase the recursion limit depending of the size of the model
    print("Current recursion limit:", sys.getrecursionlimit())
    sys.setrecursionlimit(10000)
    print("New recursion limit:", sys.getrecursionlimit())

    # Create the model
    print("Creating the model")
    gpt = GPT(max_seq_length=max_seq_length, vocab_size=vocab_size, num_layers=6, num_heads=2, embedding_dim=16)

    if load_checkpoint:
        gpt.load_weights(model_weights_filename)

    # Create a tokenizer
    tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)

    # Train the tokenizer or load from disk
    with open(data_path, encoding="utf-8") as file:
            text_corpus = file.read()

    if tokenizer_path is None:
        print("Training the tokenizer...")
        tokenizer.train(text_corpus=text_corpus, vocab_size=vocab_size, verbose=True)
        tokenizer.save("tokenizer")

    else:
        tokenizer.load(tokenizer_path)

    print("Tokenizer demo:")
    tokenizer._visualise_tokens(tokenizer.encode(text_corpus[:800], allowed_special="all"))

    # Create the datasets and the handlers
    print("Creating datasets...")
    train_dataset = TextDataset(data_file_path=train_path, tokenizer=tokenizer, max_seq_length=max_seq_length)
    train_dataset_handler = DatasetHandler(dataset=train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    val_dataset = TextDataset(data_file_path=val_path, tokenizer=tokenizer, max_seq_length=max_seq_length)
    val_dataset_handler = DatasetHandler(dataset=val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)

    print(f"Train dataset with {len(train_dataset)} sequences")
    print(f"Val dataset with {len(val_dataset)} sequences")

    # Create a loss function
    loss_fn = CrossEntropyLoss()

    # Create the optimizer
    sgd = SGD(module=gpt, learning_rate=0.00001, momentum=0.8, weight_decay=0.0)

    # Training loop
    print("Begining training...")
    for epoch in range(num_epochs):
        for it, (input_ids, target_ids) in enumerate(train_dataset_handler):
            # Convert the data into Tensors
            input_ids = Tensor(input_ids)
            target_ids = Tensor(target_ids)

            # Clean the gradients of previous iterations 
            sgd.zero_grad()

            # Do inference with the model
            start = time.time()
            output_ids = gpt(input_ids)
            end = time.time()
            forward_time = end - start

            # Prepare the target_ids as a tensor of one-hot vectors with two dimensions
            target_ids_one_hot = to_one_hot(target_ids)

            # Prepare output as a tensor with two dimensions
            output_ids = output_ids.reshape((batch_size * max_seq_length, vocab_size))        

            # Calculate the loss
            loss = loss_fn(logits=output_ids, labels=target_ids_one_hot)
            loss = loss.mean(axes=(0,))
            
            # Do the backward pass
            start = time.time()
            loss.backward()
            sgd.update() 
            end = time.time()
            backward_time = end - start

            print(
                f"[Epoch {epoch + 1:>3d}/{num_epochs}][It. {it + 1:>5d}/{len(train_dataset_handler)}]"
                f" Loss {loss.to_python():01.4f}"
                f" | forward: {forward_time:01.2f} sec. | backward {backward_time:01.2f} sec."
            )
            
            # Generate some outputs from time to time
            if it % 16 == 0:
                input_ids = tokenizer.encode("First Citizen:\n", allowed_special="all")
                output_ids = gpt.generate(token_ids=Tensor(input_ids).reshape((1, len(input_ids))), max_new_tokens=25)
                print(f"Model output: {repr(tokenizer.decode(output_ids.to_python()[0]))}")

                # Save the model
                print("Saving the weights...")
                gpt.save_weights(model_weights_filename)
                
        validation(val_dataset_handler, gpt)

        # Save the model 
        gpt.save_weights(model_weights_filename)