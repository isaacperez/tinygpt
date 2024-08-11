import sys
import time

from tinygpt.tensor import Tensor 
from tinygpt.nn import GPT
from tinygpt.tokenizer import BPETokenizer, RegexPatterns
from tinygpt.dataset import TextDataset, DatasetHandler
from tinygpt.losses import CrossEntropyLoss
from tinygpt.optimizers import Adam
from tinygpt.utils import tree_flatten


# Config
vocab_size = 1024
max_seq_length = 6
batch_size = 16
num_epochs = 2

inference_mode = False
inference_sentence = "First Citizen:\n"
sampling_temperature = 0.25

load_checkpoint = True
model_weights_filename = "gpt_model.json"

data_path = "../data/shakespeare/input.txt"
train_path = "../data/shakespeare/train.txt"
val_path = "../data/shakespeare/val.txt"
tokenizer_path = "./tokenizer.model"  # None if you want to train the tokenizer


"""
Training lasted about 3 days. It was stopped and resumed several times.
Read training_log.txt if you want to see the training log of the model.
"""


def to_one_hot(target_ids: Tensor) -> Tensor:
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


def validation(val_dataset_handler: DatasetHandler, gpt: GPT) -> None:
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


def inference(gpt: GPT) -> None:
    input_ids = tokenizer.encode(inference_sentence, allowed_special="all")

    print("Greedy")
    output_ids = gpt.generate_greedy(token_ids=Tensor(input_ids).reshape((1, len(input_ids))), max_new_tokens=25)
    print(f"Model output: {repr(tokenizer.decode(output_ids.to_python()[0]))}")

    print("Sample with temperature")
    for i in range(5):
        output_ids = gpt.generate_sample_with_temperature(
            token_ids=Tensor(input_ids).reshape((1, len(input_ids))), 
            max_new_tokens=25, 
            temperature=sampling_temperature
        )
        print(f"Model output: {repr(tokenizer.decode(output_ids.to_python()[0]))}")


def calculate_num_parameters(gpt: GPT) -> None:
    print("Parameters of the model:")
    num_parameters = 0
    for param_name, param in tree_flatten(gpt.trainable_parameters()):
        # Only trainable parameters
        if param.requires_grad:
            num_elements = 1
            for dim in param.shape:
                num_elements *= dim
            num_parameters += num_elements
            
            print("\t", param_name, num_elements)
        
    print("\nTotal parameters:", num_parameters)


if __name__ == "__main__":

    # You may need to increase the recursion limit depending of the size of the model
    print("Current recursion limit:", sys.getrecursionlimit())
    sys.setrecursionlimit(10000)
    print("New recursion limit:", sys.getrecursionlimit())

    # Create the model
    print("Creating the model")
    gpt = GPT(max_seq_length=max_seq_length, vocab_size=vocab_size, num_layers=6, num_heads=4, embedding_dim=16)

    if load_checkpoint or inference_mode:
        gpt.load_weights(model_weights_filename)

    # Get the parameters of the model
    calculate_num_parameters(gpt)
  
    # Create a tokenizer
    tokenizer = BPETokenizer(regex_pattern=RegexPatterns.GPT4)

    # Train the tokenizer or load from disk
    with open(data_path, encoding="utf-8") as file:
            text_corpus = file.read()

    if tokenizer_path is None and not inference_mode:
        print("Training the tokenizer...")
        tokenizer.train(text_corpus=text_corpus, vocab_size=vocab_size, verbose=True)
        tokenizer.save("tokenizer")

    else:
        tokenizer.load(tokenizer_path)

    print("Tokenizer demo:")
    tokenizer._visualise_tokens(tokenizer.encode(text_corpus[:300], allowed_special="all"))

    # Do inference only
    if inference_mode:
        inference(gpt)
        exit()

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
    sgd = Adam(module=gpt, learning_rate=0.0003)

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
            if it % 32 == 0:
                input_ids = tokenizer.encode("First Citizen:\n", allowed_special="all")
                for i in range(3):
                    output_ids = gpt.generate_sample_with_temperature(
                        token_ids=Tensor(input_ids).reshape((1, len(input_ids))), 
                        max_new_tokens=25, 
                        temperature=sampling_temperature
                    )
                    print(f"Model output: {repr(tokenizer.decode(output_ids.to_python()[0]))}")

                # Save the model
                print("Saving the weights...")
                gpt.save_weights(model_weights_filename)
                
        validation(val_dataset_handler, gpt)

        # Save the model 
        gpt.save_weights(model_weights_filename)