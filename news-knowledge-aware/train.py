import json
import os
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.transforms as transforms

import datasets as data
import models
import utils as ut


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # sets device for the model and PyTorch tensors
cudnn.benchmark = True 

# Specify pointers to the data.
data_dir = "img_caption_data/input_dataset_files/"  # folder with the input data files
to_base_name = "_nytimes"
data_name = "knowledge_from_metadata" + to_base_name
pretrained_word_embeddings_file = os.path.join("data", "glove.42B.300d.txt")

# Model parameters.
emb_dim = 300  # dimension of the word embeddings
attention_dim = 512  # dimension of the attention linear layers
decoder_dim = 512  # dimension of the feedforward network in the Transformer decoder
encoder_dim = 512 # dimension of the feedforward network in the Transformer encoder
num_heads = 10 # number of heads in a Transformer layer
num_layers = 3 # number of Transformer layers
# Dropout values.
dropout_val = 0.5
dropout_dec = 0.2
dropout_enc = 0.2
dropout_pos = 0.1

# Training parameters.
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of the number of epochs since there has been an improvement in loss
max_epochs_since_improvement = 20 # trigger early stopping after this number of epochs without improvement
batch_size = 3
workers = 1  # for data loading
encoder_lr = 1e-4  # learning rate for the encoder if fine-tuning is enabled
decoder_lr = 4e-4  # learning rate for the decoder
grad_clip = 5.0  # clip gradients at an absolute value of
best_loss = 1e5  # the starting best loss value (i.e. a randomly big number)
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # whether to fine-tune the encoder
checkpoint = None # path to a checkpoint to continue the training from
zero_out_epochs_since_improvement = True # if starting from a checkpoint, start from epoch 0 or continue from the last epoch recorded


def main():
    """
    Training and validation.
    """

    global best_loss, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, zero_out_epochs_since_improvement

    # Read the word map.
    word_map_file = os.path.join(data_dir, "WORDMAP_" + data_name + ".json")
    with open(word_map_file, "r") as j:
        word_map = json.load(j)

    if checkpoint is None:
        # Initialize the decoder.
        decoder = models.DecoderTransformer(
            word_map=word_map, 
            emb_dim=emb_dim, 
            decoder_dim=decoder_dim,
            encoder_dim=encoder_dim,
            num_heads=num_heads, 
            num_layers=num_layers
        )
        # Load pretrained embeddings for the vocabulary words.
        pretrained_embeddings = ut.load_embeddings(pretrained_word_embeddings_file, word_map)
        decoder.load_pretrained_embeddings(pretrained_embeddings)
        decoder.fine_tune_embeddings(True)
        #
        # Create the decoder optimizer.
        decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, decoder.parameters()),
            lr=decoder_lr,
        )
        # Initialize the encoder.
        encoder = models.Encoder(emb_dim=emb_dim)
        encoder.fine_tune(fine_tune_encoder)
        # Create the encoder optimizer if fine-tuning is enabled.
        encoder_optimizer = (
            torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr,
            )
            if fine_tune_encoder
            else None
        )

    else:
        # Load the pre-computed checkpoint.
        print("LOADING CHECKPOINT...")
        checkpoint = torch.load(checkpoint)
        decoder = checkpoint["decoder"]
        encoder = checkpoint["encoder"]
        if zero_out_epochs_since_improvement:
            start_epoch = 0
            epochs_since_improvement = 0
            best_loss = 1e5
            decoder_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, decoder.parameters()),
                lr=decoder_lr,
            )
            encoder_optimizer = (
                torch.optim.Adam(
                    params=filter(lambda p: p.requires_grad, encoder.parameters()),
                    lr=encoder_lr,
                )
                if fine_tune_encoder
                else None
            )
        else:
            start_epoch = checkpoint["epoch"] + 1
            epochs_since_improvement = checkpoint["epochs_since_improvement"]
            best_loss = checkpoint["loss"]
            decoder_optimizer = checkpoint["decoder_optimizer"]
            encoder_optimizer = checkpoint["encoder_optimizer"]
                        
    # Move to GPU, if possible.
    decoder.to(device)
    encoder.to(device)

    # Define the loss function.
    criterion_crossentropy = nn.CrossEntropyLoss(ignore_index=word_map["<pad>"]).to(device)

    # Get the dataloaders.
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_loader = torch.utils.data.DataLoader(
        data.CaptionDataset(
            data_dir,
            data_name,
            "TRAIN",
            transform=transforms.Compose([normalize]),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        data.CaptionDataset(
            data_dir,
            data_name,
            "VAL",
            transform=transforms.Compose([normalize]),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )

    # Iterate over epochs.
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after max_epochs_since_improvement.
        if epochs_since_improvement == max_epochs_since_improvement:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            ut.adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                ut.adjust_learning_rate(encoder_optimizer, 0.8)

        # Train for one epoch.
        train(
            train_loader=train_loader,
            encoder=encoder,
            decoder=decoder,
            criterion_crossentropy=criterion_crossentropy,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch,
        )

        # Validate the epoch.
        last_loss = validate(
            val_loader=val_loader,
            encoder=encoder,
            decoder=decoder,
            criterion_crossentropy=criterion_crossentropy,
        )

        # Check if there was an improvement in loss.
        is_best = last_loss < best_loss
        best_loss = min(last_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print(
                "\nEpochs since last improvement: %d\n"
                % (epochs_since_improvement,)
            )
        else:
            epochs_since_improvement = 0

        # Save the checkpoint.
        ut.save_checkpoint(
            data_name,
            epoch,
            epochs_since_improvement,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            last_loss,
            is_best,
        )


def train(
    train_loader,
    encoder,
    decoder,
    criterion_crossentropy,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
):
    """
    Performs one epoch of training.

    :param train_loader: DataLoader for the training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion_crossentropy: loss layer
    :param encoder_optimizer: optimizer to update the encoder's weights (if fine-tuning is enabled)
    :param decoder_optimizer: optimizer to update the decoder's weights
    :param epoch: epoch number
    """
    # Set the train mode (dropout and batchnorm are used).
    decoder.train() 
    encoder.train()
    #
    batch_time = ut.AverageMeter()  # forward prop. + back prop. time
    data_time = ut.AverageMeter()  # data loading time
    losses = ut.AverageMeter()  # overall loss (per word decoded)
    #
    start = time.time()
    # Iterate over the data batches.
    for i, (
        imgs,
        captions,
        caption_lengths,
        caption_masks,
        entity_features,
        _,
        facts,
        _,
    ) in enumerate(train_loader):
        data_time.update(time.time() - start)
        # Move to GPU, if possible.
        imgs = imgs.to(device)
        captions = captions.to(device)
        caption_lengths = caption_lengths.to(device)
        caption_masks = caption_masks.to(device)
        facts = facts.to(device)
        #
        # Encode and decode.
        imgs = encoder(imgs)
        scores, captions_sorted, decode_lengths = decoder(
            captions, imgs, caption_masks, caption_lengths, entity_features, facts
        )
        #
        # Remove the first <start> token.
        targets = captions_sorted[:, 1:]
        # Remove <end> and <pad> tokens.
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        #
        # Calculate the loss.
        loss = criterion_crossentropy(scores, targets)
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()
        # Clip the gradients.
        if grad_clip is not None:
            ut.clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                ut.clip_gradient(encoder_optimizer, grad_clip)
        # Update the trainable parameter weights.
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()
        #
        # Keep track of the loss.
        losses.update(loss.item(), sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()
        if i % print_freq == 0:
            if True:
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                        epoch,
                        i,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                    )
                )


def validate(val_loader, encoder, decoder, criterion_crossentropy):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion_crossentropy: loss layer
    :return: loss average
    """
    # Set the evaluation mode (no dropout or batchnorm).
    decoder.eval()
    if encoder is not None:
        encoder.eval()
    #
    batch_time = ut.AverageMeter()
    losses = ut.AverageMeter()
    #
    start = time.time()
    # Disable gradient calculation to avoid a memory error.
    with torch.no_grad():
        # Iterate over the data batches.
        for i, (
            imgs,
            captions,
            caption_lengths,
            caption_masks,
            entity_features,
            _,
            facts,
            _,
        ) in enumerate(val_loader):
            # Move to GPU, if possible.
            imgs = imgs.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)
            caption_masks = caption_masks.to(device)
            facts = facts.to(device)
            #
            # Encode and decode.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, captions_sorted, decode_lengths = decoder(
                captions, imgs, caption_masks, caption_lengths, entity_features, facts
            )
            #
            # Remove the first <start> token.
            targets = captions_sorted[:, 1:]
            # Remove <end> and <pad> tokens.
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            #
            # Calculate the loss.
            loss = criterion_crossentropy(scores, targets)
            #
            # Keep track of the loss.
            losses.update(loss.item(), sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()
            if i % print_freq == 0:
                if True:
                    print(
                        "Validation: [{0}/{1}]\t"
                        "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                            i,
                            len(val_loader),
                            batch_time=batch_time,
                            loss=losses,
                        )
                    )

    return losses.avg


if __name__ == "__main__":
    main()
