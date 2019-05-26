# This code defines the NN architecture for captioning
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import os
from PIL import Image
from cache import cache
import json
from copy import copy

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from helpers import load_json
from helpers import load_image
from helpers import print_progress
from captions_preprocess import TokenizerWrap
from captions_preprocess import flatten
from captions_preprocess import mark_captions

# ONLY IMPORT DESIRED CNN MODEL
#from tensorflow.python.keras.applications import VGG16
#image_model = VGG16(include_top=True, weights='imagenet')
#transfer_layer=image_model.get_layer('fc2')

#from tensorflow.python.keras.applications import VGG19
#image_model = VGG19(include_top=True, weights='imagenet')
#transfer_layer=image_model.get_layer('fc2')

#from tensorflow.python.keras.applications import ResNet50
#image_model = ResNet50(include_top=True, weights='imagenet')
#transfer_layer=image_model.get_layer('avg_pool')

from tensorflow.python.keras.applications import InceptionV3
image_model = InceptionV3(include_top=True, weights='imagenet')
transfer_layer=image_model.get_layer('avg_pool')

# LOAD THE CORRECT TRANSFER VALUES
# SPECIFY THE CNN, resnet50 or vgg
transfer_values_train=np.load('image_features/transfer_values/InceptionV3/transfer_values_train.npy')
transfer_values_val=np.load('image_features/transfer_values/InceptionV3/transfer_values_val.npy')
transfer_values_test=np.load('image_features/transfer_values/InceptionV3/transfer_values_test.npy')

image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)
img_size=K.int_shape(image_model.input)[1:3]

# This function return a list of random token sequences
# with the given indices in the training set
def get_random_caption_tokens(idx):
    """
    Given a list of indices for images in the training-set,
    select a token-sequence for a random caption,
    and return a list of all these token-sequences.
    """
    
    # Initialize an empty list for the results.
    result = []

    # For each of the indices.
    for i in idx:
        # The index i points to an image in the training-set.
        # Each image in the training-set has at least 5 captions
        # which have been converted to tokens in tokens_train.
        # We want to select one of these token-sequences at random.

        # Get a random index for a token-sequence.
        j = np.random.choice(len(tokens_train[i]))

        # Get the j'th token-sequence for image i.
        tokens = tokens_train[i][j]

        # Add this token-sequence to the list of results.
        result.append(tokens)

    return result

# This function generates random batches of training data of the given size
num_images_train=8734
def batch_generator(batch_size):
    """
    Generator function for creating random batches of training-data.
    
    Note that it selects the data completely randomly for each
    batch, corresponding to sampling of the training-set with
    replacement. This means it is possible to sample the same
    data multiple times within a single epoch - and it is also
    possible that some data is not sampled at all within an epoch.
    However, all the data should be unique within a single batch.
    """

    # Infinite loop.
    while True:
        # Get a list of random indices for images in the training-set.
        idx = np.random.randint(num_images_train,
                                size=batch_size)
        
        # Get the pre-computed transfer-values for those images.
        # These are the outputs of the pre-trained image-model.
        transfer_values = transfer_values_train[idx]

        # For each of the randomly chosen images there are
        # at least 5 captions describing the contents of the image.
        # Select one of those captions at random and get the
        # associated sequence of integer-tokens.
        tokens = get_random_caption_tokens(idx)

        # Count the number of tokens in all these token-sequences.
        num_tokens = [len(t) for t in tokens]
        
        # Max number of tokens.
        max_tokens = np.max(num_tokens)
        
        # Pad all the other token-sequences with zeros
        # so they all have the same length and can be
        # input to the neural network as a numpy array.
        tokens_padded = pad_sequences(tokens,
                                      maxlen=max_tokens,
                                      padding='post',
                                      truncating='post')
        
        # Further prepare the token-sequences.
        # The decoder-part of the neural network
        # will try to map the token-sequences to
        # themselves shifted one time-step.
        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]

        # Dict for the input-data. Because we have
        # several inputs, we use a named dict to
        # ensure that the data is assigned correctly.
        x_data = \
        {
            'decoder_input': decoder_input_data,
            'transfer_values_input': transfer_values
        }

        # Dict for the output-data.
        y_data = \
        {
            'decoder_output': decoder_output_data
        }
        
        yield (x_data, y_data)
        
def batch_generator_val(batch_size):
    """
    Generator function for creating random batches of training-data.
    
    Note that it selects the data completely randomly for each
    batch, corresponding to sampling of the training-set with
    replacement. This means it is possible to sample the same
    data multiple times within a single epoch - and it is also
    possible that some data is not sampled at all within an epoch.
    However, all the data should be unique within a single batch.
    """

    # Infinite loop.
    while True:
        # Get a list of random indices for images in the training-set.
        idx = np.random.randint(1093,
                                size=batch_size)
        
        # Get the pre-computed transfer-values for those images.
        # These are the outputs of the pre-trained image-model.
        transfer_values = transfer_values_val[idx]

        # For each of the randomly chosen images there are
        # at least 5 captions describing the contents of the image.
        # Select one of those captions at random and get the
        # associated sequence of integer-tokens.
        tokens = get_random_caption_tokens(idx)

        # Count the number of tokens in all these token-sequences.
        num_tokens = [len(t) for t in tokens]
        
        # Max number of tokens.
        max_tokens = np.max(num_tokens)
        
        # Pad all the other token-sequences with zeros
        # so they all have the same length and can be
        # input to the neural network as a numpy array.
        tokens_padded = pad_sequences(tokens,
                                      maxlen=max_tokens,
                                      padding='post',
                                      truncating='post')
        
        # Further prepare the token-sequences.
        # The decoder-part of the neural network
        # will try to map the token-sequences to
        # themselves shifted one time-step.
        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]

        # Dict for the input-data. Because we have
        # several inputs, we use a named dict to
        # ensure that the data is assigned correctly.
        x_data = \
        {
            'decoder_input': decoder_input_data,
            'transfer_values_input': transfer_values
        }

        # Dict for the output-data.
        y_data = \
        {
            'decoder_output': decoder_output_data
        }
        
        yield (x_data, y_data)

# This connects all the layers of the decoder to some input of transfer-values
def connect_decoder(transfer_values):
    # Map the transfer-values so the dimensionality matches
    # the internal state of the GRU layers. This means
    # we can use the mapped transfer-values as the initial state
    # of the GRU layers.
    initial_state = decoder_transfer_map(transfer_values)

    # Start the decoder-network with its input-layer.
    net = decoder_input
    
    # Connect the embedding-layer.
    net = decoder_embedding(net)
    
    # Connect all the GRU layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)
    net = decoder_gru4(net, initial_state=initial_state)
    net = decoder_gru5(net, initial_state=initial_state)

    # Connect the final dense layer that converts to
    # one-hot encoded arrays.
    decoder_output = decoder_dense(net)
    
    return decoder_output

# for the cost-function calculation
def sparse_cross_entropy(y_true, y_pred):
    """
    Calculate the cross-entropy loss between y_true and y_pred.
    
    y_true is a 2-rank tensor with the desired output.
    The shape is [batch_size, sequence_length] and it
    contains sequences of integer-tokens.

    y_pred is the decoder's output which is a 3-rank tensor
    with shape [batch_size, sequence_length, num_words]
    so that for each sequence in the batch there is a one-hot
    encoded array of length num_words.
    """

    # Calculate the loss. This outputs a
    # 2-rank tensor of shape [batch_size, sequence_length]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire 2-rank tensor, we reduce it
    # to a single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean


# recreate the tokenizer
mark_start='ssss '
mark_end=' eeee'
captions_train=load_json('captions_train')
captions_train_marked=mark_captions(captions_train)
captions_train_flat=flatten(captions_train_marked)
tokenizer=TokenizerWrap(texts=captions_train_flat,
                        num_words=2000)
token_start=tokenizer.word_index[mark_start.strip()]
token_end=tokenizer.word_index[mark_end.strip()]
tokens_train=tokenizer.captions_to_tokens(captions_train_marked)

filenames_val=load_json('filenames_val')


def generate_caption(image_path, max_tokens=30):
    """
    Generate a caption for the image in the given path.
    The caption is limited to the given number of tokens (words).
    """

    # Load and resize the image.
    image = load_image(image_path, size=img_size)
    
    # Expand the 3-dim numpy array to 4-dim
    # because the image-model expects a whole batch as input,
    # so we give it a batch with just one image.
    image_batch = np.expand_dims(image, axis=0)

    # Process the image with the pre-trained image-model
    # to get the transfer-values.
    transfer_values = image_model_transfer.predict(image_batch)

    # Pre-allocate the 2-dim array used as input to the decoder.
    # This holds just a single sequence of integer-tokens,
    # but the decoder-model expects a batch of sequences.
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    # The first input-token is the special start-token for 'ssss '.
    token_int = token_start

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0

    # While we haven't sampled the special end-token for ' eeee'
    # and we haven't processed the max number of tokens.
    while token_int != token_end and count_tokens < max_tokens:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = \
        {
            'transfer_values_input': transfer_values,
            'decoder_input': decoder_input_data
        }

        # Note that we input the entire sequence of tokens
        # to the decoder. This wastes a lot of computation
        # because we are only interested in the last input
        # and output. We could modify the code to return
        # the GRU-states when calling predict() and then
        # feeding these GRU-states as well the next time
        # we call predict(), but it would make the code
        # much more complicated.
        
        # Input this data to the decoder and get the predicted output.
        decoder_output = decoder_model.predict(x_data)

        # Get the last predicted token as a one-hot encoded array.
        # Note that this is not limited by softmax, but we just
        # need the index of the largest element so it doesn't matter.
        token_onehot = decoder_output[0, count_tokens, :]

        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer.token_to_word(token_int)

        # Append the word to the output-text.
        output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    # This is the sequence of tokens output by the decoder.
    output_tokens = decoder_input_data[0]

    # Plot the image.
#    plt.imshow(image)
#    plt.title(output_text.replace(" eeee",""))
#    plt.axis('off')
#    plt.show()
#    plt.savefig("test_results/test.png", bbox_inches='tight')
    
    # Print the predicted caption.
#    print("Predicted caption:")
#    print(output_text.replace(" eeee",""))
#    print()
    return output_text.replace(" eeee","")

def validation_check(verbose=0):
#    Check on 100 images picked at random from validation set
    val_captions=list()
    for i in range(100):
        idx=np.random.randint(0,1093)
        c=generate_caption(image_dir+filenames_val[idx])
        val_captions.append(copy(c))
        if verbose:
            print_progress(i+1,100)
    return val_captions

# MODEL DEFINITION


tokens_train=load_json('tokens_train')
filenames_train=load_json('filenames_train')

# This has to be tuned in order to not run out of memory
batch_size = 64
generator = batch_generator(batch_size=batch_size)
generator_val = batch_generator_val(batch_size=batch_size)


num_captions_train = [len(captions) for captions in captions_train]
total_num_captions_train = np.sum(num_captions_train)

# every epoch feeds to the network num_images_train images
# NOTE that these could be duplicate, as batch selection is completely random
# and doesn't exclude images from re-picking after they're picked once
steps_per_epoch = int(total_num_captions_train / batch_size)

# MODEL DEFINITION
state_size=512
embedding_size=128
# input the transfer values to the decoder
# NOTE that the transfer values size depends on the chosen CNN
# for VGG16, it's 4096
transfer_values_size=transfer_values_train.shape[1]
transfer_values_input = Input(shape=(transfer_values_size,),name='transfer_values_input')
# tanh is needed to limit the mapping output between -1 and 1
decoder_transfer_map = Dense(state_size,
                             activation='tanh',
                             name='decoder_transfer_map')
# the input "port" for the tokens
decoder_input = Input(shape=(None, ), name='decoder_input')
# embedding layer
# converts sequences of tokens to sequences of vectors
num_words=2000
decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')
# 3 Layers of GRU
decoder_gru1 = GRU(state_size, name='decoder_gru1',
                   return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',
                   return_sequences=True)
decoder_gru4 = GRU(state_size, name='decoder_gru4',
                   return_sequences=True)
decoder_gru5 = GRU(state_size, name='decoder_gru5',
                   return_sequences=True)
# output of gru layers is a tensor of shape
# [batch_size, sequence_length, state_size]
# each word is encoded as a vector of state_size length
# this must be converted back to integers and then to words
decoder_dense=Dense(num_words,
                    activation='linear',
                    name='decoder_output')

# connect and create the model
decoder_output = connect_decoder(transfer_values=transfer_values_input)

decoder_model = Model(inputs=[transfer_values_input, decoder_input],
                      outputs=[decoder_output])

# LOSS FUNCTION
# supply the correct caption (converted to one-hot) as target
# the sparse cross-entropy function does this automatically
# this optimizer seems to work better than adam optimizer for RNNs
optimizer=RMSprop(lr=1e-3)
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))

decoder_model.compile(optimizer=optimizer,
                      loss=sparse_cross_entropy,
                      target_tensors=[decoder_target])

# save each epoch in a different checkpoint
epoch_ctr=0
def train_net(num_epochs,epoch_ctr=0):
    eval_results=list()
    for i in range(num_epochs):
        path_checkpoint = str(epoch_ctr)+'_checkpoint.keras'
        callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                              verbose=1,
                                              save_weights_only=True)
        callback_tensorboard = TensorBoard(log_dir='./logs_ResNet50/',
                                           histogram_freq=0,
                                           write_graph=False)

        callbacks = [callback_checkpoint, callback_tensorboard]

        decoder_model.fit_generator(generator=generator,steps_per_epoch=steps_per_epoch,epochs=1,callbacks=callbacks)
        evaluation=decoder_model.evaluate_generator(generator=generator_val,steps=200)
        eval_results.append(evaluation)
        print(evaluation)
        epoch_ctr+=1
    return eval_results

    
