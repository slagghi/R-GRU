from helpers import load_image
import numpy as np
import copy
from helpers import load_json
import numpy as np

from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

from captions_preprocess import TokenizerWrap
from captions_preprocess import flatten
from captions_preprocess import mark_captions

# NOTE only load the necessary CNN
#from tensorflow.python.keras.applications import VGG16
#from tensorflow.python.keras.applications import VGG19
#from tensorflow.python.keras.applications import InceptionV3

#image_model = InceptionV3(include_top=True, weights='imagenet')
#transfer_layer=image_model.get_layer('avg_pool')
decoder_model.load_weights('best_models/InceptionV3_5layers/checkpoint.keras')
#transfer_values_test=np.load('../image_features/transfer_values/InceptionV3/transfer_values_test.npy')

# define the softmax function
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# This code implements beam search for a less-greedy sentence generation
# NOTE:
#   Before running this code, the NN_architecture code should be run beforehand
#   since it contains the model for the decoder

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

#img_size=(228,228)



#image_model = VGG19(include_top=True, weights='imagenet')
#transfer_layer=image_model.get_layer('fc2')

image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)
img_size=K.int_shape(image_model.input)[1:3]

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

# ASSUME I ALREADY HAVE THE TRANSFER VALUES FOR THE IMAGE
filenames_test=load_json('filenames_test')
path='../../../../../Desktop/parsingDataset/RSICD_images/'

# path for desktop computer
#path='../../../RSICD_images/'

#filename=filenames_test[812]
#image_path=path+filename
#image = load_image(image_path, size=img_size)
#image_batch = np.expand_dims(image, axis=0)
#transfer_values = image_model_transfer.predict(image_batch)

def get_test_captions(debug=0):
    test_captions=list()
    ctr=0
    for filename in filenames_test:
        print('Analysing ',filename)
        if filename=='square_40.jpg':
            filename='square_4.jpg'
        image_path=path+filename
        image=load_image(image_path,size=img_size)
        image_batch=np.expand_dims(image,axis=0)
        transfer_values=image_model_transfer.predict(image_batch)
        captions_list=beam_search(transfer_values)
        image_captions=list()
        for caption in captions_list:
            s=sequence_to_sentence(caption['sequence'])
            conf=getAvgConfidence(caption)
            cap={'sentence':s,'score':conf}
            image_captions.append(cap)
        test_captions.append(copy.copy(image_captions))
        
        
        ctr+=1
        progress=100*ctr/len(filenames_test)
        print('processed ',ctr,'images\t%.2f%%'% progress)
        if debug:
            if ctr==3:
                return test_captions
    return test_captions
    

tv_shape=transfer_values_test[0].shape[0]
def get_test_captions_tv(debug=0):
    test_captions=list()
    ctr=0
    for i in range(1093):
        transfer_values=transfer_values_test[i]
        transfer_values=np.reshape(transfer_values,(1,tv_shape))
        captions_list=beam_search(transfer_values)
        image_captions=list()
        for caption in captions_list:
            s=sequence_to_sentence(caption['sequence'])
            conf=getAvgConfidence(caption)
            cap={'sentence':s,'score':conf}
            image_captions.append(cap)
        test_captions.append(copy.copy(image_captions))
        
        
        ctr+=1
        print_progress(i+1,1094)
        if debug:
            if ctr==3:
                return test_captions
    return test_captions


# This code, given a transfer vector and the previous sequence, predicts the
# K best next words (start with 2)
prev_sequence=np.zeros(shape=(1,30),dtype=np.int)
prev_sequence[0,0]=token_start
count_tokens=0

def predict_next_word(transfer_values,prev_sequence,count_tokens,guessNr):
    x_data=\
    {
     'transfer_values_input':transfer_values,
     'decoder_input':prev_sequence
     }
    decoder_output=decoder_model.predict(x_data)
#    compute the softmax in order to get confidence scores between 0 and 1
    token_onehot = decoder_output[0, count_tokens, :]
    token_onehot = softmax(token_onehot)

    [outToken,confidence]=nth_best(token_onehot,guessNr)
    outWord=tokenizer.token_to_word(outToken)
    
    return outToken,confidence

#def beam_search(transfer_value):
##    Cycle on words, with a max of 30
##   the first predicted sequence is just the start token
#    prev_sequence=np.zeros(shape=(1,30),dtype=np.int)
#    prev_sequence[0,0]=token_start
#    count_tokens=0
#    nr_guesses=3
#    caption_list=list()
#    starter_caption={
#            'sequence':prev_sequence,
#            'confidence':0
#            }
#    caption_list.append(starter_caption)
#    for count_tokens in range(2):
##        for each caption in the list compute nr_guesses new captions and put them in a list
#        new_captions=list()
#        for caption in caption_list:
#            prev_sequence=caption['sequence']
#            prev_confidence=caption['confidence']
#            for guess_iter in range(1,nr_guesses):
#                [token,confidence]=predict_next_word(transfer_values,prev_sequence,count_tokens,guess_iter)
##                update the sequence
#                new_sequence=copy.copy(prev_sequence)
#                new_sequence[0,count_tokens+1]=token
#                new_caption={
#                        'sequence':copy.copy(new_sequence),
#                        'confidence':prev_confidence+confidence
#                        }
#                new_captions.append(copy.copy(new_caption))
#        caption_list=list()
#        caption_list=new_captions[:]
#    return caption_list

# this function takes a list of incomplete captions and makes
# N guesses for the next word
nr_guesses=3
def get_guesses(transfer_values,caption,prev_confidence):
#    if the caption is already completed, don't make further guesses
    if token_end in caption:
        return caption
    new_captions=list()
    count_token=get_tokencount(caption)
    for guess_iter in range(1,nr_guesses+1):
        [token,confidence]=predict_next_word(transfer_values,caption,count_token,guess_iter)
        new_sequence=copy.copy(caption)
        new_sequence[0,count_token+1]=token
        new_caption={'sequence':copy.copy(new_sequence),'confidence':prev_confidence+confidence}
        new_captions.append(copy.copy(new_caption))
    return new_captions
def get_tokencount(sequence):
    ctr=-1
    for i in range(30):
        if sequence[0,i]==0:
            break
        ctr+=1
    return ctr
def sequence_to_sentence(sequence,verbose=0):
    length=sequence.shape[1]
    s=""
    for i in range(length):
        t=sequence[0,i]
        if t==0:
            break
        w=tokenizer.token_to_word(t)
        s+=" "
        s+=w
    if verbose:
        print(s)
    s=s.replace('ssss ','')
    s=s.replace(' eeee','')
    return s

sent_len=29
def beam_search(transfer_values):
    starter_sequence=np.zeros(shape=(1,30),dtype=np.int)
    starter_sequence[0,0]=token_start
    caption_list=list()
    starter_caption={
            'sequence':starter_sequence,
            'confidence':0
            }
    caption_list.append(starter_caption)
    for i in range(sent_len):
        new_captions=list()
        for caption in caption_list:
# if caption is complete, automatically save it
            if isComplete(caption):
                new_captions.append(copy.copy(caption))
                continue
            guesses=get_guesses(transfer_values,caption['sequence'],caption['confidence'])
            for guess in guesses:
                new_captions.append(copy.copy(guess))
#        caption_list=list()
#        caption_list=copy.copy(new_captions)
#        print(i)
#        only keep n of the best captions
        confVector=list()
        for caption in new_captions:
            conf=getAvgConfidence(caption)
            confVector.append(conf)
#        get the n best confidences
        guesses2keep=5
        best_guesses=list()
        for bestGuess_iter in range(1,min(guesses2keep+1,len(confVector)+1)):
            [best_position,best_value]=nth_best(confVector,bestGuess_iter)
#            debug print
#            print(best_guesses)
#            print(new_captions[best_position])
#            if new_captions[best_position] not in best_guesses:
            best_guesses.append(copy.copy(new_captions[best_position]))
        caption_list=list()
        caption_list=copy.copy(best_guesses)
#        if all the captions are complete, save the list and return it
        if allComplete(caption_list):
            return caption_list
#        print('candidate captions: ',len(caption_list))
    return caption_list
# This function returns the n-th highest value in a vector
def nth_best(vector,n):
    v=copy.copy(vector)
#    discard (n-1) biggest elements
    for i in range(n-1):
        best=np.argmax(v)
        v[best]=-100
    best_position=np.argmax(v)
    best_value=max(v)
    return best_position,best_value
# This code normalises a vector in a [0,1] range
def normalise(vector):
    M=np.max(vector)
    m=np.min(vector)
    vector=(vector-m)/(M-m)
    return vector

def getAvgConfidence(caption):
    sequence=caption['sequence']
    confidence=caption['confidence']
#    get the length of the sequence
    l=getSeqLen(sequence)
    avgConfidence=confidence/l
    return avgConfidence

# this function checks whether the current caption is complete
def isComplete(caption):
    if 2 in caption['sequence']:
        return True
    else: 
        return False
# this function checks if all the captions in the list are complete
def allComplete(capList):
    allComplete=True
    for c in capList:
        if isComplete(c):
            continue
#        this never gets executed if all the captions are complete
        allComplete=False
    return allComplete
def getSeqLen(sequence,verbose=0):
    l=0
    for i in range(30):
        if sequence[0,i]==0:
            break
        l+=1
        if sequence[0,i]==2:
            break
    if verbose:
        print(l)
    return l