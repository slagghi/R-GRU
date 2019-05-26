# This code generates the caption using consensus references
# word by word

from helpers import load_image
import numpy as np
import copy
from helpers import load_json
import numpy as np
from helpers import print_progress

from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

from captions_preprocess import TokenizerWrap
from captions_preprocess import flatten
from captions_preprocess import mark_captions

from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

decoder_model.load_weights('best_models/InceptionV3_5layers/1_checkpoint.keras')

image_dir='../../Desktop/parsingDataset/RSICD_images/'

inception_tv_train=np.load('image_features/transfer_values/InceptionV3/transfer_values_train.npy')
inception_tv_test =np.load('image_features/transfer_values/InceptionV3/transfer_values_test.npy')

captions_train=load_json('captions_train')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

chencherry=SmoothingFunction()
def bleu(reference,candidate,grade=1):
    reference_tokenized=word_tokenize(reference)
    reference_list=list()
    reference_list.append(reference_tokenized)
    candidate_tokenized=word_tokenize(candidate)
    
    if grade==1:
        weights=[1,0,0,0]
    if grade==2:
        weights=[0.5,0.5,0,0]
    if grade==3:
        weights=[0.33,0.33,0.33,0]
    if grade==4:
        weights=[0.25,0.25,0.25,0.25]
    
    score=sentence_bleu(reference_list,candidate_tokenized,weights=weights,smoothing_function=chencherry.method1)
    return score

def generate_caption_greedy(image_path, max_tokens=30):
    """
    Generate a caption for the image in the given path.
    At each step, we only keep the best word
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

    return output_text[1:].replace(" eeee","")

def nth_best(vector,n):
    v=copy.copy(vector)
#    discard (n-1) biggest elements
    for i in range(n-1):
        best=np.argmax(v)
        v[best]=-100
    best_position=np.argmax(v)
    best_value=max(v)
    return best_position,best_value

# these functions only returns the nth highest and lowest values
#    (not their indices)
def get_nth_minimum(vector,n):
    v=copy.copy(vector)
    for i in range(n):
        best=np.argmin(v)
        v[best]=1000
    return best


beam_size=10
def get_guesses_GRU(transfer_value,prev_sequence,count_tokens):
    transfer_values=np.reshape(transfer_value,(1,2048))
    x_data=\
    {
     'transfer_values_input':transfer_values,
     'decoder_input':prev_sequence
     }
    decoder_output=decoder_model.predict(x_data)
    
    word_likelihood = decoder_output[0, count_tokens, :]
    word_likelihood = softmax(word_likelihood)
    word_likelihood = np.log(word_likelihood)
    
    GRU_guesses=dict()
    
    GRU_tokens=list()
    GRU_confidences=list()
    
    for i in range(1,beam_size+1):
        [outToken,confidence]=nth_best(word_likelihood,i)
        GRU_guesses[outToken]=copy.copy(confidence)
        
        GRU_tokens.append(outToken)
        GRU_confidences.append(confidence)
    
#    GRU_confidences=normalise(GRU_confidences)
    return GRU_tokens,GRU_confidences

def normalise(vector):
    M=np.max(vector)
    m=np.min(vector)
    if M==m:
        return np.zeros(len(vector))
    vector=(vector-m)/(M-m)
    return vector

# get K most similar images
k=5
inception_tv_train=np.load('image_features/transfer_values/InceptionV3/transfer_values_train.npy')
num_train_images=len(inception_tv_train)
def get_k_nearest_images(transfer_value,verbose=0):
    if verbose:
        print('Computing differences...')
    
    diff_list=list()
    for i in range(num_train_images):
        diff=get_difference(transfer_value,transfer_values_train[i])
        diff_list.append(diff)
    
    if verbose:
        print('Getting best images...')
    
    best_train_ids=list()
    for i in range(1,k+1):
        best_i=get_nth_minimum(diff_list,i)
        best_train_ids.append(best_i)
#        print(diff_list[best_i])
        
    return best_train_ids

def get_difference(transfer_value1,transfer_value2):
    '''
    Compute the difference between 2 images
    via the squared norm of the difference of the transfer values
    '''
    diff=transfer_value1-transfer_value2
    norm=np.linalg.norm(diff)
    return norm*norm


def consensus_score(candidate_caption,best_train_ids,verbose=0):
    num_candidate_images=np.shape(best_train_ids)[0]
    captions_repository=list()
    for i in range(num_candidate_images):
        captions=captions_train[best_train_ids[i]]
        for caption in captions:
#            Don't put a caption on the list if it's already present
            if caption in captions_repository:
                continue
            captions_repository.append(caption)
    
    if verbose:
        print('Computing caption score')
    
    bleu_scores=list()
    for refCaption in captions_repository:
        s=bleu(refCaption,candidate_caption,4)
        bleu_scores.append(s)
#    consensus_score=np.mean(bleu_scores)
        
#    the final consensus is defined as the mean of the m highest bleu4 scores
    m=3
    best_bleus=list()
    for i in range(1,m+1):
        [nth_position,nth_value]=nth_best(bleu_scores,i)
        best_bleus.append(nth_value)
    
    consensus_score=np.mean(best_bleus)
    
#    consensus_score=np.mean(bleu_scores)
    
    return consensus_score

max_sent_len=29
def generate_caption_reference(transfer_value,alpha=0.5,verbose=0):
    token_sequence=np.zeros(shape=(1,30), dtype=np.int)
    token_sequence[0,0]=token_start
    generated_caption=""
    
#    most similar images (used for computing the consensus score)
    k_nearest_ids=get_k_nearest_images(transfer_value)
    
    for word_ctr in range(max_sent_len):
        [GRU_tokens,GRU_confidences]=get_guesses_GRU(transfer_value,token_sequence,word_ctr)
#        normalise the GRU guesses
        
        word_scores=list()

        for i in range(len(GRU_tokens)):
            word_data=dict()
            word_data['token']=GRU_tokens[i]
            word_data['GRU_score']=GRU_confidences[i]
            word_data['word']=tokenizer.token_to_word(GRU_tokens[i])
            word_data['sentence']=generated_caption+" "+word_data['word']
            
            cons_score=consensus_score(word_data['sentence'],best_train_ids)
            word_data['consensus_score']=cons_score
            
            word_scores.append(copy.copy(word_data))
        
#            print(word_data)
        
#        return word_scores
        
        
            
#            the scores must be normalised
        GRU_to_normalise=list()
        cons_to_normalise=list()
        for i in range(len(word_scores)):
            GRU_to_normalise.append(word_scores[i]['GRU_score'])
            cons_to_normalise.append(word_scores[i]['consensus_score'])
                        
#        print(cons_to_normalise)
        
        GRU_normalised=normalise(GRU_to_normalise)
        cons_normalised=normalise(cons_to_normalise)
        
        reference_scores=alpha*GRU_normalised+(1-alpha)*cons_normalised
        winner=np.argmax(reference_scores)
        chosen_word=word_scores[winner]['word']
        chosen_token=word_scores[winner]['token']
        
        if verbose:
            print(chosen_word)

        token_sequence[0,word_ctr+1]=chosen_token
        generated_caption=generated_caption+" "+chosen_word
        
        if chosen_token==token_end:
            break
        
#    print(generated_caption)
    return generated_caption[1:].replace(' eeee','')


# this function is used to generate the captions for all the images in the test set
# tune alpha accordingly

def generate_captions_testset(alpha):
    print("Generating captions for alpha =",alpha)
    generated_captions=list()
    for i in range(len(transfer_values_test)):
        transfer_value=transfer_values_test[i]
        generated_caption=generate_caption_reference(transfer_value,alpha=alpha)
        generated_captions.append(generated_caption)
        print_progress(i+1,len(transfer_values_test))
    filename='generated_captions_reference_alpha_'+str(alpha)+'.json'
    with open(filename,'w') as f:
        json.dump(generated_captions,f)