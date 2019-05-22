''' 
this code computes the consensus score and word score
'''
import numpy as np
import matplotlib.pyplot as plt
from helpers import load_image
from helpers import load_json
from helpers import print_progress
from copy import copy
import json

from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

#from tensorflow.python.keras.models import Model

transfer_values_train=np.load('../image_features/transfer_values/VGG16/transfer_values_train.npy')
transfer_values_test=np.load('../image_features/transfer_values/VGG16/transfer_values_test.npy')
captions_train=load_json('captions_train')

# LOAD THE CANDIDATE CAPTIONS
BS_filename='test_captions_InceptionV3_5layers.json'
with open(BS_filename,'r') as f:
    candidate_captions=json.load(f)
# Load the transfer model
# After first execution, you can comment these lines
#from tensorflow.python.keras.applications import VGG16
#image_model = VGG16(include_top=True, weights='imagenet')
#transfer_layer=image_model.get_layer('fc2')
#image_model_transfer = Model(inputs=image_model.input,
#                             outputs=transfer_layer.output)
image_dir='../../../../../Desktop/parsingDataset/RSICD_images/'
filenames_test=load_json('filenames_test')

img_size=(224,224)
def get_transfer_values_old(image_path):
    '''
    Compute the transfer values for an image
    given the image transfer model
    '''
    
    # Load and resize the image.
    image = load_image(image_path, size=img_size)
    
    # Expand the 3-dim numpy array to 4-dim
    # because the image-model expects a whole batch as input,
    # so we give it a batch with just one image.
    image_batch = np.expand_dims(image, axis=0)

    # Process the image with the pre-trained image-model
    # to get the transfer-values.
    transfer_values = image_model_transfer.predict(image_batch)
    return transfer_values

# This function loads the transfer values from a pre-existing numpy matrix
tv_len=transfer_values_test[0].shape[0]
def get_transfer_values(image_path):
    filename=image_path[len(image_dir):]
    for i in range(len(filenames_test)):
        if filenames_test[i] == filename:
            break
    transfer_values=transfer_values_test[i]
    transfer_values=np.reshape(transfer_values,(1,tv_len))
    return transfer_values

def get_difference(transfer_value1,transfer_value2):
    '''
    Compute the difference between 2 images
    via the squared norm of the difference of the transfer values
    '''
    diff=transfer_value1-transfer_value2
    norm=np.linalg.norm(diff)
    return norm*norm

# This function returns the n-th lowest value in a vector
def get_nth_minimum(vector,n):
    v=copy(vector)
    for i in range(n):
        best=np.argmin(v)
        v[best]=1000
    return best
# same as before, only for the highest values
def get_nth_maximum(vector,n):
    v=copy(vector)
    for i in range(n):
        best=np.argmax(v)
        v[best]=0
    return best
    
chencherry=SmoothingFunction()
def bleu(reference,candidate):
    reference_tokenized=word_tokenize(reference)
    reference_list=list()
    reference_list.append(reference_tokenized)
    candidate_tokenized=word_tokenize(candidate)
    
    score=sentence_bleu(reference_list,candidate_tokenized,weights=[0.5,0.5,0,0],smoothing_function=chencherry.method1)
    return score

K=5
M=3
def KNN_caption(image_name,verbose=0):
    image_path=image_dir+image_name
    
    if verbose:
        print('Computing transfer values...')
    
    transfer_value=get_transfer_values(image_path)
    num_train_images=np.shape(transfer_values_train)[0]
    diff_list=list()
    
    if verbose:
        print('Computing differences...')
    
    for i in range(num_train_images):
        diff=get_difference(transfer_value,transfer_values_train[i])
        diff_list.append(diff)
    
    if verbose:
        print('Getting best images...')
    
    best_train_ids=list()
    for i in range(1,K+1):
        best_i=get_nth_minimum(diff_list,i)
        best_train_ids.append(best_i)
    
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
        print('Choosing most significant caption...')
    
    consensus_score=list()
    for caption in captions_repository:
        caption_scores=list()
#        create a reference list, which contains all OTHER captions
        ref_captions=list()
        ref_captions=copy(captions_repository)
        ref_captions.remove(caption)
        
        for refCaption in ref_captions:
            score=bleu(refCaption,caption)
            caption_scores.append(score)
#        caption_consensus=np.max(caption_scores)
#        CONSENSUS SCORE COMPUTATION
#            get the M highest scores and average them
        M_highest_scores=list()
        for i in range(1,M+1):
            high_score=get_nth_maximum(caption_scores,i)
            M_highest_scores.append(high_score)
        highscore_list=list()
        for idx in M_highest_scores:
            highscore_list.append(caption_scores[idx])
        caption_consensus=np.mean(highscore_list)
        
        consensus_score.append(caption_consensus)
#    return the best caption
    best_scoring_index=np.argmax(consensus_score)
    best_caption=captions_repository[best_scoring_index]
    
    return best_caption

def bulk_generation_KNN():
    num_test_images=np.shape(filenames_test)[0]
    generated_captions=list()
    for i in range(num_test_images):
        image_name=filenames_test[i]
        if i==884:
            image_name=filenames_test[883]
        caption=KNN_score(image_name)
        generated_captions.append(caption)
        print_progress(i,num_test_images)
    return generated_captions
        
def consensus_score(candidate_caption,transfer_value,transfer_values_train,verbose=0):
    num_train_images=np.shape(transfer_values_train)[0]
    diff_list=list()
    
    if verbose:
        print('Computing differences...')
    
    for i in range(num_train_images):
        diff=get_difference(transfer_value,transfer_values_train[i])
        diff_list.append(diff)
    
    if verbose:
        print('Getting best images...')
    
    best_train_ids=list()
    for i in range(1,K+1):
        best_i=get_nth_minimum(diff_list,i)
        best_train_ids.append(best_i)
    
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
        s=bleu(refCaption,candidate_caption)
        bleu_scores.append(s)
    consensus_score=np.mean(bleu_scores)
    return consensus_score

def word_in_image(word,captions):
#    returns 1 if the description of the image contain the word
    for caption in captions:
        caption_t=nltk.word_tokenize(caption)
        if word in caption_t:
            return 1
    return 0

def images_with_word(captions_set,word):
    idxSet=list()
    for i in range(len(captions_set)):
        if word_in_image(word,captions_set[i]):
            idxSet.append(i)
    return idxSet

# Create a word lookup table
#    that tells you in which images the word appears
with open('train_captions_bow.json','r') as f:
    bow=json.load(f)
def create_word_LUT():
    word_lookup=list()
    for i in range(len(bow)):
#    for i in range(10):
        word=bow[i]['word']
        indices=images_with_word(captions_train,word)
        
        entry={'word':word,'indices':indices}
        word_lookup.append(copy(entry))
        print_progress(i+1,len(bow))
    return word_lookup
        
sigma_VGG16=221.75
def word_score(word,image,captions_train):
    '''
    Word score computation:
    get the images that contain the candidate word
    get the kernel score: exp(-(diff(I-Ij))/sigma^2)
    score is sum of kernel scores divided by number of images with cand. word
    '''
    # this denotes the images that contain the word
    G_idx=images_with_word(captions_train,word)
    count_G=len(G_idx)
#    if no other caption contains the word, its score is 0
    if count_G==0:
        return 0
#    
#    K_list=list()
#    for i in G_idx:
#        K=gaussian_kernel(image,transfer_values_train[i])
#        K_list.append(K)
#    return np.mean(K_list)


#        this computes the gaussian kernel
def gaussian_kernel(image1,image2,sigma=sigma_VGG16):
    diff=get_difference(image1,image2)
    k_arg=-(diff/(sigma))
    kernel=np.e**(k_arg)
    return kernel

    