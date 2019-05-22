# This code computes the bleu score for the candidate sentence

import nltk
import math
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import json

from helpers import load_json
#from NN_architecture import generate_caption
# run the NN architecture before
captions_test=load_json('captions_test')

##generate_caption(path+filenames_test[0])
##with open('generated_captions_VGG19.txt') as inFile:
#with open('captions_vgg16/4_generated_captions_VGG16.txt') as inFile:
#    generated_test_captions=inFile.readlines()
#for i in range(len(generated_test_captions)):
##    THIS LINE REMOVES THE FIRST EMPTY SPACE
##    generated_test_captions[i]=generated_test_captions[i][1:]
#    generated_test_captions[i]=generated_test_captions[i].replace('\n','')
#

# load from json
with open('captions_vgg16/12_generated_captions_VGG16.json') as inFile:
    generated_test_captions=json.load(inFile)
c_to_insert=generated_test_captions[883]
generated_test_captions.insert(884,c_to_insert)


# build the references and candidate library
num_samples=len(generated_test_captions)
candidates=list()
references=list()
for i in range(num_samples):
    C=generated_test_captions[i]
    R=captions_test[i]
    
    refList=list()
    for j in range(5):
        refList.append(nltk.word_tokenize(R[j]))
    candidates.append(C)
    references.append(refList)

score=list()

chencherry=SmoothingFunction()

# function to add the control tokens to the sentences
def addCtrlSequence(string):
    outstr='ssss '+string+' eeee'
    return outstr

def appendPeriod(string):
    outstr=string+'.'
    return outstr

# try the corpus bleu

B1=list()
B2=list()
B3=list()
B4=list()
candidate_list=list()
reference_list=list()
for i in range(1093):
    references=captions_test[i]
    references_tokenized=list()
    for j in range(len(references)):
#        COMMENT/UNCOMMENT CORRESPONDING LINE TO CONSIDER SENTENCES w/ ssss eeee
#        references_tokenized.append(nltk.word_tokenize(addCtrlSequence(references[j])))
        references_tokenized.append(nltk.word_tokenize(appendPeriod(references[j])))
#        references_tokenized.append(nltk.word_tokenize(references[j]))
    candidate=generated_test_captions[i]
#    candidate=addCtrlSequence(generated_test_captions[i])
    candidate=appendPeriod(generated_test_captions[i])
    candidate_tokenized=nltk.word_tokenize(candidate)
#    s1=sentence_bleu(references_tokenized,candidate_tokenized,weights=[1,0,0,0])
#    s2=sentence_bleu(references_tokenized,candidate_tokenized,weights=[0.5,0.5,0,0])
#    s3=sentence_bleu(references_tokenized,candidate_tokenized,weights=[0.33,0.33,0.33,0])
#    s4=sentence_bleu(references_tokenized,candidate_tokenized,weights=[0.25,0.25,0.25,0.25])
#    B1.append(s1)
#    B2.append(s2)
#    B3.append(s3)
#    B4.append(s4)
# FOR THE CORPUS EVALUATION
    candidate_list.append(candidate_tokenized)
    reference_list.append(references_tokenized)

corpus_B1=corpus_bleu(reference_list,candidate_list,weights=[1,0,0,0],smoothing_function=chencherry.method1)
corpus_B2=corpus_bleu(reference_list,candidate_list,weights=[0.5,0.5,0,0],smoothing_function=chencherry.method2)
corpus_B3=corpus_bleu(reference_list,candidate_list,weights=[0.33,0.33,0.33,0],smoothing_function=chencherry.method3)
corpus_B4=corpus_bleu(reference_list,candidate_list,weights=[0.25,0.25,0.25,0.25],smoothing_function=chencherry.method4)

print('BLEU1:',corpus_B1)
print('BLEU2:',corpus_B2)
print('BLEU3:',corpus_B3)
print('BLEU4:',corpus_B4)

#avgB1=sum(B1)/len(B1)
#avgB2=sum(B2)/len(B2)
#avgB3=sum(B3)/len(B3)
#avgB4=sum(B4)/len(B4)


#    
#ref1a=nltk.word_tokenize(captions_test[i][0])
#ref1b=nltk.word_tokenize(captions_test[i][1])
#ref1c=nltk.word_tokenize(captions_test[i][2])
#ref1d=nltk.word_tokenize(captions_test[i][3])
#ref1e=nltk.word_tokenize(captions_test[i][4])
#hyp1=nltk.word_tokenize(generated_test_captions[i])
#
#ref2a=nltk.word_tokenize(captions_test[i+1][0])
#ref2b=nltk.word_tokenize(captions_test[i+1][1])
#ref2c=nltk.word_tokenize(captions_test[i+1][2])
#ref2d=nltk.word_tokenize(captions_test[i+1][3])
#ref2e=nltk.word_tokenize(captions_test[i+1][4])
#hyp2=nltk.word_tokenize(generated_test_captions[i+1])
#list_of_references=[[ref1a,ref1b,ref1c,ref1d,ref1e],[ref2a,ref2b,ref2c,ref2d,ref2e]]
#hypotheses=[hyp1,hyp2]
#corpus_score=corpus_bleu(list_of_references,hypotheses)
#score_1=sentence_bleu([ref1a,ref1b,ref1c,ref1d,ref1e],hyp1)
#score_2=sentence_bleu([ref2a,ref2b,ref2c,ref2d,ref2e],hyp2)