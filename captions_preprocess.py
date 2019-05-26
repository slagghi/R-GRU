#  This script preprocesses the images

import json
from helpers import load_json
from tensorflow.python.keras.preprocessing.text import Tokenizer


mark_start='ssss '
mark_end=' eeee'
def mark_captions(captions_listlist):
    captions_marked = [[mark_start + caption + mark_end
                        for caption in captions_list]
                        for captions_list in captions_listlist]
    
    return captions_marked
def flatten(captions_listlist):
    captions_list=[caption
                  for captions_list in captions_listlist
                  for caption in captions_list]
    return captions_list
# to tokenize the captions
num_words=2000
class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""
    
    def __init__(self, texts, num_words=None):
        """
        :param texts: List of strings with the data-set.
        :param num_words: Max number of words to use.
        """

        Tokenizer.__init__(self, num_words=num_words)

        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""

        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        
        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text
    
    def captions_to_tokens(self, captions_listlist):
        """
        Convert a list-of-list with text-captions to
        a list-of-list of integer-tokens.
        """
        
        # Note that text_to_sequences() takes a list of texts.
        tokens = [self.texts_to_sequences(captions_list)
                  for captions_list in captions_listlist]
        
        return tokens
# mark the training captions and flatten them
captions_train=load_json('captions_train')
captions_train_marked=mark_captions(captions_train)
captions_train_flat=flatten(captions_train_marked)
tokenizer=TokenizerWrap(texts=captions_train_flat,
                        num_words=2000)
token_start=tokenizer.word_index[mark_start.strip()]
token_end=tokenizer.word_index[mark_end.strip()]
tokens_train=tokenizer.captions_to_tokens(captions_train_marked)

with open('dataset/tokens_train.json','w') as outfile:
    json.dump(tokens_train,outfile)
