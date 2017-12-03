import nltk
from nltk import *

# First of all, this function creates a list including all elements from the input, which are longer than 5, whereby it only selects the last two chars ([w[-2:] for w in text_list if len(w)>=5]). Afterwards, a nltk.Text object is created and the frequency distribution of the most common 10 words is returned, which are in this case just the endings. 

def top_suffixes(text_list):
     return FreqDist(Text([w[-2:] for w in text_list if len(w)>=5])).most_common(10)
