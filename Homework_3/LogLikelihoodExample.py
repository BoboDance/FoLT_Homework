'''
Homework 3.1 support material

Log-likelyhood(LL) is used to measure how improbable a certain phrase is
with respect to a certain background corpus. You can find the
expalnation of LL in the Homework 3.1 task description.

This example uses words but it is equally applicable to bigrams and any
other type of phrases.

Created on Oct 19, 2017
@author: Daniil Sorokin<sorokin@ukp.informatik.tu-darmstadt.de>
'''


import math
import nltk
from nltk.corpus import brown, gutenberg

# Access ids of text that are available in the Gutenberg corpus
print("Gutenberg file ids: ", gutenberg.fileids())

# Retrieve a certain text by id
alice_words = gutenberg.words('carroll-alice.txt')
print(alice_words[:10])

# Access the whole brown corpus (more than a million words)
brown_words = brown.words()
print(brown_words[:10])

# Compute A,B,C,D for the word "Alice" in "Alice in Wonderland"
# as our document and Brown corpus as background corpus
A_alice = alice_words.count("Alice") # How often "Alice" is used in the Document
B_alice = brown_words.count("Alice") # How often "Alice" is used in general in the language (computed on Brown Corpus)
C = len(alice_words) # Size of the Document
D = len(brown_words) # Size of the Brown Corpus

# Compute A,B for the word "me" in "Alice in the Wonderland" (C,D are the same)
A_me = alice_words.count("me") # How often "me" is used in the Document
B_me = brown_words.count("me") # How often "me" is used in general in the language

# Compute expectations - how often we expect to see a word in the Document and the Corpus
# Expectation is the probablity to see the word based on all data that we have:
# 	(A_alice+B_alice) / (C + D)
# multiplied by the size of the corpus or the document.
E1_alice = C * (A_alice+B_alice) / (C + D) # Expectation for "Alice" in "Alice in Wonderland"
E2_alice = D * (A_alice+B_alice) / (C + D) # Expectation for "Alice" in the Brown Corpus

E1_me = C * (A_me+B_me) / (C + D) # Expectation for "me" in "Alice in Wonderland"
E2_me = D * (A_me+B_me) / (C + D) # Expectation for "me" in the Brown Corpus

# Compute LL for "Alice" in "Alice in Wonderland". math.log method is used for logarithm.
LL_alice = 2*(A_alice * math.log(A_alice/E1_alice) + B_alice * math.log(B_alice/E2_alice))

# Compute LL for "me" in "Alice in Wonderland".
LL_me = 2*(A_me * math.log(A_me/E1_me) + B_me * math.log(B_me/E2_me))

print("LL_alice = ", LL_alice)
print("LL_me = ", LL_me)