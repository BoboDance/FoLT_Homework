# Task 11.3

# a)
# the little yellow dog
# the cat

# b)
# Rapunzel
# her long golden hair

# c)
# Mary
# the cat
# the mat

# d)
# the president
# United

import nltk

grammar = r"""
NP: {<DT|PP\$>?<JJ>*<NN>} 
{<DT|PRP><NN|NNP|JJS>+<VBG>?<NNS>?}
{<CD>?<NN|JJR><TO|IN><CD><NN>}
{<POS><NN>}
{<NNP>+} 
"""

cp = nltk.RegexpParser(grammar)
sentence = [("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"), ("her", "PP$"), ("long", "JJ"), ("golden", "JJ"),
            ("hair", "NN")]
print(cp.parse(sentence))
