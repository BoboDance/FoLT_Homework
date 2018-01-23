import nltk

grammar1 = nltk.CFG.fromstring("""
S -> NP VP
VP -> TV NP | TV NP PP | ITV
P -> "on" | "to"
PP -> P NP
TV -> "brought" | "gave" | "put"
ITV -> "fell"
NP -> "John" | "Mary" | "Bob" | Det N
Det -> "a" | "A" | "The" | "the"
N -> "gift" | "dog" | "cake" | "party" | "bottle" | "table"
""")

# Task 12.1 a)
sentence1 = 'Mary brought a cake'  # Yes
sentence2 = 'Mary gave a gift to Bob'  # Yes
sentence3 = 'Mary brought John to a party'  # Yes
sentence4 = 'Mary gave Bob a gift'  # No
sentence5 = 'A dog brought a party to Bob'  # Yes


# Task 12.1 b)
rd_parser = nltk.RecursiveDescentParser(grammar1)
for t in rd_parser.parse(sentence3.split()):
    print(t)

# Task 12.1 c)
sentence6 = 'Bob put a bottle on a table'
sentence7 = 'The bottle fell'

rd_parser = nltk.RecursiveDescentParser(grammar1)
for t in rd_parser.parse(sentence6.split()):
    print(t)

rd_parser = nltk.RecursiveDescentParser(grammar1)
for t in rd_parser.parse(sentence7.split()):
    print(t)

sentence8 = 'Bob brought'
for t in rd_parser.parse(sentence8.split()):
    print(t)
