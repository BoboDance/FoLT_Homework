import nltk
from math import sqrt
from nltk.corpus import udhr, gutenberg


def build_language_models(languages, words):
    # I only check for values not given, they still can be wrong, so please avoid invalid inputs :)
    if languages is None or words is None:
        raise ValueError("languages and words are mandatory.")

    return nltk.ConditionalFreqDist(
        (lang, char.lower())
        for lang in languages
        for word in words[lang]
        for char in word if char.isalpha()
        # alternatively, one could not remove non alphabetic chars, however numbers, bracket and the like
        # are not relevant to detect the language, means are equally used, at least according to a lecture I attended.
        # for char in word
    )


def calc_score(text, language_model, language):
    # get char distribution/frequency of given text
    fd = nltk.FreqDist(
        (char.lower()
         for word in text
         for char in word if char.isalpha())
    )

    difference = 0

    # Iterate over all keys of given language and compare its frequency to the frequency of the given text
    # Calculate Minkowski L1 distance for the language
    # Distance should be zero in order to have a 100% match with the given language
    # Means, the smaller the overall distance the better --> more likely to have the given language
    # Further, euclidean distance does not seem to work well.
    # Detecting german was not really possible, however when I extended the input sentence it worked better
    # and german could be predicted as well.
    # Nevertheless, Minkowski L1 distance seems to yield in better results.
    # Maybe because compared to euclidean distance it scales better for greater vector sizes,
    # which probably could be considered the case.
    # The vector eventually includes all possible alphanumeric values of its language.
    # Cosine similarity is also not really feasible for this comparison.
    # My guess is, it mainly depends on the fact that languages can have different distributions,
    # which have a similar angle from the (0,0).
    # Therefore the results may vary, especially for short texts like in this example.
    # In (Thomas Gottron und Nedim Lipka, A Comparison of Language Identification Approaches on Short Query-Style Texts,
    # 2010. URL: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.155.5553&rep=rep1&type=pdf,
    # accessed 19.11.2017) is stated that Naive Bayes is probably the best classifier for short texts.
    # However, I am assuming implementing a Naive Bayes classifier for this exercise is out of the scope for now.

    # euclidean distance
    # for char in language_model[language].keys():
    #     difference += pow(language_model[language].freq(char) - fd.freq(char), 2)
    # return sqrt(difference)

    # cosine similarity
    # product = 0
    # sumA = 0
    # sumB = 0
    #
    # for char in language_model[language].keys():
    #     A = fd.freq(char)
    #     B = language_model[language].freq(char)
    #     product += A * B
    #     sumA += pow(A, 2)
    #     sumB += pow(B, 2)
    # if sumA == 0 or sumB == 0:
    #     return 0
    # return product / (sqrt(sumA) * sqrt(sumB))

    # Minkowski L1 distance
    for char in language_model[language].keys():
        difference += abs(language_model[language].freq(char) - fd.freq(char))

    # add frequencies to distance which can be found in the text but not in the language base.
    s = set(lang_model[language])
    t = set(fd)
    for char in t.difference(s):
        difference += fd.freq(char)

    return difference


def guess_language(language_model_cfd, text):

    result = nltk.defaultdict(int)

    # compare language base frequencies with given text frequencies
    for lang in language_model_cfd:
        result[lang] = calc_score(text, language_model_cfd, lang)

    # return minimum score
    return min(result, key=result.get)


languages = ['English', 'German_Deutsch', 'French_Francais']
# build the language models
lang_base = dict((language, udhr.words(language + '-Latin1')) for language in languages)
lang_model = build_language_models(languages, lang_base)

text1 = "Peter had been to the office before they arrived."
text2 = "Si tu finis tes devoirs, je te donnerai des bonbons."
text3 = "Das ist ein schon recht langes deutsches Beispiel."
# guess the language by comparing the frequency distributions
print('guess for english text is', guess_language(lang_model, text1))
print('guess for french text is', guess_language(lang_model, text2))
print('guess for german text is', guess_language(lang_model, text3))

# test guesser for emma txt
# print(guess_language(lang_model, gutenberg.words("austen-emma.txt")))

# print and check if cfd is correct
# print the models for visual inspection (you always should have a look at the data :)
# for language in languages:
#     for key in list(language_model_cfd[language].keys())[:10]:
#         print(language, key, "->", language_model_cfd[language].freq(key))
