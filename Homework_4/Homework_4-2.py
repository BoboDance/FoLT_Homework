import nltk
from math import sqrt
from nltk.corpus import udhr

# language model for tokens
def build_language_models_token(languages, words):
    if languages is None or words is None:
        raise ValueError("languages and words are mandatory.")

    # initialize CondFreqDist with tuples (language, word)
    # remove non alphabetic words, they do not contribute to the language prediction
    return nltk.ConditionalFreqDist(
        (lang, token.lower())
        for lang in languages
        for token in words[lang] if token.isalpha())

# language model for character bigrams
def build_language_models_char_bigram(languages, words):
    if languages is None or words is None:
        raise ValueError("languages and words are mandatory.")

    lang_dict = nltk.defaultdict(int)

    # build character bigrams for every language beforehand
    for lang in languages:
        lang_dict[lang] = nltk.bigrams(
            char.lower()
            for token in words[lang]
            for char in token if char.isalpha()
        )

    # use previously created dict to initialize CondFreqDist with (language, char_bigram) tuples
    return nltk.ConditionalFreqDist(
        (lang, bigram)
        for lang in languages
        for bigram in lang_dict[lang]
    )


# language model for token bigrams
def build_language_models_token_bigram(languages, words):
    if languages is None or words is None:
        raise ValueError("languages and words are mandatory.")

    lang_dict = nltk.defaultdict(int)

    # build token bigrams for every language beforehand
    for lang in languages:
        lang_dict[lang] = nltk.bigrams(
            token.lower()
            for token in words[lang] if token.isalpha()
        )
    # use previously created dict to initialize CondFreqDist with (language, token_bigram) tuples
    return nltk.ConditionalFreqDist(
        (lang, bigram)
        for lang in languages
        for bigram in lang_dict[lang]
    )


def calc_score(language_model, language, fd):

    difference = 0

    # Euclidean distance
    # for char in language_model[language].keys():
    #     difference += pow(language_model[language].freq(char) - fd.freq(char), 2)
    # return sqrt(difference)

    # Cosine similarity
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
    for value in language_model[language].keys():
        difference += abs(language_model[language].freq(value) - fd.freq(value))

    # add frequencies to distance which can be found in the text but not in the language base.
    s = set(lang_model[language])
    t = set(fd)
    for v in t.difference(s):
        difference += fd.freq(v)

    return difference


def guess_language_token(language_model_cfd, text):

    # create token FreqDist for input text
    fd = nltk.FreqDist(
        (word.lower()
         for word in text if word.isalpha())
    )
    result = nltk.defaultdict(int)

    # compare language base frequencies with given text frequencies
    for lang in language_model_cfd:
        result[lang] = calc_score(language_model_cfd, lang, fd)

    return min(result, key=result.get)


def guess_language_char_bigram(language_model_cfd, text):

    # create character bigram FreqDist for input text
    fd = nltk.FreqDist(
        nltk.bigrams(char.lower()
                     for word in text
                     for char in word if char.isalpha())
    )
    result = nltk.defaultdict(int)

    # compare language base frequencies with given text frequencies
    for lang in language_model_cfd:
        result[lang] = calc_score(language_model_cfd, lang, fd)

    return min(result, key=result.get)


def guess_language_token_bigram(language_model_cfd, text):

    # create token bigram FreqDist for input text
    fd = nltk.FreqDist(
        nltk.bigrams(word.lower()
                     for word in text if word.isalpha())
    )
    result = nltk.defaultdict(int)

    # compare language base frequencies with given text frequencies
    for lang in language_model_cfd:
        result[lang] = calc_score(language_model_cfd, lang, fd)

    return min(result, key=result.get)


languages = ['English', 'German_Deutsch', 'French_Francais']
# build the language models
lang_base = dict((language, udhr.words(language + '-Latin1')) for language in languages)

text1 = "Peter had been to the office before they arrived."
text2 = "Si tu finis tes devoirs, je te donnerai des bonbons."
text3 = "Das ist ein schon recht langes deutsches Beispiel."

lang_model = build_language_models_token(languages, lang_base)
# guess the language by comparing the frequency distributions of tokens
print('token: guess for english text is', guess_language_token(lang_model, text1))
print('token: guess for french text is', guess_language_token(lang_model, text2))
print('token: guess for german text is', guess_language_token(lang_model, text3))
print()

lang_model = build_language_models_char_bigram(languages, lang_base)
# guess the language by comparing the frequency distributions of tokens
print('char bigram: guess for english text is', guess_language_char_bigram(lang_model, text1))
print('char bigram: guess for french text is', guess_language_char_bigram(lang_model, text2))
print('char bigram: guess for german text is', guess_language_char_bigram(lang_model, text3))
print()

lang_model = build_language_models_token_bigram(languages, lang_base)
# guess the language by comparing the frequency distributions of tokens
print('token bigram: guess for english text is', guess_language_token_bigram(lang_model, text1))
print('token bigram: guess for french text is', guess_language_token_bigram(lang_model, text2))
print('token bigram: guess for german text is', guess_language_token_bigram(lang_model, text3))
