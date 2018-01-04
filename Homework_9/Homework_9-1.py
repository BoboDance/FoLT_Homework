import random

import nltk
from nltk.corpus import stopwords, movie_reviews

# short heads up: This code is messy, I am sorry for that :(

# uhhhh, ugly global variable
stop = stopwords.words('english')

# definitely do not read this one
badWords = [
    '2g1c',
    '2 girls 1 cup',
    'acrotomophilia',
    'anal',
    'anilingus',
    'anus',
    'arsehole',
    'ass',
    'asshole',
    'assmunch',
    'auto erotic',
    'autoerotic',
    'babeland',
    'baby batter',
    'ball gag',
    'ball gravy',
    'ball kicking',
    'ball licking',
    'ball sack',
    'ball sucking',
    'bangbros',
    'bareback',
    'barely legal',
    'barenaked',
    'bastardo',
    'bastinado',
    'bbw',
    'bdsm',
    'beaver cleaver',
    'beaver lips',
    'bestiality',
    'bi curious',
    'big black',
    'big breasts',
    'big knockers',
    'big tits',
    'bimbos',
    'birdlock',
    'bitch',
    'black cock',
    'blonde action',
    'blonde on blonde action',
    'blow j',
    'blow your l',
    'blue waffle',
    'blumpkin',
    'bollocks',
    'bondage',
    'boner',
    'boob',
    'boobs',
    'booty call',
    'brown showers',
    'brunette action',
    'bukkake',
    'bulldyke',
    'bullet vibe',
    'bung hole',
    'bunghole',
    'busty',
    'butt',
    'buttcheeks',
    'butthole',
    'camel toe',
    'camgirl',
    'camslut',
    'camwhore',
    'carpet muncher',
    'carpetmuncher',
    'chocolate rosebuds',
    'circlejerk',
    'cleveland steamer',
    'clit',
    'clitoris',
    'clover clamps',
    'clusterfuck',
    'cock',
    'cocks',
    'coprolagnia',
    'coprophilia',
    'cornhole',
    'cum',
    'cumming',
    'cunnilingus',
    'cunt',
    'darkie',
    'date rape',
    'daterape',
    'deep throat',
    'deepthroat',
    'dick',
    'dildo',
    'dirty pillows',
    'dirty sanchez',
    'dog style',
    'doggie style',
    'doggiestyle',
    'doggy style',
    'doggystyle',
    'dolcett',
    'domination',
    'dominatrix',
    'dommes',
    'donkey punch',
    'double dong',
    'double penetration',
    'dp action',
    'eat my ass',
    'ecchi',
    'ejaculation',
    'erotic',
    'erotism',
    'escort',
    'ethical slut',
    'eunuch',
    'faggot',
    'fecal',
    'felch',
    'fellatio',
    'feltch',
    'female squirting',
    'femdom',
    'figging',
    'fingering',
    'fisting',
    'foot fetish',
    'footjob',
    'frotting',
    'fuck',
    'fucking',
    'fuck buttons',
    'fudge packer',
    'fudgepacker',
    'futanari',
    'g-spot',
    'gang bang',
    'gay sex',
    'genitals',
    'giant cock',
    'girl on',
    'girl on top',
    'girls gone wild',
    'goatcx',
    'goatse',
    'gokkun',
    'golden shower',
    'goo girl',
    'goodpoop',
    'goregasm',
    'grope',
    'group sex',
    'guro',
    'hand job',
    'handjob',
    'hard core',
    'hardcore',
    'hentai',
    'homoerotic',
    'honkey',
    'hooker',
    'hot chick',
    'how to kill',
    'how to murder',
    'huge fat',
    'humping',
    'incest',
    'intercourse',
    'jack off',
    'jail bait',
    'jailbait',
    'jerk off',
    'jigaboo',
    'jiggaboo',
    'jiggerboo',
    'jizz',
    'juggs',
    'kike',
    'kinbaku',
    'kinkster',
    'kinky',
    'knobbing',
    'leather restraint',
    'leather straight jacket',
    'lemon party',
    'lolita',
    'lovemaking',
    'make me come',
    'male squirting',
    'masturbate',
    'menage a trois',
    'milf',
    'missionary position',
    'motherfucker',
    'mound of venus',
    'mr hands',
    'muff diver',
    'muffdiving',
    'nambla',
    'nawashi',
    'negro',
    'neonazi',
    'nig nog',
    'nigga',
    'nigger',
    'nimphomania',
    'nipple',
    'nipples',
    'nsfw images',
    'nude',
    'nudity',
    'nympho',
    'nymphomania',
    'octopussy',
    'omorashi',
    'one cup two girls',
    'one guy one jar',
    'orgasm',
    'orgy',
    'paedophile',
    'panties',
    'panty',
    'pedobear',
    'pedophile',
    'pegging',
    'penis',
    'phone sex',
    'piece of shit',
    'piss pig',
    'pissing',
    'pisspig',
    'playboy',
    'pleasure chest',
    'pole smoker',
    'ponyplay',
    'poof',
    'poop chute',
    'poopchute',
    'porn',
    'porno',
    'pornography',
    'prince albert piercing',
    'pthc',
    'pubes',
    'pussy',
    'queaf',
    'raghead',
    'raging boner',
    'rape',
    'raping',
    'rapist',
    'rectum',
    'reverse cowgirl',
    'rimjob',
    'rimming',
    'rosy palm',
    'rosy palm and her 5 sisters',
    'rusty trombone',
    's&m',
    'sadism',
    'scat',
    'schlong',
    'scissoring',
    'semen',
    'sex',
    'sexo',
    'sexy',
    'shaved beaver',
    'shaved pussy',
    'shemale',
    'shibari',
    'shit',
    'shota',
    'shrimping',
    'slanteye',
    'slut',
    'smut',
    'snatch',
    'snowballing',
    'sodomize',
    'sodomy',
    'spic',
    'spooge',
    'spread legs',
    'strap on',
    'strapon',
    'strappado',
    'strip club',
    'style doggy',
    'suck',
    'sucks',
    'suicide girls',
    'sultry women',
    'swastika',
    'swinger',
    'tainted love',
    'taste my',
    'tea bagging',
    'threesome',
    'throating',
    'tied up',
    'tight white',
    'tit',
    'tits',
    'titties',
    'titty',
    'tongue in a',
    'topless',
    'tosser',
    'towelhead',
    'tranny',
    'tribadism',
    'tub girl',
    'tubgirl',
    'tushy',
    'twat',
    'twink',
    'twinkie',
    'two girls one cup',
    'undressing',
    'upskirt',
    'urethra play',
    'urophilia',
    'vagina',
    'venus mound',
    'vibrator',
    'violet blue',
    'violet wand',
    'vorarephilia',
    'voyeur',
    'vulva',
    'wank',
    'wet dream',
    'wetback',
    'white power',
    'women rapping',
    'wrapping men',
    'wrinkled starfish',
    'xx',
    'xxx',
    'yaoi',
    'yellow showers',
    'yiffy',
    'zoophilia']


def get_features(text, word_dict, bigram_dict):
    res_dict = nltk.defaultdict(int)
    positive_fd = word_dict["pos"]
    negative_fd = word_dict["neg"]
    word_pos = sum([positive_fd[word] for word in text])
    word_neg = sum([-negative_fd[word] for word in text])
    res_dict.update({'pos_score': word_pos, 'neg score': word_neg,
                     'is_positive': True if word_pos - word_neg > 0 else False})

    # words w/o stopwords, delimiters, etc.
    cleaned_words = [token.lower() for token in text if token not in stop and token.isalnum()]

    # just the bigram freqDist
    text_fdist = nltk.FreqDist(nltk.bigrams(cleaned_words))
    res_dict.update(text_fdist)

    # using sentiSynsets in order to determine positive or negative tendency
    # this is terrible: takes waaaaaaaaaaaaaaaaaaaay too long and is useless :(
    # res_pos_swn = sum([senti.pos_score() for word in text for senti in list(swn.senti_synsets(word))])
    # res_neg_swn = sum([senti.neg_score() for word in text for senti in list(swn.senti_synsets(word))])
    # my_dict.update({'senti_score_neg': res_neg_swn, 'senti_score_pos': res_pos_swn})

    bigrams = nltk.bigrams(cleaned_words)
    pos_score = 0
    neg_score = 0
    # get bigram score
    # basically checks if a bigram found in the text is more likely to be found in positive or negative reviews
    # the basis freqDist, which determines the freq of each bigram can be found below
    # Also important: remove stopwords otherwise this is stupid ("of the", "this is", etc.)
    for bigram in bigrams:
        neg_score += bigram_dict['neg'][bigram]
        pos_score += bigram_dict['pos'][bigram]
        # add score for each bigram, does not change anything
        # res_dict[str(bigram) + "_neg"] += bigram_dict['neg'][bigram]
        # res_dict[str(bigram) + "_pos"] += bigram_dict['neg'][bigram]

    res_dict.update({'pos_score_bi': pos_score, 'neg score_bi': neg_score, 'difference': pos_score - neg_score,
                     'is_positive_bi': True if pos_score - neg_score > 0 else False})

    # just some other plain simple features
    # looks like there are no bad words in reviews :(
    res_dict.update(
        {'length': len(text), "contains_bad_word": True if [word for word in text if word in badWords] else False})

    return res_dict


# prepare review data as a list of tuples:
# (list of tokens, category)
# category is positive / negative
review_data = [(movie_reviews.words(fileid), category)
               for category in movie_reviews.categories()
               for fileid in movie_reviews.fileids(category)
               ]

# do not forget to shuffle
random.seed(10)
random.shuffle(review_data)

# train, dev, test
train_data = [(text, category) for (text, category) in review_data[((len(review_data) * 8) // 10):]]
dev_data = [(text, category) for (text, category) in
            review_data[(len(review_data) * 8) // 10:(len(review_data) * 9) // 10]]
test_data = [(text, category) for (text, category) in review_data[:((len(review_data) * 9) // 10)]]
train_dev_data = train_data + dev_data

# get most frequent words in train and dev data
# choosing 1000 or 5000 does only improve the results slightly
threshold = 5000

# this is bad, do not use all words, just use the training data --> overfitting, but good accuracy, though :)
# fd_all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

# alternative with removing stopwords and commas, etc.
fd_all_words = nltk.FreqDist([w.lower() for elem, _ in train_dev_data for w in elem
                              if w not in stop
                              and w.isalnum()])
top_words = [word for (word, _) in fd_all_words.most_common(threshold)]
# review_data_fdist = [(nltk.FreqDist(token.lower() for token in words if token in top_words), category)
#                      for words, category in review_data]

review_data_fdist = [(nltk.FreqDist(token.lower() for token in words
                                    if token in top_words), category)
                     for words, category in review_data]

# get most frequent bigrams
# basically the same as above, but with bigrams
threshold_bigram = 5000

fd_all_bigrams = nltk.FreqDist(nltk.bigrams([w.lower() for elem, _ in train_dev_data for w in elem
                                             if w not in stop
                                             and w.isalnum()]))
top_bigrams = [bigram for (bigram, _) in fd_all_bigrams.most_common(threshold_bigram)]

# get frequency of bigrams in train and dev data
review_data_fdist_bigrams = [(nltk.FreqDist(elem for elem in nltk.bigrams([token.lower()
                                                                           for token in words]) if elem in top_bigrams),
                              category)
                             for words, category in train_dev_data]

# build dicts, which contain freqs of positive and negative annotated bigrams
bigrams_total = {'pos': nltk.FreqDist(), 'neg': nltk.FreqDist()}
for k, v in review_data_fdist_bigrams:
    bigrams_total[v].update(k)

# build dicts, which contain freqs of positive and negative annotated tokens
words_total = {'pos': nltk.FreqDist(), 'neg': nltk.FreqDist()}
for k, v in review_data_fdist:
    words_total[v].update(k)

# relative values
# this is not really necessary, positive and negative are both evaluated on the same dicts (from above)
# so the basis is the same
# n_neg = res['neg'].N()
# print([(word, count/n_neg) for (word, count) in res['neg'].most_common(100)])
# print([(word, count/n_neg) for (word, count) in res['pos'].most_common(100)])

nbc = nltk.NaiveBayesClassifier.train(
    [(get_features(text, words_total, bigrams_total), category) for (text, category) in train_dev_data])
print(nltk.classify.accuracy(nbc, [(get_features(text, words_total, bigrams_total), category) for (text, category) in
                                   test_data]))
for elem in nbc.most_informative_features(20):
    print(elem)
