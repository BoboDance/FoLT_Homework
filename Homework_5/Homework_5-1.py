import nltk
import operator

from texttable import Texttable
from nltk.corpus import wordnet as wn
from scipy.stats import rankdata, spearmanr

tuples = [("car", "automobile"), ("gem", "jewel"), ("journey", "voyage"), ("boy", "lad"), ("coast", "shore"),
          ("asylum", "madhouse"), ("magician", "wizard"), ("midday", "noon"), ("furnace", "stove"), ("food", "fruit"),
          ("bird", "cock"), ("bird", "crane"), ("tool", "implement"), ("brother", "monk"), ("lad", "brother"),
          ("crane", "implement"), ("journey", "car"), ("monk", "oracle"), ("cemetery", "woodland"), ("food", "rooster"),
          ("coast", "hill"), ("forest", "graveyard"), ("shore", "woodland"), ("monk", "slave"), ("coast", "forest"),
          ("lad", "wizard"), ("chord", "smile"), ("glass", "magician"), ("rooster", "voyage"), ("noon", "string")]


def compare_pairs(tuple_list):
    tuple_scores = nltk.defaultdict(float)

    for pair in tuple_list:
        # get synsets for the given pair
        synset_first = wn.synsets(pair[0])
        synset_second = wn.synsets(pair[1])

        # keep best similarity score for each pair
        best = 0.0

        # iterate over all possible combinations of synsets
        for el1 in synset_first:
            for el2 in synset_second:
                sim = el1.path_similarity(el2)
                # sim returned some value which is not None
                # check if the new synset pair has a better score and replace current best value
                if sim is not None and best < sim:
                    best = sim
        # add best score for each pair to result dict
        tuple_scores[pair] = best

    # print out results for (a)
    print("Synset Pairs Ranked by Similarity Score")
    print("{:<15} {:<15} {:<10}".format('Syn1', 'Syn2', 'Score'))
    print("-" * 55)
    for elem in sorted(tuple_scores.items(), key=operator.itemgetter(1), reverse=True):
        syn1, syn2 = elem[0]
        print("{:<15} {:<15} {:<10}".format(syn1, syn2, elem[1]))
    print()

    return tuple_scores


def get_ranking(tuple_scores, method="average"):
    """
    From scipy.stats.rankdata:
    ‘average’: The average of the ranks that would have been assigned to all the tied values is assigned to each value.
    ‘min’: The minimum of the ranks that would have been assigned to all the tied values is assigned to each value.
        (This is also referred to as “competition” ranking.)
    ‘max’: The maximum of the ranks that would have been assigned to all the tied values is assigned to each value.
    ‘ordinal’: All values are given a distinct rank, corresponding to the order that the values occur in a.
    ‘dense’: Like ‘min’, but the rank of the next highest element is assigned the rank immediately after those assigned
        to the tied elements.

    :param tuple_scores: dict in form of dict(synset_tuple, similarity_score)
    :param method: scipy.stats.rankdata method param
    :return: dict(synset_tuple, rank)
    """

    # get all scores and rank them
    # rankdata ranks from lowest to highest, therefore subtract the double amount to reverse score into negative
    # and get rank high too low
    # zip with synset pairs
    return dict(zip([k for k, _ in tuple_scores.items()],
                    rankdata([v - v * 2 for v in tuple_scores.values()], method=method)))


def calc_spearman_correlation(ranked_tuples, tuple_dict):

    """
    :param ranked_tuples: dict in form of dict(synset_tuple, rank)
    :param tuple_dict: original hierarchy in form of dict(synset_tuple, rank)
    :return: Spearman Correlation coefficient rho
    """

    # calc difference between ranks
    rank_difference = 0
    for key in ranked_tuples.keys():
        rank_difference += pow(ranked_tuples[key] - tuple_dict[key], 2)

    return 1 - ((6 * rank_difference) / (len(tuples) * (pow(len(tuples), 2) - 1)))


def print_and_calc():
    original_tuple_dict = dict(zip(tuples, range(1, len(tuples) + 1)))
    tuple_scores = compare_pairs(tuples)

    print("Synset Pairs Rank Comparison")
    print("{:<15} {:<15} {:<10} {:<8}".format('Syn1', 'Syn2', 'My (avg)', 'Original'))
    print("-" * 55)
    for k, v in get_ranking(tuple_scores).items():
        syn1, syn2 = k
        v2 = original_tuple_dict[k]
        print("{:<15} {:<15} {:<14} {:<8}".format(syn1, syn2, v, v2))

    print()
    # my stuff
    print("My Spearman Correlation Calculation")
    t = Texttable()
    t.add_rows([['Ranking Type', 'Correlation'],
                ['average', calc_spearman_correlation(get_ranking(tuple_scores), original_tuple_dict)],
                ['min', calc_spearman_correlation(get_ranking(tuple_scores, "min"), original_tuple_dict)],
                ['max', calc_spearman_correlation(get_ranking(tuple_scores, "max"), original_tuple_dict)],
                ['ordinal', calc_spearman_correlation(get_ranking(tuple_scores, "ordinal"), original_tuple_dict)],
                ['dense', calc_spearman_correlation(get_ranking(tuple_scores, "dense"), original_tuple_dict)]
                ])
    print(t.draw())

    print()

    # Scipy
    print("Scipy Spearman Correlation Calculation")
    t = Texttable()
    t.add_rows([['Ranking Type', 'Correlation'],
                ['average', spearmanr(a=list(get_ranking(tuple_scores).values()),
                                      b=list(original_tuple_dict.values()))[0]],
                ['min', spearmanr(a=list(get_ranking(tuple_scores, "min").values()),
                                  b=list(original_tuple_dict.values()))[0]],
                ['max', spearmanr(a=list(get_ranking(tuple_scores, "max").values()),
                                  b=list(original_tuple_dict.values()))[0]],
                ['ordinal', spearmanr(a=list(get_ranking(tuple_scores, "ordinal").values()),
                                      b=list(original_tuple_dict.values()))[0]],
                ['dense', spearmanr(a=list(get_ranking(tuple_scores, "dense").values()),
                                    b=list(original_tuple_dict.values()))[0]]
                ])
    print(t.draw())


print_and_calc()
# On average the result has a correlation of .7, this means both models are correlate decently.
# None of the different ranking styles changed anything, besides the choice for dense, when using the manual
# implementation of Spearman's Correlation coefficient.
# Currently I am not 100% sure what causes this significant deviation.
# I assume the cause can be found in the nature of the ordinal ranking approach.
# The total number of ranks is reduced, means the difference may result from the fact that just adding up the ranks
# results in a smaller value than for the original pairs.
# Scipy probably takes this into account and normalizes the values.
# Further, the ordinal ranking seems to work best in this case for scipy and the manual implementation.
# This may happen, because both synset collections have the same order, therefore the first element with the same
# rank in the similarity score list is ranked higher, which corresponds to the higher ranking in the original
# collection.
# The comparison even shows that having rank duplicates (avg, min, max, dense) does not influence the coefficient
# significantly, however the correlation is still maxed at ordinal ranking with no duplicates.
# In general the goal is to score > .8 or < -.8, however, in practical problems this is often not realistic and
# correlations with .7 are highly welcome.
# To put it briefly, there is definitely a significant correlation present, albeit not a perfect one.
