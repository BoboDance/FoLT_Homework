from nltk.corpus.reader import CategorizedPlaintextCorpusReader, PlaintextCorpusReader
from nltk.corpus.util import LazyCorpusLoader

reviews_train = LazyCorpusLoader(
    'aclImdb/train', CategorizedPlaintextCorpusReader, r'(?!\.).*\.txt', cat_pattern=r'(neg|pos)/.*')

reviews_test = LazyCorpusLoader(
    'aclImdb/test', PlaintextCorpusReader, r'(?!\.).*\.txt')
