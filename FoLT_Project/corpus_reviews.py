from nltk.corpus.reader import CategorizedPlaintextCorpusReader, PlaintextCorpusReader
from nltk.corpus.util import LazyCorpusLoader

reviews = LazyCorpusLoader(
    'aclImdb/train', CategorizedPlaintextCorpusReader, r'(?!\.).*\.txt', cat_pattern=r'(neg|pos)/.*')
