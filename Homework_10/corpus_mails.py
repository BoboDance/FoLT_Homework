from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.corpus.util import LazyCorpusLoader

mails = LazyCorpusLoader(
    'mails', CategorizedPlaintextCorpusReader, r'(?!\.).*\.txt', cat_pattern=r'(spam|nospam)/.*')
