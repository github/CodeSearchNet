## Code adapted from https://github.com/soaxelbrooke/python-bpe/blob/master/bpe/encoder.py
## MIT License (see repository)


""" An encoder which learns byte pair encodings for white-space separated text.  Can tokenize, encode, and decode. """
import typing
from typing import Optional
from collections import Counter

import toolz

try:
    from typing import Dict, Iterable, Callable, List, Any, Iterator
except ImportError:
    pass


DEFAULT_EOW = '__eow'
DEFAULT_SOW = '__sow'
DEFAULT_UNK = '__unk'
DEFAULT_PAD = '__pad'


class BpeVocabulary(typing.Sized):
    """
    Encodes white-space separated text using byte-pair encoding.  See https://arxiv.org/abs/1508.07909 for details.
    """

    def __init__(self, vocab_size: int=8192, pct_bpe: float=0.2,
                 ngram_min: int=2, ngram_max: int=8, required_tokens: Optional[Iterable[str]]=None, strict=True,
                 EOW=DEFAULT_EOW, SOW=DEFAULT_SOW, UNK=DEFAULT_UNK, PAD=DEFAULT_PAD):
        if vocab_size < 1:
            raise ValueError('vocab size must be greater than 0.')

        self.EOW = EOW
        self.SOW = SOW
        self.eow_len = len(EOW)
        self.sow_len = len(SOW)
        self.UNK = UNK
        self.PAD = PAD
        self.required_tokens = list(set(required_tokens or []).union({self.UNK, self.PAD}))
        self.vocab_size = vocab_size
        self.pct_bpe = pct_bpe
        self.word_vocab_size = max([int(vocab_size * (1 - pct_bpe)), len(self.required_tokens or [])])
        self.bpe_vocab_size = vocab_size - self.word_vocab_size
        self.word_vocab = {}  # type: Dict[str, int]
        self.bpe_vocab = {}  # type: Dict[str, int]
        self.inverse_word_vocab = {}  # type: Dict[int, str]
        self.inverse_bpe_vocab = {}  # type: Dict[int, str]
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.strict = strict

    def __len__(self):
        return self.vocab_size

    def byte_pair_counts(self, words: Iterable[str]) -> Iterable[typing.Counter]:
        """ Counts space separated token character pairs:
            [('T h i s </w>', 4}] -> {'Th': 4, 'hi': 4, 'is': 4}
        """
        for token, count in self.count_tokens(words).items():
            bp_counts = Counter()  # type: Counter
            for ngram in token.split(' '):
                bp_counts[ngram] += count
            for ngram_size in range(self.ngram_min, min([self.ngram_max, len(token)]) + 1):
                ngrams = [''.join(ngram) for ngram in toolz.sliding_window(ngram_size, token.split(' '))]

                for ngram in ngrams:
                    bp_counts[''.join(ngram)] += count

            yield bp_counts

    def count_tokens(self, words: Iterable[str]) -> Dict[str, int]:
        """ Count tokens into a BPE vocab """
        token_counts = Counter(words)
        return {' '.join(token): count for token, count in token_counts.items()}

    def learn_word_vocab(self, word_counts: typing.Counter[str]) -> Dict[str, int]:
        """ Build vocab from self.word_vocab_size most common tokens in provided sentences """
        for token in set(self.required_tokens or []):
            word_counts[token] = int(2**31)
        word_counts[self.PAD] = int(2**32)  # Make sure that PAD gets id=0
        sorted_word_counts = sorted(word_counts.items(), key=lambda p: -p[1])
        return {word: idx for idx, (word, count) in enumerate(sorted_word_counts[:self.word_vocab_size])}

    def learn_bpe_vocab(self, words: Iterable[str]) -> Dict[str, int]:
        """ Learns a vocab of byte pair encodings """
        vocab = Counter()  # type: typing.Counter
        for token in {self.SOW, self.EOW}:
            vocab[token] = int(2**63)
        for idx, byte_pair_count in enumerate(self.byte_pair_counts(words)):
            for byte_pair, count in byte_pair_count.items():
                vocab[byte_pair] += count

            if (idx + 1) % 10000 == 0:
                self.trim_vocab(10 * self.bpe_vocab_size, vocab)

        sorted_bpe_counts = sorted(vocab.items(), key=lambda p: -p[1])[:self.bpe_vocab_size]
        return {bp: idx + self.word_vocab_size for idx, (bp, count) in enumerate(sorted_bpe_counts)}

    def fit(self, word_counts: typing.Counter[str]) -> None:
        """ Learn vocab from text. """

        # First, learn word vocab
        self.word_vocab = self.learn_word_vocab(word_counts)

        remaining_words = Counter({word: count for word, count in word_counts.items()
                           if word not in self.word_vocab})
        self.bpe_vocab = self.learn_bpe_vocab(remaining_words.elements())

        self.inverse_word_vocab = {idx: token for token, idx in self.word_vocab.items()}
        self.inverse_bpe_vocab = {idx: token for token, idx in self.bpe_vocab.items()}

    @staticmethod
    def trim_vocab(n: int, vocab: Dict[str, int]) -> None:
        """  Deletes all pairs below 10 * vocab size to prevent memory problems """
        pair_counts = sorted(vocab.items(), key=lambda p: -p[1])
        pairs_to_trim = [pair for pair, count in pair_counts[n:]]
        for pair in pairs_to_trim:
            del vocab[pair]

    def subword_tokenize(self, word: str) -> List[str]:
        """ Tokenizes inside an unknown token using BPE """
        end_idx = min([len(word), self.ngram_max])
        sw_tokens = [self.SOW]
        start_idx = 0

        while start_idx < len(word):
            subword = word[start_idx:end_idx]
            if subword in self.bpe_vocab:
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            elif len(subword) == 1:
                sw_tokens.append(self.UNK)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            else:
                end_idx -= 1

        sw_tokens.append(self.EOW)
        return sw_tokens

    def tokenize(self, word_tokens: List[str]) -> List[str]:
        """ Split a sentence into word and subword tokens """

        tokens = []
        for word_token in word_tokens:
            if word_token in self.word_vocab:
                tokens.append(word_token)
            else:
                tokens.extend(self.subword_tokenize(word_token))

        return tokens

    def transform(self, sentences: Iterable[List[str]], reverse=False, fixed_length=None)-> Iterable[List[str]]:
        """ Turns tokens into vocab idxs """
        direction = -1 if reverse else 1
        for sentence in sentences:
            encoded = []
            tokens = list(self.tokenize(sentence))
            for token in tokens:
                if token in self.word_vocab:
                    encoded.append(self.word_vocab[token])
                elif token in self.bpe_vocab:
                    encoded.append(self.bpe_vocab[token])
                else:
                    encoded.append(self.word_vocab[self.UNK])

            if fixed_length is not None:
                encoded = encoded[:fixed_length]
                while len(encoded) < fixed_length:
                    encoded.append(self.word_vocab[self.PAD])

            yield encoded[::direction]

    def inverse_transform(self, rows: Iterable[List[int]]) -> Iterator[str]:
        """ Turns token indexes back into space-joined text. """
        for row in rows:
            words = []

            rebuilding_word = False
            current_word = ''
            for idx in row:
                if self.inverse_bpe_vocab.get(idx) == self.SOW:
                    if rebuilding_word and self.strict:
                        raise ValueError('Encountered second SOW token before EOW.')
                    rebuilding_word = True

                elif self.inverse_bpe_vocab.get(idx) == self.EOW:
                    if not rebuilding_word and self.strict:
                        raise ValueError('Encountered EOW without matching SOW.')
                    rebuilding_word = False
                    words.append(current_word)
                    current_word = ''

                elif rebuilding_word and (idx in self.inverse_bpe_vocab):
                    current_word += self.inverse_bpe_vocab[idx]

                elif rebuilding_word and (idx in self.inverse_word_vocab):
                    current_word += self.inverse_word_vocab[idx]

                elif idx in self.inverse_word_vocab:
                    words.append(self.inverse_word_vocab[idx])

                elif idx in self.inverse_bpe_vocab:
                    if self.strict:
                        raise ValueError("Found BPE index {} when not rebuilding word!".format(idx))
                    else:
                        words.append(self.inverse_bpe_vocab[idx])

                else:
                    raise ValueError("Got index {} that was not in word or BPE vocabs!".format(idx))

            yield ' '.join(w for w in words if w != '')
