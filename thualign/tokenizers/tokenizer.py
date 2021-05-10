import abc
import json
import base64
import collections

from typing import List, NoReturn


def _load_vocab(vocab_file) -> collections.OrderedDict:
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "rb") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip(b"\n")
        vocab[token] = index
    return vocab


class Tokenizer(object):

    def __init__(self, name: str):
        self._name = name

    @abc.abstractmethod
    def __repr__(self) -> NoReturn:
        raise NotImplementedError("Tokenizer.__repr__ not implemented.")

    @property
    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    def encode(self, inp: bytes) -> NoReturn:
        raise NotImplementedError("Tokenizer.encode not implemented.")

    @abc.abstractmethod
    def decode(self, inp: List[bytes]) -> NoReturn:
        raise NotImplementedError("Tokenizer.decode not implemented.")
    

class WhiteSpaceTokenizer(Tokenizer):

    def __init__(self):
        super(WhiteSpaceTokenizer, self).__init__("WhiteSpaceTokenizer")

    def __repr__(self) -> str:
        return "WhiteSpaceTokenizer()"

    def encode(self, inp: bytes) -> List[bytes]:
        return inp.strip().split()

    def decode(self, inp: List[bytes]) -> bytes:
        return b" ".join(inp)
    
class WordPieceTokenizer(Tokenizer):

    def __init__(self, vocab, unk=b"[UNK]", max_chars_per_word=200):
        super(WordPieceTokenizer, self).__init__("WordPieceTokenizer")
        self.vocab = vocab if isinstance(vocab, dict) else _load_vocab(vocab)
        self.unk = unk
        self.max_chars_per_word = max_chars_per_word

    def __repr__(self) -> str:
        return "WordPieceTokenizer(vocab_size=%d, unk=%s, " \
               "max_chars_per_word=%d)" % (len(self.vocab), self.unk,
                                           self.max_chars_per_word)

    def encode(self, inp: bytes) -> List[bytes]:
        tokens = inp.strip().split()
        output_tokens = []
        for token in tokens:
            chars = list(token)
            if len(chars) > self.max_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []

            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = bytes(chars[start:end])
                    if start > 0:
                        substr = b"##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


    def decode(self, inp: List[bytes]) -> bytes:
        return b' '.join(inp).replace(b' ##', b'')