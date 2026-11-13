import pickle
from typing import Iterable, Iterator
import regex as re

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        if special_tokens:
            for special_token in special_tokens:
                if special_token.encode("utf-8") not in vocab.values():
                    self.vocab[self.vocab_size] = special_token.encode('utf-8')
                    self.vocab_size += 1
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_rank = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
    
    @classmethod
    def from_files(cls, vocabpath, special_tokens=None):
        with open(vocabpath, "rb") as f:
            vocab, merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        special_pattern = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        parts = re.split(f"({special_pattern})", text) if self.special_tokens else [text]
        
        results = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                results.append(self.reverse_vocab[part.encode("utf-8")])
                continue
            for match in re.finditer(PAT, part):
                pre_token = match.group(0)
                sequence = [bytes([value]) for value in pre_token.encode("utf-8")]

                while len(sequence) > 1:
                    best_pair = min(zip(sequence, sequence[1:]), key=lambda pair: self.merge_rank.get(pair, float('inf')))
                    if best_pair not in self.merge_rank:
                        break
                    new_sequence = []
                    i = 0
                    length = len(sequence)
                    while i < length:
                        if i + 1 < length and best_pair == (sequence[i], sequence[i + 1]):
                            new_sequence.append(sequence[i] + sequence[i + 1])
                            i += 2
                        else:
                            new_sequence.append(sequence[i])
                            i += 1
                    sequence = new_sequence
                results.extend(self.reverse_vocab[token] for token in sequence)

        return results

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        encode_text = b"".join(self.vocab[id] for id in ids)
        text = encode_text.decode("utf-8", errors='replace')
        return text