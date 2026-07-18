import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """

        vocab = {}
        for n, w in enumerate([self.pad_token, self.unk_token, self.bos_token, self.eos_token]):
            vocab[w] = n
        
        n = len(vocab)
        words = sorted(set(w for t in texts for w in t.lower().split()))
        
        for w in words:
            if w in vocab:
                continue
            else:
                vocab[w] = n
                n = n + 1
        
        self.word_to_id = vocab

        for k, v in vocab.items():
            self.id_to_word[v] = k

        self.vocab_size = len(vocab)
        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """

        return [
            self.word_to_id.get(t, self.word_to_id[self.unk_token])
            for t in text.lower().split()
        ]
        
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """

        return " ".join(self.id_to_word.get(i, self.unk_token)
        for i in ids)
        
