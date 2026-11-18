import json
from pathlib import Path

class CharTokenizer:
    def __init__(self, text=None, vocab=None):
        if vocab is None:
            if text is None:
                raise ValueError("Передай text или vocab")
            self.stoi = {ch: i for i, ch in enumerate(sorted(set(text)))}
        else:
            self.stoi = {ch: int(i) for ch, i in vocab.items()}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        missing = set(text) - set(self.stoi)
        if missing:
            raise ValueError(f"Неизвестные символы: {sorted(missing)!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path):
        Path(path).write_text(json.dumps(self.stoi, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(vocab=json.loads(Path(path).read_text(encoding="utf-8")))
