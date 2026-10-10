"""
chroma/korpus.py
WikiText-103 laden, eigenen BPE-Tokenizer trainieren (die gemeinsame erste Gewohnheit aller
vier Warten), Splits als uint16-memmap ablegen (Spec § 4.3).
"""

import json
from pathlib import Path

import numpy as np

EOT = "<|endoftext|>"
WIKITEXT = "Salesforce/wikitext wikitext-103-raw-v1"


def _kdir(ordner):
    return Path(ordner) / "korpus"


def lade_wikitext():
    """Absätze ohne Überschriften (Zeilen, die mit ' =' beginnen) und ohne Leerzeilen."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    namen = {"train": "train", "val": "validation", "test": "test"}
    return {
        k: [t for t in ds[v]["text"] if t.strip() and not t.lstrip().startswith("=")]
        for k, v in namen.items()
    }


def lade_tokenizer(ordner):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(_kdir(ordner) / "tokenizer.json"))


def _kodiere(tok, texte, eot_id, stueck=10000):
    teile = []
    for i in range(0, len(texte), stueck):
        ids = []
        for enc in tok.encode_batch(texte[i:i + stueck]):
            ids.extend(enc.ids)
            ids.append(eot_id)
        teile.append(np.asarray(ids, dtype=np.uint16))
    return np.concatenate(teile) if teile else np.zeros(0, np.uint16)


def baue_korpus(ordner, splits=None, vocab_size=8192, quelle=WIKITEXT):
    from tokenizers import ByteLevelBPETokenizer

    if vocab_size > 65535:
        raise ValueError("uint16 erlaubt höchstens 65535 Tokens")
    kdir = _kdir(ordner)
    kdir.mkdir(parents=True, exist_ok=True)
    splits = splits if splits is not None else lade_wikitext()
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(splits["train"], vocab_size=vocab_size, min_frequency=2, special_tokens=[EOT])
    bpe.save(str(kdir / "tokenizer.json"))
    tok = lade_tokenizer(ordner)
    eot_id = tok.token_to_id(EOT)
    anzahl = {}
    for name, texte in splits.items():
        arr = _kodiere(tok, texte, eot_id)
        arr.tofile(kdir / f"{name}.bin")
        anzahl[name] = int(len(arr))
    V = tok.get_vocab_size()
    freq = np.bincount(np.fromfile(kdir / "train.bin", dtype=np.uint16), minlength=V).astype(np.int64)
    np.save(kdir / "frequenz.npy", freq)
    meta = {"vocab_size": V, "tokens": anzahl, "quelle": quelle}
    (kdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def lade_meta(ordner):
    return json.loads((_kdir(ordner) / "meta.json").read_text(encoding="utf-8"))


def lade_split(ordner, name):
    return np.memmap(_kdir(ordner) / f"{name}.bin", dtype=np.uint16, mode="r")


def lade_frequenz(ordner):
    return np.load(_kdir(ordner) / "frequenz.npy")


def wortliste(vocab, freq, n=2000, min_len=3):
    """Die n häufigsten Ganzwort-Tokens: 'Ġ'-Präfix (Byte-Level-Leerzeichen), ASCII-Buchstaben, ≥ min_len."""
    kand = []
    for s, i in vocab.items():
        w = s[1:]
        if s.startswith("Ġ") and w.isascii() and w.isalpha() and len(w) >= min_len and i < len(freq):
            kand.append((-int(freq[i]), i))
    kand.sort()
    return [i for _, i in kand[:n]]
