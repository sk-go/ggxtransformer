import numpy as np

from chroma import korpus

WOERTER = ["haus", "baum", "katze", "hund", "stein", "fluss", "berg", "licht", "farbe", "zeit", "art", "folge"]


def synthetische_texte(n, seed):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WOERTER, size=12)) + "." for _ in range(n)]


def baue(tmp_path):
    splits = {"train": synthetische_texte(400, 0), "val": synthetische_texte(50, 1), "test": synthetische_texte(50, 2)}
    korpus.baue_korpus(tmp_path, splits=splits, vocab_size=400, quelle="synthetisch")


def test_baue_korpus(tmp_path):
    baue(tmp_path)
    meta = korpus.lade_meta(tmp_path)
    assert meta["vocab_size"] <= 400 and meta["quelle"] == "synthetisch"
    train = korpus.lade_split(tmp_path, "train")
    assert train.dtype == np.uint16 and len(train) == meta["tokens"]["train"]
    assert korpus.lade_frequenz(tmp_path).sum() == len(train)


def test_wortliste(tmp_path):
    baue(tmp_path)
    tok = korpus.lade_tokenizer(tmp_path)
    freq = korpus.lade_frequenz(tmp_path)
    ids = korpus.wortliste(tok.get_vocab(), freq, n=5)
    assert len(ids) == 5
    for i in ids:
        s = tok.decode([i])
        assert s.startswith(" ") and s[1:].isalpha() and len(s) >= 4
    assert freq[ids[0]] >= freq[ids[-1]]


def test_wortliste_hf_vokabular():
    vocab = {"Ġhaus": 0, "Ġab": 1, "Ġ123": 2, "baum": 3, "Ġlicht": 4}
    freq = np.array([5, 9, 9, 9, 7])
    assert korpus.wortliste(vocab, freq, n=10) == [4, 0]
