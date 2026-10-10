# Chroma-Gleichartigkeit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ein Python-Paket `chroma/`, das vier Transformer-Arme (A₀, A, Bα, Bβ) mit variabler Basis trainiert, Gleichartigkeit (Ersetzbarkeit, G) und Basispräferenz misst, die Vorstudie an GPT-2/Pythia durchführt und die präregistrierten Entscheidungsregeln auswertet.

**Architecture:** Kleine, einzeln testbare Module: `modell.py` (Architektur mit Arm-Schalter) → `korpus.py` (WikiText-103, eigener BPE) → `training.py` (Loss-Stufen-Checkpoints) → `messung/*` (Maße) → `messen.py` (Pipeline je Checkpoint) → `auswertung.py` (Statistik über Seeds) und `vorstudie.py` (HF-Modelle). `__main__.py` verbindet alles zur CLI mit Selbsttest. Dieser Plan baut nur Code und Tests; keine echten Läufe, denn die brauchen zuerst ihre Präregistrierungen.

**Tech Stack:** Python 3.14 (`venv/`), PyTorch 2.14 (MPS/CPU), transformers 5.17, tokenizers 0.23, datasets 5.0, numpy 2.5, scipy, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-chroma-gleichartigkeit-design.md` (philosophischer Hintergrund: `docs/chroma_ideen.md`).

## Global Constraints

- Python immer über `venv/bin/python` (3.14, torch 2.14, transformers 5.17). Tests: `venv/bin/python -m pytest`.
- Code-Kommentare, Docstrings, Berichte und Fehlermeldungen auf Deutsch (CLAUDE.md).
- Keine Korpustexte committen; `*/korpus/` steht bereits in `.gitignore`. WikiText-103 wird nur heruntergeladen.
- **Commits nur mit den Dateien der Aufgabe:** `git add <dateien> && git commit -m "<msg>" -- <dateien>`. Im Arbeitsbaum liegen fremde, bereits gestagte Änderungen des Autors (Umzüge nach `legacy_experiments/`, `praereg/`, `runs/`); sie dürfen nicht mitcommittet werden. Jede Commit-Nachricht endet mit `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Modell: 6 Schichten, d = 256, 8 Köpfe, Kontext 256, MLP 4d mit GELU, gelernte absolute Positionen, Embedding und Unembedding **entkoppelt**, RMSNorm mit **einem skalaren Gain**.
- Arme: A₀ = SGD mit Momentum, farbe „keine“; A = AdamW, „keine“; Bα = AdamW, „alpha“ (E = softplus(W)); Bβ = AdamW, „beta“ (jeder Block liest RMSNorm(softplus(x))).
- Tokenizer: eigener BPE mit 8192 Tokens. Wortliste: 2000 häufigste Ganzwort-Tokens (Leerzeichen-Präfix, alphabetisch, ≥ 3 Zeichen).
- Ersetzbarkeit: etwa 20 000 Paare, geschichtet nach Kosinus-Dezilen; 20 Testkontexte; Fenster 64; KL über 8 Folgepositionen; symmetrisiert. G = Spearman im mittleren 80-%-Bereich der Kosinus-Ränge.
- Entscheidungsregeln: δ = 0,05; α = 0,05; 5 Seeds je Arm, Seed als Einheit; Manipulationsschwellen vorläufig A₀ < 2, A > 10. Register: stabil in ≥ 3 von 5 Seeds bei Jaccard ≥ 0,5.
- Implementierungswahlen, die der Spec offenlässt (im Plan festgelegt, in PRAEREG_9 zu bestätigen): G_top über 150 Ankerwörter, k = 10; 100 Cluster fürs Register; Bα-Initialisierung W ~ N(−2, 1) mit Skalierung 0,1; Gradient-Clipping 1,0 für alle Arme; kein Weight Decay.
- Nach jeder Codeänderung: `venv/bin/python -m chroma selbsttest` (ab Task 10) und die Testsuite.
- Explorative Maße (E1–E4, G_QK, G_top, Register) in Berichten ausdrücklich als „explorativ“ kennzeichnen.

## Review Focus

1. **A₀ erreicht eine Loss-Stufe nie** → Auswertung bricht nicht ab, sondern weicht auf die niedrigste gemeinsame Stufe aus oder meldet „keine gemeinsame Stufe“ (Test in Task 8).
2. **Ein Wort hat im Testsplit zu wenige Kontexte** → wird übersprungen und im Info-Feld gezählt; G wird über die übrigen Paare berechnet (Test in Task 5).
3. **Neustart in einen fertigen Laufordner** (Nachtlauf versehentlich erneut gestartet) → Training verweigert mit klarer Meldung statt zu überschreiben (Test in Task 3).
4. **Entartete Ähnlichkeit** (konstantes S, G = NaN in einem Seed) → G ist NaN, die Auswertung lässt den Seed weg und berichtet die Anzahl (Tests in Task 5 und Task 8).
5. **Nicht-endlicher Loss** (zu hohe Lernrate im SGD-Sweep) → Training bricht mit `FloatingPointError` ab und schreibt `abbruch.json`, statt still weiterzulaufen (Test in Task 3).

---

### Task 1: Paketgerüst, Abhängigkeiten und Modell mit Invarianztest

**Files:**
- Move: `chroma_transformer/` → `legacy_experiments/chroma_transformer_alt/` (bisher unversioniert; wird mit dem Umzug erstmals committet)
- Modify: `requirements.txt`
- Create: `pytest.ini`, `chroma/__init__.py`, `chroma/arme.py`, `chroma/modell.py`
- Test: `tests/test_modell.py`

**Interfaces:**
- Produces: `ChromaConfig(vocab_size=8192, block_size=256, n_layer=6, n_head=8, n_embd=256, farbe="keine")`; `ChromaModell(cfg)` mit `forward(idx: LongTensor[B,T]) -> Tensor[B,T,V]`, `strom(idx) -> list[Tensor[B,T,d]]` (Eingang jedes Blocks, dann Ausgang des letzten; Länge n_layer+1), `archetyp() -> Tensor[V,d]`; Attribute `wte, wpe, h[i].les_1, h[i].attn.c_attn, h[i].attn.c_proj, h[i].les_2, h[i].mlp.c_fc, h[i].mlp.c_proj, les_f, lm_head`. `ARME: dict[str, dict]` mit Schlüsseln `"A0","A","Ba","Bb"` und Feldern `farbe`, `optimierer`.

- [ ] **Step 1: Alten Chroma-Code verschieben, Abhängigkeiten installieren**

```bash
# chroma_transformer/ ist nicht versioniert, daher mv statt git mv
mv chroma_transformer legacy_experiments/chroma_transformer_alt
rm -rf legacy_experiments/chroma_transformer_alt/__pycache__
printf 'scipy\nscikit-learn\npytest\n' >> requirements.txt
venv/bin/pip install scipy scikit-learn pytest
```

`pytest.ini`:

```ini
[pytest]
testpaths = tests
pythonpath = .
```

`chroma/__init__.py`:

```python
"""Chroma: Gleichartigkeit, Zugleichsein und bevorzugte Basis (Spec 2026-10-10)."""
```

`chroma/arme.py`:

```python
"""Die vier Warten der Hauptstudie (Spec § 4.1)."""

ARME = {
    "A0": {"farbe": "keine", "optimierer": "sgd"},    # keine Vorliebe, weder Anlage noch Gewohnheit
    "A": {"farbe": "keine", "optimierer": "adamw"},   # Vorliebe nur durch Gewohnheit
    "Ba": {"farbe": "alpha", "optimierer": "adamw"},  # Farbe im Archetyp
    "Bb": {"farbe": "beta", "optimierer": "adamw"},   # Farbe im Vollzug
}
```

- [ ] **Step 2: Failing tests schreiben**

`tests/test_modell.py`:

```python
import copy

import pytest
import torch

from chroma.modell import ChromaConfig, ChromaModell


def klein(farbe):
    torch.manual_seed(0)
    cfg = ChromaConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=8, farbe=farbe)
    m = ChromaModell(cfg).double().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, 0.5)  # große Gewichte, damit die Nichtlinearitäten sichtbar wirken
    return m


def transformiere(model, M):
    """Wendet die orthogonale Matrix M auf alle Gewichte an, die den Stream lesen oder schreiben."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.wte.weight.copy_(m.wte.weight @ M)
        m.wpe.weight.copy_(m.wpe.weight @ M)
        for b in m.h:
            for lin in (b.attn.c_attn, b.mlp.c_fc):
                lin.weight.copy_(lin.weight @ M)
            for lin in (b.attn.c_proj, b.mlp.c_proj):
                lin.weight.copy_(M.T @ lin.weight)
                lin.bias.copy_(lin.bias @ M)
        m.lm_head.weight.copy_(m.lm_head.weight @ M)
    return m


def drehung(d, seed=1):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r))


def permutation(d, seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.eye(d, dtype=torch.float64)[torch.randperm(d, generator=g)]


IDX = torch.tensor([[1, 5, 7, 3, 9, 2, 4, 8]])


def test_logits_form_und_entkopplung():
    m = klein("keine")
    assert m(IDX).shape == (1, 8, 50)
    assert m.lm_head.weight.data_ptr() != m.wte.weight.data_ptr()


def test_alpha_archetyp_nichtnegativ():
    assert (klein("alpha").archetyp() >= 0).all()


def test_keine_farbe_ist_drehinvariant():
    m = klein("keine")
    assert torch.allclose(m(IDX), transformiere(m, drehung(8))(IDX), atol=1e-9)


@pytest.mark.parametrize("farbe", ["alpha", "beta"])
def test_farbe_bricht_drehung(farbe):
    m = klein(farbe)
    assert (m(IDX) - transformiere(m, drehung(8))(IDX)).abs().max() > 1e-3


@pytest.mark.parametrize("farbe", ["keine", "alpha", "beta"])
def test_alle_sind_permutationsinvariant(farbe):
    m = klein(farbe)
    assert torch.allclose(m(IDX), transformiere(m, permutation(8))(IDX), atol=1e-9)


def test_strom_laenge():
    assert len(klein("beta").strom(IDX)) == 3  # Eingang Block 0, Eingang Block 1, Ausgang


def test_falsche_farbe():
    with pytest.raises(ValueError):
        ChromaConfig(farbe="rot")


def test_zu_langer_kontext():
    with pytest.raises(ValueError):
        klein("keine")(torch.zeros(1, 17, dtype=torch.long))
```

- [ ] **Step 3: Tests laufen lassen, Fehlschlag prüfen**

Run: `venv/bin/python -m pytest tests/test_modell.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'chroma.modell'`

- [ ] **Step 4: Modell implementieren**

`chroma/modell.py`:

```python
"""
chroma/modell.py
Decoder-only-Transformer, dessen Basis eine Variable ist (Spec § 4.2).

farbe = "keine": architektonisch rotationsinvariant (A₀, A)
farbe = "alpha": nichtnegatives Embedding, Archetyp = 0,1 · softplus(W) (Bα, Farbe im Archetyp)
farbe = "beta":  jede Lesung des Streams geht durch softplus (Bβ, Farbe im Vollzug)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

FARBEN = ("keine", "alpha", "beta")
ALPHA_SKALA = 0.1        # skalare Skalierung ändert die Basiseigenschaft nicht
ALPHA_INIT_MITTEL = -2.0  # sonst lägen alle softplus-Vektoren nahe ln 2 und wären fast gleich


@dataclass
class ChromaConfig:
    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    farbe: str = "keine"

    def __post_init__(self):
        if self.farbe not in FARBEN:
            raise ValueError(f"farbe muss eine von {FARBEN} sein, nicht {self.farbe!r}")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd muss durch n_head teilbar sein")


class SkalarRMSNorm(nn.Module):
    """RMSNorm mit einem einzigen skalaren Gain: Das Maß zeichnet keine Richtung aus."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(()))
        self.eps = eps

    def forward(self, x):
        return self.gain * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Lesung(nn.Module):
    """Wie eine Teilschicht den Stream erblickt; bei farbe='beta' durch die farbige Linse softplus."""

    def __init__(self, farbe):
        super().__init__()
        self.farbig = farbe == "beta"
        self.norm = SkalarRMSNorm()

    def forward(self, x):
        return self.norm(F.softplus(x) if self.farbig else x)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.les_1 = Lesung(cfg.farbe)
        self.attn = Attention(cfg)
        self.les_2 = Lesung(cfg.farbe)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.les_1(x))
        return x + self.mlp(self.les_2(x))


class ChromaModell(nn.Module):
    def __init__(self, cfg: ChromaConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.h = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.les_f = Lesung(cfg.farbe)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # entkoppelt: kein weight tying
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)  # isotrop
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        if cfg.farbe == "alpha":
            nn.init.normal_(self.wte.weight, ALPHA_INIT_MITTEL, 1.0)

    def archetyp(self):
        """Die Embedding-Matrix, wie das Modell sie benutzt (bei alpha nichtnegativ)."""
        W = self.wte.weight
        return ALPHA_SKALA * F.softplus(W) if self.cfg.farbe == "alpha" else W

    def strom(self, idx):
        """Stream-Zustände: Eingang jedes Blocks, dann Ausgang des letzten (Länge n_layer + 1)."""
        T = idx.shape[1]
        if T > self.cfg.block_size:
            raise ValueError(f"Sequenz der Länge {T} überschreitet block_size {self.cfg.block_size}")
        x = F.embedding(idx, self.archetyp()) + self.wpe(torch.arange(T, device=idx.device))
        zustaende = [x]
        for block in self.h:
            x = block(x)
            zustaende.append(x)
        return zustaende

    def forward(self, idx):
        return self.lm_head(self.les_f(self.strom(idx)[-1]))
```

- [ ] **Step 5: Tests laufen lassen, Erfolg prüfen**

Run: `venv/bin/python -m pytest tests/test_modell.py -v`
Expected: 11 passed (die Parametrisierungen mitgezählt)

- [ ] **Step 6: Commit**

```bash
git add pytest.ini requirements.txt chroma/__init__.py chroma/arme.py chroma/modell.py tests/test_modell.py legacy_experiments/chroma_transformer_alt
git commit -m "Chroma: Modell mit Arm-Schalter und Invarianztest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- pytest.ini requirements.txt chroma/__init__.py chroma/arme.py chroma/modell.py tests/test_modell.py legacy_experiments/chroma_transformer_alt
```

---

### Task 2: Korpus und Tokenizer

**Files:**
- Create: `chroma/korpus.py`
- Test: `tests/test_korpus.py`

**Interfaces:**
- Produces: `baue_korpus(ordner, splits=None, vocab_size=8192, quelle=...) -> dict` (schreibt `<ordner>/korpus/{tokenizer.json, train.bin, val.bin, test.bin, frequenz.npy, meta.json}`); `lade_meta(ordner) -> dict` mit `vocab_size`, `tokens`, `quelle`; `lade_split(ordner, name) -> np.memmap[uint16]`; `lade_tokenizer(ordner) -> tokenizers.Tokenizer`; `lade_frequenz(ordner) -> np.ndarray[int64]`; `wortliste(vocab: dict[str,int], freq: np.ndarray, n=2000, min_len=3) -> list[int]` (auch für HF-Vokabulare nutzbar).

- [ ] **Step 1: Failing tests schreiben**

`tests/test_korpus.py`:

```python
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
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `venv/bin/python -m pytest tests/test_korpus.py -v`
Expected: FAIL mit `ImportError: cannot import name 'korpus'`

- [ ] **Step 3: Korpus implementieren**

`chroma/korpus.py`:

```python
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
```

- [ ] **Step 4: Tests laufen lassen, Erfolg prüfen**

Run: `venv/bin/python -m pytest tests/test_korpus.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add chroma/korpus.py tests/test_korpus.py
git commit -m "Chroma: Korpus, eigener BPE-Tokenizer, Wortliste

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- chroma/korpus.py tests/test_korpus.py
```

---

### Task 3: Training mit Loss-Stufen-Checkpoints

**Files:**
- Create: `chroma/training.py`
- Test: `tests/test_training.py`

**Interfaces:**
- Consumes: `ChromaConfig`, `ChromaModell` (Task 1); `ARME` (Task 1); `lade_meta`, `lade_split` (Task 2).
- Produces: `TrainConfig(arm, seed, lr, batch_size=32, block_size=256, n_layer=6, n_head=8, n_embd=256, max_tokens=100_000_000, eval_alle=200, eval_batches=20, loss_stufen=(), momentum=0.9)`; `trainiere(korpus_ordner, lauf_ordner, tc, device="cpu", log=print) -> dict` mit `kurve: list[{schritt,tokens,val_loss}]`, `erreicht: dict[str,int]`; schreibt `stufe_<x.xx>.pt` (je erreichter Stufe), `ende.pt`, `kurve.json`, `fertig.json` bzw. `abbruch.json`; `speichere(model, pfad, extra)`; `lade_modell(pfad, device="cpu") -> ChromaModell` (eval-Modus); `stufen_name(s: float) -> str` (`"4.00"`).

- [ ] **Step 1: Failing tests schreiben**

`tests/test_training.py`:

```python
import json
import math

import numpy as np
import pytest

from chroma.arme import ARME
from chroma.training import TrainConfig, lade_modell, trainiere

KLEIN = dict(batch_size=8, block_size=16, n_layer=1, n_head=2, n_embd=16, eval_alle=10, eval_batches=2)


def mini_korpus(tmp_path, vocab=20, n=20000):
    k = tmp_path / "korpus"
    k.mkdir()
    muster = np.tile(np.arange(vocab, dtype=np.uint16), n // vocab)  # lernbare Zählfolge
    for name in ("train", "val", "test"):
        muster.tofile(k / f"{name}.bin")
    meta = {"vocab_size": vocab, "tokens": {"train": len(muster)}, "quelle": "test"}
    (k / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_lernt(tmp_path):
    k = mini_korpus(tmp_path)
    tc = TrainConfig(arm="A", seed=0, lr=3e-3, max_tokens=8 * 16 * 80, **KLEIN)
    erg = trainiere(k, tmp_path / "lauf", tc, log=lambda *a: None)
    assert erg["kurve"][-1]["val_loss"] < math.log(20) - 1.0
    assert (tmp_path / "lauf" / "fertig.json").exists()
    assert lade_modell(tmp_path / "lauf" / "ende.pt").cfg.farbe == "keine"


def test_stufe_wird_gesichert_und_beendet(tmp_path):
    k = mini_korpus(tmp_path)
    tc = TrainConfig(arm="A", seed=0, lr=3e-3, max_tokens=8 * 16 * 80, loss_stufen=(100.0,), **KLEIN)
    erg = trainiere(k, tmp_path / "lauf", tc, log=lambda *a: None)
    assert (tmp_path / "lauf" / "stufe_100.00.pt").exists()
    assert erg["erreicht"] == {"100.00": 10}
    assert erg["kurve"][-1]["schritt"] == 10


def test_deterministisch(tmp_path):
    k = mini_korpus(tmp_path)
    tc = TrainConfig(arm="Bb", seed=3, lr=3e-3, max_tokens=8 * 16 * 20, **KLEIN)
    a = trainiere(k, tmp_path / "l1", tc, log=lambda *a: None)
    b = trainiere(k, tmp_path / "l2", tc, log=lambda *a: None)
    assert a["kurve"] == b["kurve"]


def test_fertiger_lauf_wird_nicht_ueberschrieben(tmp_path):
    k = mini_korpus(tmp_path)
    tc = TrainConfig(arm="A", seed=0, lr=3e-3, max_tokens=8 * 16 * 10, **KLEIN)
    trainiere(k, tmp_path / "lauf", tc, log=lambda *a: None)
    with pytest.raises(FileExistsError):
        trainiere(k, tmp_path / "lauf", tc, log=lambda *a: None)


def test_nicht_endlicher_loss_bricht_ab(tmp_path):
    k = mini_korpus(tmp_path)
    tc = TrainConfig(arm="A0", seed=0, lr=1e30, max_tokens=8 * 16 * 40, **KLEIN)
    with pytest.raises(FloatingPointError):
        trainiere(k, tmp_path / "lauf", tc, log=lambda *a: None)
    assert (tmp_path / "lauf" / "abbruch.json").exists()


@pytest.mark.parametrize("arm", list(ARME))
def test_alle_arme_laufen(tmp_path, arm):
    k = mini_korpus(tmp_path)
    tc = TrainConfig(arm=arm, seed=0, lr=1e-3, max_tokens=8 * 16 * 2, **KLEIN)
    erg = trainiere(k, tmp_path / arm, tc, log=lambda *a: None)
    assert np.isfinite(erg["kurve"][-1]["val_loss"])
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `venv/bin/python -m pytest tests/test_training.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'chroma.training'`

- [ ] **Step 3: Training implementieren**

`chroma/training.py`:

```python
"""
chroma/training.py
Trainingsschleife der vier Arme. Gesichert wird an festen Loss-Stufen, nicht nach Schrittzahl:
Verglichen werden Warten gleicher Erfahrungssättigung (Spec § 4.4).
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .arme import ARME
from .korpus import lade_meta, lade_split
from .modell import ChromaConfig, ChromaModell


@dataclass
class TrainConfig:
    arm: str
    seed: int
    lr: float
    batch_size: int = 32
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    max_tokens: int = 100_000_000
    eval_alle: int = 200
    eval_batches: int = 20
    loss_stufen: tuple = ()
    momentum: float = 0.9


def stufen_name(s):
    return f"{s:.2f}"


def _batch(daten, rng, b, t, device):
    starts = rng.integers(0, len(daten) - t - 1, size=b)
    x = np.stack([np.asarray(daten[s:s + t], dtype=np.int64) for s in starts])
    y = np.stack([np.asarray(daten[s + 1:s + t + 1], dtype=np.int64) for s in starts])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def _optimierer(model, art, tc):
    if art == "sgd":
        return torch.optim.SGD(model.parameters(), lr=tc.lr, momentum=tc.momentum)
    if art == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=0.0)
    raise ValueError(f"unbekannter Optimierer {art!r}")


@torch.no_grad()
def _val_loss(model, val_batches):
    model.eval()
    werte = [F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item() for x, y in val_batches]
    model.train()
    return float(np.mean(werte))


def speichere(model, pfad, extra=None):
    torch.save({"cfg": asdict(model.cfg), "state": model.state_dict(), **(extra or {})}, pfad)


def lade_modell(pfad, device="cpu"):
    daten = torch.load(pfad, map_location=device, weights_only=False)
    model = ChromaModell(ChromaConfig(**daten["cfg"]))
    model.load_state_dict(daten["state"])
    return model.to(device).eval()


def trainiere(korpus_ordner, lauf_ordner, tc, device="cpu", log=print):
    lauf = Path(lauf_ordner)
    if (lauf / "fertig.json").exists():
        raise FileExistsError(f"{lauf} ist bereits fertig trainiert; anderen Laufordner wählen")
    lauf.mkdir(parents=True, exist_ok=True)
    if tc.arm not in ARME:
        raise ValueError(f"unbekannter Arm {tc.arm!r}, erlaubt: {list(ARME)}")
    torch.manual_seed(tc.seed)
    meta = lade_meta(korpus_ordner)
    cfg = ChromaConfig(vocab_size=meta["vocab_size"], block_size=tc.block_size, n_layer=tc.n_layer,
                       n_head=tc.n_head, n_embd=tc.n_embd, farbe=ARME[tc.arm]["farbe"])
    model = ChromaModell(cfg).to(device).train()
    opt = _optimierer(model, ARME[tc.arm]["optimierer"], tc)
    train, val = lade_split(korpus_ordner, "train"), lade_split(korpus_ordner, "val")
    rng = np.random.default_rng(tc.seed)  # gleiche Datenreihenfolge über die Arme
    val_rng = np.random.default_rng(0)
    val_batches = [_batch(val, val_rng, tc.batch_size, tc.block_size, device) for _ in range(tc.eval_batches)]
    stufen = sorted(tc.loss_stufen, reverse=True)
    erreicht, kurve = {}, []
    pro_schritt = tc.batch_size * tc.block_size
    max_schritte = max(1, tc.max_tokens // pro_schritt)
    t0 = time.time()
    for schritt in range(1, max_schritte + 1):
        x, y = _batch(train, rng, tc.batch_size, tc.block_size, device)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            (lauf / "abbruch.json").write_text(json.dumps({"schritt": schritt, "train": asdict(tc)}, indent=2))
            raise FloatingPointError(f"Loss nicht endlich bei Schritt {schritt} (lr={tc.lr})")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if schritt % tc.eval_alle == 0 or schritt == max_schritte:
            v = _val_loss(model, val_batches)
            kurve.append({"schritt": schritt, "tokens": schritt * pro_schritt, "val_loss": v})
            log(f"[{tc.arm} s{tc.seed}] Schritt {schritt}: val_loss {v:.4f}")
            for s in stufen:
                if stufen_name(s) not in erreicht and v <= s:
                    erreicht[stufen_name(s)] = schritt
                    speichere(model, lauf / f"stufe_{stufen_name(s)}.pt", {"schritt": schritt, "val_loss": v})
            if stufen and len(erreicht) == len(stufen):
                break
    speichere(model, lauf / "ende.pt", {"schritt": schritt})
    (lauf / "kurve.json").write_text(json.dumps(kurve, indent=2))
    erg = {"kurve": kurve, "erreicht": erreicht}
    fertig = {"train": asdict(tc), "erreicht": erreicht, "dauer_s": round(time.time() - t0, 1)}
    (lauf / "fertig.json").write_text(json.dumps(fertig, indent=2))
    return erg
```

- [ ] **Step 4: Tests laufen lassen, Erfolg prüfen**

Run: `venv/bin/python -m pytest tests/test_training.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add chroma/training.py tests/test_training.py
git commit -m "Chroma: Training mit Loss-Stufen-Checkpoints

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- chroma/training.py tests/test_training.py
```

---

### Task 4: Basismaße (Kurtosis, Participation Ratio, ICA-Drehung)

**Files:**
- Create: `chroma/messung/__init__.py`, `chroma/messung/basis.py`
- Test: `tests/test_basis.py`

**Interfaces:**
- Produces: `zeilen_kurtosis(X: (n,d)) -> np.ndarray[n]` (Exzess-Kurtosis über die Dimensionen je Zeile); `exzess_kurtosis(X) -> float` (Mittel davon); `zufallsdrehung(d, rng) -> np.ndarray[d,d]`; `kurtosis_kontrolle(X, n_drehungen, rng) -> list[float]`; `participation_ratio(V: (n,d)) -> np.ndarray[n]`; `ica_drehung(X: (n,d), seed=0) -> np.ndarray[d,d]` (orthogonal; gedrehte Daten `X @ M`; verlangt n ≥ d).

- [ ] **Step 1: Failing tests schreiben**

`tests/test_basis.py`:

```python
import numpy as np

from chroma.messung import basis


def test_kurtosis_gauss_nahe_null():
    X = np.random.default_rng(0).standard_normal((2000, 256))
    assert abs(basis.exzess_kurtosis(X)) < 0.1


def test_ausreisserdimension_und_drehungskontrolle():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 64))
    X[:, 0] += 30.0  # eine bevorzugte Achse
    assert basis.exzess_kurtosis(X) > 10
    assert max(basis.kurtosis_kontrolle(X, 5, rng)) < 2


def test_participation_ratio():
    assert np.allclose(basis.participation_ratio(np.eye(5)), 1.0)
    assert np.allclose(basis.participation_ratio(np.ones((2, 7))), 7.0)


def test_ica_findet_versteckte_spaerliche_basis():
    rng = np.random.default_rng(2)
    S = rng.standard_normal((4000, 6)) * (rng.random((4000, 6)) < 0.1)
    Q = basis.zufallsdrehung(6, rng)
    X = S @ Q
    M = basis.ica_drehung(X)
    assert np.allclose(M @ M.T, np.eye(6), atol=1e-8)
    pr_vorher = basis.participation_ratio(X).mean()
    pr_nachher = basis.participation_ratio(X @ M).mean()
    assert pr_nachher < pr_vorher - 0.3
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `venv/bin/python -m pytest tests/test_basis.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'chroma.messung'`

- [ ] **Step 3: Basismaße implementieren**

`chroma/messung/__init__.py`:

```python
"""Messinstrumente der Studie (Spec § 4.5)."""
```

`chroma/messung/basis.py`:

```python
"""
chroma/messung/basis.py
Wie stark eine Warte ihre Achsen unterscheidet: Exzess-Kurtosis über die Dimensionen
(homogene gegen qualitative Mannigfaltigkeit), Participation Ratio (auf wie vielen Achsen
ein Begriff liegt) und die ICA-Drehung als beste Basis, die der Betrachter nachträglich wählt.
"""

import numpy as np


def zeilen_kurtosis(X):
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=1, keepdims=True)
    m2 = (Xc ** 2).mean(axis=1)
    m4 = (Xc ** 4).mean(axis=1)
    return m4 / np.maximum(m2 ** 2, 1e-30) - 3.0


def exzess_kurtosis(X):
    return float(zeilen_kurtosis(X).mean())


def zufallsdrehung(d, rng):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def kurtosis_kontrolle(X, n_drehungen, rng):
    """Dieselben Aktivierungen nach zufälligen Drehungen: Was hier bleibt, ist nicht der Achse eigen."""
    X = np.asarray(X, dtype=np.float64)
    return [exzess_kurtosis(X @ zufallsdrehung(X.shape[1], rng)) for _ in range(n_drehungen)]


def participation_ratio(V):
    V = np.asarray(V, dtype=np.float64)
    s2 = (V ** 2).sum(axis=1)
    s4 = (V ** 4).sum(axis=1)
    return s2 ** 2 / np.maximum(s4, 1e-30)


def ica_drehung(X, seed=0):
    """Orthogonale Matrix M, die FastICA wählt: Hauptachsen U, dann ICA-Drehung im weißen Raum,
    auf die nächste orthogonale Matrix projiziert. Gedrehte Daten: X @ M."""
    from sklearn.decomposition import FastICA

    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if n < d:
        raise ValueError(f"ICA braucht mindestens so viele Zeilen ({n}) wie Dimensionen ({d})")
    Xc = X - X.mean(axis=0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    U = Vt.T
    Z = (Xc @ U) / np.maximum(s / np.sqrt(n), 1e-12)
    ica = FastICA(whiten=False, random_state=seed, max_iter=2000, tol=1e-4)
    ica.fit(Z)
    u, _, vt = np.linalg.svd(ica.components_.T)
    return U @ (u @ vt)
```

- [ ] **Step 4: Tests laufen lassen, Erfolg prüfen**

Run: `venv/bin/python -m pytest tests/test_basis.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add chroma/messung/__init__.py chroma/messung/basis.py tests/test_basis.py
git commit -m "Chroma: Basismaße (Kurtosis, Participation Ratio, ICA)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- chroma/messung/__init__.py chroma/messung/basis.py tests/test_basis.py
```

---

### Task 5: Ersetzbarkeit und Gleichartigkeitsindex G

**Files:**
- Create: `chroma/messung/ersetzbarkeit.py`
- Test: `tests/test_ersetzbarkeit.py`

**Interfaces:**
- Consumes: ein Modell mit `forward(idx) -> logits[B,T,V]` (Task 1).
- Produces: `kosinus_matrix(E: (n,d)) -> (n,n)`; `ziehe_paare(cos: (n,n), n_paare, rng, n_schichten=10) -> np.ndarray[m,2]` (i < j, Indizes in die Wortliste); `finde_kontexte(ids, wort, n, fenster, nach, rng) -> np.ndarray` (Startindizes); `gerichtete_divergenz(model, ids, gerichtet: dict[int,list[int]], n_kontexte=20, min_kontexte=5, fenster=64, nach=8, device="cpu", seed=0, batch=256) -> (D: dict[(x,y)]->float, anzahl: dict[x]->int)`; `ersetzbarkeit(model, ids, woerter, paare, **kw) -> (S: np.ndarray[m], info: dict)` mit `info["woerter_ohne_kontexte"]`, `info["paare_gueltig"]`, `info["paare_gesamt"]`; `ersetzbarkeit_matrix(model, ids, woerter, **kw) -> (S: (n,n), info)`; `gleichartigkeit_G(cos_werte, S, mitte=0.8) -> float` (NaN bei Entartung).

- [ ] **Step 1: Failing tests schreiben**

`tests/test_ersetzbarkeit.py`:

```python
import numpy as np
import torch

from chroma.messung import ersetzbarkeit as ers
from chroma.modell import ChromaConfig, ChromaModell

KW = dict(n_kontexte=5, min_kontexte=3, fenster=12, nach=3, device="cpu", seed=0, batch=64)


def mini():
    torch.manual_seed(0)
    model = ChromaModell(ChromaConfig(vocab_size=20, block_size=16, n_layer=1, n_head=2, n_embd=16)).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.5)  # deutlich verschiedene Verteilungen, damit KL > 0 sicher messbar ist
    ids = np.random.default_rng(0).integers(0, 19, size=3000).astype(np.uint16)  # Token 19 kommt nie vor
    return model, ids


def test_G_monoton_ist_eins():
    c = np.random.default_rng(0).random(500)
    assert abs(ers.gleichartigkeit_G(c, c ** 3) - 1.0) < 1e-12


def test_G_entartet_ist_nan():
    c = np.random.default_rng(0).random(50)
    assert np.isnan(ers.gleichartigkeit_G(c, np.ones(50)))


def test_G_ignoriert_nan():
    c = np.random.default_rng(0).random(200)
    S = c.copy()
    S[:20] = np.nan
    assert abs(ers.gleichartigkeit_G(c, S) - 1.0) < 1e-12


def test_ziehe_paare():
    rng = np.random.default_rng(0)
    E = rng.standard_normal((40, 8))
    paare = ers.ziehe_paare(ers.kosinus_matrix(E), 100, rng)
    assert paare.shape == (100, 2)
    assert (paare[:, 0] < paare[:, 1]).all()
    assert len({tuple(p) for p in paare}) == 100


def test_divergenz_null_bei_selbstersetzung_und_fehlende_kontexte():
    model, ids = mini()
    D, anzahl = ers.gerichtete_divergenz(model, ids, {3: [3, 4], 19: [3]}, **KW)
    assert abs(D[(3, 3)]) < 1e-6
    assert D[(3, 4)] > 0
    assert anzahl[19] == 0 and (19, 3) not in D


def test_ersetzbarkeit_meldet_woerter_ohne_kontexte():
    model, ids = mini()
    woerter = [3, 4, 19]
    S, info = ers.ersetzbarkeit(model, ids, woerter, np.array([[0, 1], [0, 2]]), **KW)
    assert np.isfinite(S[0]) and np.isnan(S[1])
    assert info["woerter_ohne_kontexte"] == [19]
    assert info["paare_gueltig"] == 1 and info["paare_gesamt"] == 2


def test_ersetzbarkeit_matrix_symmetrisch():
    model, ids = mini()
    M, _ = ers.ersetzbarkeit_matrix(model, ids, [3, 4, 5], **KW)
    assert M.shape == (3, 3)
    assert np.allclose(M, M.T, equal_nan=True)
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `venv/bin/python -m pytest tests/test_ersetzbarkeit.py -v`
Expected: FAIL mit `ImportError: cannot import name 'ersetzbarkeit'`

- [ ] **Step 3: Ersetzbarkeit implementieren**

`chroma/messung/ersetzbarkeit.py`:

```python
"""
chroma/messung/ersetzbarkeit.py
Kernmaß (b): Gleichartig ist, was die Warte als gleich behandelt (Gleichsetzung des Ungleichen).
Man tauscht x gegen y und misst die Ent-Täuschung der Erwartung über die folgenden Positionen
(KL-Divergenz). G = Übereinstimmung von Anlage (Kosinus im Archetyp) und Vollzug (Ersetzbarkeit)
im mittleren Bereich, also in den Graden zwischen Gleichheit und Verschiedenheit (Spec § 4.5).
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


def kosinus_matrix(E):
    E = np.asarray(E, dtype=np.float64)
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return En @ En.T


def ziehe_paare(cos, n_paare, rng, n_schichten=10):
    """Paare (i < j), geschichtet nach den Kosinus-Quantilen dieses Modells."""
    iu = np.triu_indices(len(cos), 1)
    werte = cos[iu]
    grenzen = np.quantile(werte, np.linspace(0, 1, n_schichten + 1))
    schicht = np.clip(np.searchsorted(grenzen, werte, side="right") - 1, 0, n_schichten - 1)
    pro = n_paare // n_schichten
    wahl = []
    for s in range(n_schichten):
        idx = np.flatnonzero(schicht == s)
        wahl.append(rng.choice(idx, size=min(pro, len(idx)), replace=False))
    w = np.concatenate(wahl)
    return np.stack([iu[0][w], iu[1][w]], axis=1)


def finde_kontexte(ids, wort, n, fenster, nach, rng):
    """Startindizes von Fenstern, in denen `wort` an Position fenster − nach − 1 steht."""
    p = fenster - nach - 1
    treffer = np.flatnonzero(np.asarray(ids[p:len(ids) - nach]) == wort) + p
    if len(treffer) > n:
        treffer = np.sort(rng.choice(treffer, size=n, replace=False))
    return treffer - p


@torch.no_grad()
def _logprobs(model, fenster_ids, p, nach, device, batch):
    teile = []
    for i in range(0, len(fenster_ids), batch):
        x = torch.as_tensor(fenster_ids[i:i + batch], dtype=torch.long, device=device)
        teile.append(F.log_softmax(model(x)[:, p:p + nach].float(), dim=-1).cpu())
    return torch.cat(teile)


def gerichtete_divergenz(model, ids, gerichtet, n_kontexte=20, min_kontexte=5, fenster=64, nach=8,
                         device="cpu", seed=0, batch=256):
    """D[(x, y)] = mittlere KL(p_x ‖ p_{x→y}) über Kontexte und Folgepositionen.
    Quellen mit weniger als min_kontexte Kontexten werden übersprungen (anzahl hält sie fest)."""
    model.eval()
    rng = np.random.default_rng(seed)
    p = fenster - nach - 1
    D, anzahl = {}, {}
    for x in sorted(gerichtet):
        starts = finde_kontexte(ids, x, n_kontexte, fenster, nach, rng)
        anzahl[x] = int(len(starts))
        if len(starts) < min_kontexte:
            continue
        fen = np.stack([np.asarray(ids[s:s + fenster], dtype=np.int64) for s in starts])
        lp_x = _logprobs(model, fen, p, nach, device, batch)
        p_x = lp_x.exp()
        ziele = list(gerichtet[x])
        pro_gruppe = max(1, batch // len(fen))
        for j in range(0, len(ziele), pro_gruppe):
            gruppe = ziele[j:j + pro_gruppe]
            ers = np.repeat(fen[None], len(gruppe), axis=0)
            ers[:, :, p] = np.asarray(gruppe)[:, None]
            lp_y = _logprobs(model, ers.reshape(-1, fenster), p, nach, device, batch)
            lp_y = lp_y.view(len(gruppe), len(fen), nach, -1)
            kl = (p_x[None] * (lp_x[None] - lp_y)).sum(-1).mean(dim=(1, 2))
            for y, v in zip(gruppe, kl.tolist()):
                D[(x, int(y))] = float(v)
    return D, anzahl


def ersetzbarkeit(model, ids, woerter, paare, **kw):
    """S = −(D(x→y) + D(y→x)) / 2; NaN, wenn eine Richtung fehlt."""
    woerter = [int(w) for w in woerter]
    paar_ids = [(woerter[i], woerter[j]) for i, j in paare]
    gerichtet = defaultdict(list)
    for a, b in paar_ids:
        gerichtet[a].append(b)
        gerichtet[b].append(a)
    D, anzahl = gerichtete_divergenz(model, ids, gerichtet, **kw)
    S = np.array([-(D[(a, b)] + D[(b, a)]) / 2 if (a, b) in D and (b, a) in D else np.nan
                  for a, b in paar_ids])
    min_k = kw.get("min_kontexte", 5)
    info = {
        "woerter_ohne_kontexte": sorted(x for x, k in anzahl.items() if k < min_k),
        "paare_gueltig": int(np.isfinite(S).sum()),
        "paare_gesamt": int(len(paar_ids)),
    }
    return S, info


def ersetzbarkeit_matrix(model, ids, woerter, **kw):
    n = len(woerter)
    paare = np.array(list(combinations(range(n), 2)))
    S, info = ersetzbarkeit(model, ids, woerter, paare, **kw)
    M = np.full((n, n), np.nan)
    M[paare[:, 0], paare[:, 1]] = S
    M[paare[:, 1], paare[:, 0]] = S
    return M, info


def gleichartigkeit_G(cos_werte, S, mitte=0.8):
    cos_werte = np.asarray(cos_werte, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    ok = np.isfinite(S) & np.isfinite(cos_werte)
    c, s = cos_werte[ok], S[ok]
    if len(c) < 3:
        return float("nan")
    rang = c.argsort().argsort() / (len(c) - 1)
    rand = (1 - mitte) / 2
    m = (rang >= rand) & (rang <= 1 - rand)
    if m.sum() < 3 or np.ptp(c[m]) == 0 or np.ptp(s[m]) == 0:
        return float("nan")
    return float(spearmanr(c[m], s[m]).statistic)
```

- [ ] **Step 4: Tests laufen lassen, Erfolg prüfen**

Run: `venv/bin/python -m pytest tests/test_ersetzbarkeit.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add chroma/messung/ersetzbarkeit.py tests/test_ersetzbarkeit.py
git commit -m "Chroma: Ersetzbarkeit und Gleichartigkeitsindex G

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- chroma/messung/ersetzbarkeit.py tests/test_ersetzbarkeit.py
```

---

### Task 6: Sekundärmaße (Verschachtelung, eigene Metrik, Nachbarn, Register, SimLex)

**Files:**
- Create: `chroma/messung/verschachtelung.py`, `chroma/messung/metrik.py`, `chroma/messung/register.py`, `chroma/messung/normen.py`
- Test: `tests/test_sekundaer.py`

**Interfaces:**
- Consumes: `ChromaModell` (Task 1).
- Produces: `kophenetische_korrelation(E) -> float`; `cluster_labels(E, n_cluster) -> np.ndarray[int]`; `qk_metrik(model) -> Tensor[d,d]` (symmetrisch); `qk_aehnlichkeit(model, wort_ids) -> np.ndarray[n,n]`; `nachbar_ueberlappung(A: (n,n), B: (n,n), k=10) -> float`; `stabile_cluster(labels_je_seed: list[np.ndarray], min_seeds=3, min_jaccard=0.5) -> list[dict]` mit `mitglieder: list[int]` (Indizes in die Wortliste), `seeds: int`; `schreibe_register(cluster, woerter_text: list[str], pfad)`; `lade_simlex(pfad) -> list[(str,str,float)]`; `simlex_vergleich(simlex, vocab, E) -> dict(rho, abdeckung, n)`; `lade_simlex_herunter(ziel_ordner) -> Path`.

- [ ] **Step 1: Failing tests schreiben**

`tests/test_sekundaer.py`:

```python
import numpy as np
import torch

from chroma.messung import metrik, normen, register, verschachtelung
from chroma.modell import ChromaConfig, ChromaModell


def test_kophenetisch_verschachtelt_hoeher_als_zufall():
    rng = np.random.default_rng(0)
    gross = rng.normal(size=(3, 20)) * 3
    klein = gross[:, None] + rng.normal(size=(3, 3, 20))
    punkte = (klein[:, :, None] + 0.1 * rng.normal(size=(3, 3, 10, 20))).reshape(90, 20)
    c_nest = verschachtelung.kophenetische_korrelation(punkte)
    c_zufall = verschachtelung.kophenetische_korrelation(rng.normal(size=(90, 20)))
    assert c_nest > 0.8 and c_nest > c_zufall


def test_qk_metrik_symmetrisch():
    torch.manual_seed(0)
    m = ChromaModell(ChromaConfig(vocab_size=30, block_size=8, n_layer=1, n_head=2, n_embd=8, farbe="beta"))
    M = metrik.qk_metrik(m)
    assert M.shape == (8, 8) and torch.allclose(M, M.T)
    A = metrik.qk_aehnlichkeit(m, [1, 2, 3, 4])
    assert A.shape == (4, 4) and np.allclose(A, A.T, atol=1e-6)


def test_nachbar_ueberlappung():
    rng = np.random.default_rng(0)
    A = rng.random((20, 20))
    A = A + A.T
    assert metrik.nachbar_ueberlappung(A, A, k=5) == 1.0
    B = A.copy()
    B[0, 1] = B[1, 0] = np.nan
    assert 0.0 <= metrik.nachbar_ueberlappung(A, B, k=5) <= 1.0


def test_stabile_cluster():
    rng = np.random.default_rng(0)
    basis_labels = np.repeat(np.arange(10), 10)
    umbenannt = [(basis_labels + s) % 10 for s in range(5)]  # gleiche Mengen, andere Nummern
    assert len(register.stabile_cluster(umbenannt)) == 10
    zufall = [rng.integers(0, 10, 100) for _ in range(5)]
    assert len(register.stabile_cluster(zufall)) == 0


def test_register_datei(tmp_path):
    cl = [{"mitglieder": [0, 2], "seeds": 4}]
    register.schreibe_register(cl, [" haus", " baum", " stein"], tmp_path / "register.md")
    text = (tmp_path / "register.md").read_text()
    assert "haus" in text and "stein" in text and "explorativ" in text


def test_simlex(tmp_path):
    datei = tmp_path / "SimLex-999.txt"
    datei.write_text("word1\tword2\tPOS\tSimLex999\nhaus\tbaum\tN\t2.0\nhaus\tstein\tN\t5.0\n"
                     "baum\tstein\tN\t8.0\nhaus\tfehlt\tN\t1.0\n")
    simlex = normen.lade_simlex(datei)
    assert len(simlex) == 4
    vocab = {"Ġhaus": 0, "Ġbaum": 1, "Ġstein": 2}
    E = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    erg = normen.simlex_vergleich(simlex, vocab, E)
    assert erg["n"] == 3 and abs(erg["abdeckung"] - 0.75) < 1e-12
    assert np.isfinite(erg["rho"])
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `venv/bin/python -m pytest tests/test_sekundaer.py -v`
Expected: FAIL mit `ImportError: cannot import name 'metrik'`

- [ ] **Step 3: Sekundärmaße implementieren**

`chroma/messung/verschachtelung.py`:

```python
"""
chroma/messung/verschachtelung.py
Zweitmaß (c): Lässt sich aus der Gleichartigkeit der Anlage eine Ordnung von Gattung und Art
(ein porphyrischer Baum) rekonstruieren? Gemessen als kophenetische Korrelation.
"""

import numpy as np
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist


def kophenetische_korrelation(E):
    d = pdist(np.asarray(E, dtype=np.float64), metric="cosine")
    c, _ = cophenet(linkage(d, method="average"), d)
    return float(c)


def cluster_labels(E, n_cluster):
    d = pdist(np.asarray(E, dtype=np.float64), metric="cosine")
    return fcluster(linkage(d, method="average"), n_cluster, criterion="maxclust")
```

`chroma/messung/metrik.py`:

```python
"""
chroma/messung/metrik.py
Explorativ, sekundär. Der Kosinus ist unser Vorurteil (V2 § 6.1: Metrik ist Füllung).
G_QK misst mit der eigenen, durch Gewohnheit sedimentierten Metrik des Modells: der
Bilinearform W_Qᵀ W_K der Köpfe in Schicht 0. G_top ist das topologische Gegenstück:
Überlappung der nächsten Nachbarn statt metrischer Werte.
"""

import numpy as np
import torch


@torch.no_grad()
def qk_metrik(model):
    d = model.cfg.n_embd
    W = model.h[0].attn.c_attn.weight
    M = W[:d].T @ W[d:2 * d] / model.cfg.n_head  # Summe über Köpfe der Blöcke W_q,hᵀ W_k,h
    return (M + M.T) / 2


@torch.no_grad()
def qk_aehnlichkeit(model, wort_ids):
    """Ähnlichkeit in der Lesung von Block 0 (bei Bβ also durch die farbige Linse); Biases bleiben außen vor."""
    E = model.archetyp()[torch.as_tensor(list(wort_ids), device=model.wte.weight.device)]
    X = model.h[0].les_1(E).double()
    return (X @ qk_metrik(model).double() @ X.T).cpu().numpy()


def nachbar_ueberlappung(A, B, k=10):
    """Mittlere Jaccard-Überlappung der k nächsten Nachbarn (ohne das Wort selbst); NaN zählt als fern."""
    def nachbarn(M):
        M = np.where(np.isfinite(M), M, -np.inf).astype(np.float64)
        np.fill_diagonal(M, -np.inf)
        return np.argsort(-M, axis=1)[:, :k]

    na, nb = nachbarn(np.asarray(A)), nachbarn(np.asarray(B))
    j = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(na, nb)]
    return float(np.mean(j))
```

`chroma/messung/register.py`:

```python
"""
chroma/messung/register.py
Register „Begriffsbedarf“ (explorativ): Cluster, die über Biographien (Seeds) hinweg stabil sind.
Der Autor benennt sie; was sich nicht benennen lässt, verlangt einen Begriff (Asynchronizität
auf der Ebene der Methode).
"""

from pathlib import Path

import numpy as np


def _mengen(labels):
    labels = np.asarray(labels)
    return [frozenset(np.flatnonzero(labels == c).tolist()) for c in np.unique(labels)]


def _jaccard(a, b):
    return len(a & b) / len(a | b)


def stabile_cluster(labels_je_seed, min_seeds=3, min_jaccard=0.5):
    mengen = [_mengen(l) for l in labels_je_seed]
    behalten = []
    for i, ms in enumerate(mengen):
        for m in ms:
            if any(_jaccard(m, frozenset(b["mitglieder"])) >= min_jaccard for b in behalten):
                continue
            seeds = 1 + sum(any(_jaccard(m, o) >= min_jaccard for o in mengen[j])
                            for j in range(len(mengen)) if j != i)
            if seeds >= min_seeds:
                behalten.append({"mitglieder": sorted(m), "seeds": int(seeds)})
    return behalten


def schreibe_register(cluster, woerter_text, pfad):
    zeilen = [
        "# Register „Begriffsbedarf“ (explorativ)",
        "",
        "Stabile Cluster über Seeds. Spalte „Begriff“ füllt der Autor; leer gebliebene Zeilen "
        "sind Phänomene, die einen Begriff verlangen.",
        "",
        "| Nr. | Seeds | Mitglieder | Begriff |",
        "|---|---|---|---|",
    ]
    for n, c in enumerate(cluster, 1):
        mitglieder = ", ".join(woerter_text[i].strip() for i in c["mitglieder"])
        zeilen.append(f"| {n} | {c['seeds']} | {mitglieder} | |")
    Path(pfad).write_text("\n".join(zeilen) + "\n", encoding="utf-8")
```

`chroma/messung/normen.py`:

```python
"""
chroma/messung/normen.py
Vergleich (a): menschliche Ähnlichkeitsnormen (SimLex-999). Entscheidet nichts, denn es setzte
voraus, dass die Maschine eine perfekte Metapher menschlicher Erfahrung ist.
"""

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SIMLEX_URL = "https://fh295.github.io/SimLex-999.zip"


def lade_simlex_herunter(ziel_ordner):
    ziel = Path(ziel_ordner)
    ziel.mkdir(parents=True, exist_ok=True)
    zip_pfad = ziel / "SimLex-999.zip"
    urllib.request.urlretrieve(SIMLEX_URL, zip_pfad)
    with zipfile.ZipFile(zip_pfad) as z:
        z.extract("SimLex-999/SimLex-999.txt", ziel)
    return ziel / "SimLex-999" / "SimLex-999.txt"


def lade_simlex(pfad):
    zeilen = Path(pfad).read_text(encoding="utf-8").splitlines()[1:]
    paare = []
    for z in zeilen:
        teile = z.split("\t")
        if len(teile) >= 4:
            paare.append((teile[0], teile[1], float(teile[3])))
    return paare


def simlex_vergleich(simlex, vocab, E):
    """Nur Paare, deren beide Wörter einzelne Ganzwort-Tokens sind; Abdeckung wird berichtet."""
    E = np.asarray(E, dtype=np.float64)
    treffer = [(vocab["Ġ" + a], vocab["Ġ" + b], s) for a, b, s in simlex
               if "Ġ" + a in vocab and "Ġ" + b in vocab]
    erg = {"n": len(treffer), "abdeckung": len(treffer) / max(len(simlex), 1), "rho": float("nan")}
    if len(treffer) >= 3:
        En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        cos = [float(En[i] @ En[j]) for i, j, _ in treffer]
        erg["rho"] = float(spearmanr(cos, [s for _, _, s in treffer]).statistic)
    return erg
```

- [ ] **Step 4: Tests laufen lassen, Erfolg prüfen**

Run: `venv/bin/python -m pytest tests/test_sekundaer.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add chroma/messung/verschachtelung.py chroma/messung/metrik.py chroma/messung/register.py chroma/messung/normen.py tests/test_sekundaer.py
git commit -m "Chroma: Sekundärmaße (Verschachtelung, QK-Metrik, Nachbarn, Register, SimLex)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- chroma/messung/verschachtelung.py chroma/messung/metrik.py chroma/messung/register.py chroma/messung/normen.py tests/test_sekundaer.py
```

---

### Task 7: Mess-Pipeline je Checkpoint

**Files:**
- Create: `chroma/messen.py`
- Test: `tests/test_messen.py`

**Interfaces:**
- Consumes: `lade_modell` (Task 3); `lade_split`, `lade_tokenizer`, `lade_frequenz`, `wortliste` (Task 2); `basis.*` (Task 4); `ersetzbarkeit.*` (Task 5); `verschachtelung`, `metrik`, `normen` (Task 6).
- Produces: `MessConfig(n_woerter=2000, n_paare=20000, n_kontexte=20, min_kontexte=5, fenster=64, nach=8, n_anker=150, k_nachbarn=10, n_fenster_strom=64, schicht_strom=3, n_cluster=100, seed=0, batch=256)`; `messe_checkpoint(ckpt, korpus_ordner, aus_json, mc=MessConfig(), device="cpu", simlex_pfad=None) -> dict`. Schreibt `aus_json` und daneben `<stem>_cluster.npy`. JSON-Schlüssel: `G, G_QK, G_top, kurtosis (Liste je Schicht), kurtosis_ohne_pos0, kurtosis_kontrolle, pr_archetyp_eigen, pr_archetyp_ica, pr_strom_eigen, pr_strom_ica, kophenetisch, simlex, info, farbe, ckpt`.

- [ ] **Step 1: Failing test schreiben**

`tests/test_messen.py`:

```python
import json

import numpy as np

from chroma import korpus
from chroma.messen import MessConfig, messe_checkpoint
from chroma.training import TrainConfig, trainiere
from tests.test_korpus import synthetische_texte

SCHLUESSEL = {"G", "G_QK", "G_top", "kurtosis", "kurtosis_ohne_pos0", "kurtosis_kontrolle", "pr_archetyp_eigen",
              "pr_archetyp_ica", "pr_strom_eigen", "pr_strom_ica", "kophenetisch", "simlex", "info",
              "farbe", "ckpt"}


def test_messe_checkpoint(tmp_path):
    splits = {"train": synthetische_texte(600, 0), "val": synthetische_texte(80, 1), "test": synthetische_texte(300, 2)}
    korpus.baue_korpus(tmp_path, splits=splits, vocab_size=400, quelle="synthetisch")
    tc = TrainConfig(arm="Bb", seed=0, lr=3e-3, batch_size=8, block_size=32, n_layer=2, n_head=2, n_embd=16,
                     max_tokens=8 * 32 * 30, eval_alle=10, eval_batches=2)
    trainiere(tmp_path, tmp_path / "lauf", tc, log=lambda *a: None)
    mc = MessConfig(n_woerter=10, n_paare=30, n_kontexte=4, min_kontexte=2, fenster=16, nach=4, n_anker=6,
                    k_nachbarn=3, n_fenster_strom=16, schicht_strom=1, n_cluster=3, batch=64)
    erg = messe_checkpoint(tmp_path / "lauf" / "ende.pt", tmp_path, tmp_path / "lauf" / "messung_ende.json", mc)
    assert SCHLUESSEL <= set(erg)
    assert erg["farbe"] == "beta" and len(erg["kurtosis"]) == 3
    gespeichert = json.loads((tmp_path / "lauf" / "messung_ende.json").read_text())
    assert gespeichert["G"] == erg["G"] or (np.isnan(gespeichert["G"]) and np.isnan(erg["G"]))
    assert (tmp_path / "lauf" / "messung_ende_cluster.npy").exists()
```

- [ ] **Step 2: Test laufen lassen, Fehlschlag prüfen**

Run: `venv/bin/python -m pytest tests/test_messen.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'chroma.messen'`

- [ ] **Step 3: Pipeline implementieren**

`tests/__init__.py` (leer, damit `from tests.test_korpus import ...` funktioniert):

```python
```

`chroma/messen.py`:

```python
"""
chroma/messen.py
Alle Maße für einen Checkpoint (Spec § 4.5): Kernmaß G, Basismaße, Verschachtelung,
explorativ G_QK, G_top, Cluster fürs Register, SimLex als Vergleich.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .korpus import lade_frequenz, lade_split, lade_tokenizer, wortliste
from .messung import basis, metrik, normen, verschachtelung
from .messung import ersetzbarkeit as ers
from .training import lade_modell


@dataclass
class MessConfig:
    n_woerter: int = 2000
    n_paare: int = 20000
    n_kontexte: int = 20
    min_kontexte: int = 5
    fenster: int = 64
    nach: int = 8
    n_anker: int = 150
    k_nachbarn: int = 10
    n_fenster_strom: int = 64
    schicht_strom: int = 3
    n_cluster: int = 100
    seed: int = 0
    batch: int = 256


@torch.no_grad()
def _strom_proben(model, ids, mc, device, rng):
    T = model.cfg.block_size
    starts = rng.integers(0, len(ids) - T - 1, size=mc.n_fenster_strom)
    x = torch.as_tensor(np.stack([np.asarray(ids[s:s + T], dtype=np.int64) for s in starts]), device=device)
    zust = [z.double().cpu().numpy() for z in model.strom(x)]
    return x.cpu().numpy(), zust


def messe_checkpoint(ckpt, korpus_ordner, aus_json, mc=MessConfig(), device="cpu", simlex_pfad=None):
    model = lade_modell(ckpt, device)
    farbe = model.cfg.farbe
    tok = lade_tokenizer(korpus_ordner)
    woerter = wortliste(tok.get_vocab(), lade_frequenz(korpus_ordner), mc.n_woerter)
    test = lade_split(korpus_ordner, "test")
    rng = np.random.default_rng(mc.seed)
    kw = dict(n_kontexte=mc.n_kontexte, min_kontexte=mc.min_kontexte, fenster=mc.fenster, nach=mc.nach,
              device=device, seed=mc.seed, batch=mc.batch)

    with torch.no_grad():
        archetyp = model.archetyp().double().cpu().numpy()
    E = archetyp[woerter]
    cos = ers.kosinus_matrix(E)

    # Kernmaß (b)
    paare = ers.ziehe_paare(cos, mc.n_paare, rng)
    S, info = ers.ersetzbarkeit(model, test, woerter, paare, **kw)
    G = ers.gleichartigkeit_G(cos[paare[:, 0], paare[:, 1]], S)

    # explorativ: eigene Metrik und Topologie
    qk = metrik.qk_aehnlichkeit(model, woerter)
    G_QK = ers.gleichartigkeit_G(qk[paare[:, 0], paare[:, 1]], S)
    anker = woerter[:mc.n_anker]
    S_anker, info_anker = ers.ersetzbarkeit_matrix(model, test, anker, **kw)
    G_top = metrik.nachbar_ueberlappung(cos[:len(anker), :len(anker)], S_anker, k=mc.k_nachbarn)

    # Basis: Kurtosis je Schicht, mit und ohne Position 0, und Drehungskontrolle
    x, zust = _strom_proben(model, test, mc, device, rng)
    d = model.cfg.n_embd
    kurt = [basis.exzess_kurtosis(z.reshape(-1, d)) for z in zust]
    kurt0 = [basis.exzess_kurtosis(z[:, 1:].reshape(-1, d)) for z in zust]
    kontrolle = [float(np.mean(basis.kurtosis_kontrolle(z.reshape(-1, d)[:4096], 3, rng))) for z in zust]

    # P2: Archetyp und Stream-Lesung in Schicht schicht_strom, eigene Basis gegen ICA-Basis
    pr_arch_eigen = float(basis.participation_ratio(E).mean())
    pr_arch_ica = float(basis.participation_ratio(E @ basis.ica_drehung(E, mc.seed)).mean()) if len(E) >= E.shape[1] else float("nan")
    l = min(mc.schicht_strom, model.cfg.n_layer)
    maske = np.isin(x, woerter)
    R = zust[l][maske]
    R_lesung = F.softplus(torch.from_numpy(R)).numpy() if farbe == "beta" else R
    pr_strom_eigen = float(basis.participation_ratio(R_lesung).mean())
    pr_strom_ica = float(basis.participation_ratio(R @ basis.ica_drehung(R, mc.seed)).mean()) if len(R) >= d else float("nan")

    # Verschachtelung und Cluster fürs Register
    koph = verschachtelung.kophenetische_korrelation(E)
    labels = verschachtelung.cluster_labels(E, mc.n_cluster)

    simlex = normen.simlex_vergleich(normen.lade_simlex(simlex_pfad), tok.get_vocab(), archetyp) if simlex_pfad else None

    erg = {
        "ckpt": str(ckpt), "farbe": farbe, "G": G, "G_QK": G_QK, "G_top": G_top,
        "kurtosis": kurt, "kurtosis_ohne_pos0": kurt0, "kurtosis_kontrolle": kontrolle,
        "pr_archetyp_eigen": pr_arch_eigen, "pr_archetyp_ica": pr_arch_ica,
        "pr_strom_eigen": pr_strom_eigen, "pr_strom_ica": pr_strom_ica,
        "kophenetisch": koph, "simlex": simlex,
        "info": {**info, "anker": info_anker, "strom_zeilen": int(len(R)), "messconfig": asdict(mc)},
    }
    aus = Path(aus_json)
    aus.parent.mkdir(parents=True, exist_ok=True)
    aus.write_text(json.dumps(erg, indent=2), encoding="utf-8")  # NaN bleibt NaN (gültig für json.loads)
    np.save(aus.with_name(aus.stem + "_cluster.npy"), labels)
    return erg
```

- [ ] **Step 4: Test laufen lassen, Erfolg prüfen**

Run: `venv/bin/python -m pytest tests/test_messen.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add chroma/messen.py tests/__init__.py tests/test_messen.py
git commit -m "Chroma: Mess-Pipeline je Checkpoint

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- chroma/messen.py tests/__init__.py tests/test_messen.py
```

---

### Task 8: Auswertung über Seeds (Entscheidungsregeln P1–P3)

**Files:**
- Create: `chroma/auswertung.py`
- Test: `tests/test_auswertung.py`

**Interfaces:**
- Consumes: Messungs-JSON aus Task 7 unter `<haupt>/<arm>/seed<k>/messung_stufe_<x.xx>.json`; `ARME` (Task 1).
- Produces: `lade_messungen(haupt) -> dict[(arm, seed, stufe: float)] -> dict`; `gemeinsame_stufe(messungen, arme=tuple(ARME)) -> float | None`; `werte(messungen, arm, stufe, schluessel) -> (np.ndarray, int)` (endliche Werte, Anzahl verworfener); `entscheide_p1(g_a0, g_a, delta=0.05, alpha=0.05) -> dict(urteil, p_differenz, p_tost, n_a0, n_a)` mit `urteil ∈ {"bestätigt","widerlegt","unentschieden","nicht auswertbar"}`; `entscheide_p2(pr_b, pr_a_ica, g_b, g_a, delta=0.05, alpha=0.05) -> dict(lesbar, p_lesbar, preis_ok, p_preis)`; `bericht(haupt, delta=0.05, alpha=0.05, schwelle_a0=2.0, schwelle_a=10.0) -> dict` (schreibt `<haupt>/bericht.md`).

- [ ] **Step 1: Failing tests schreiben**

`tests/test_auswertung.py`:

```python
import json

import numpy as np

from chroma import auswertung as aw


def schreibe(haupt, arm, seed, stufe, **werte):
    d = haupt / arm / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    basis = {"G": 0.5, "kurtosis": [0.0], "pr_archetyp_eigen": 50.0, "pr_archetyp_ica": 50.0,
             "pr_strom_eigen": 50.0, "pr_strom_ica": 50.0, "kophenetisch": 0.5, "G_QK": 0.1, "G_top": 0.1}
    basis.update(werte)
    (d / f"messung_stufe_{stufe:.2f}.json").write_text(json.dumps(basis))


def test_p1_bestaetigt_widerlegt_unentschieden():
    assert aw.entscheide_p1(np.array([.10, .12, .11, .09, .13]), np.array([.50, .52, .51, .49, .53]))["urteil"] == "bestätigt"
    assert aw.entscheide_p1(np.array([.400, .402, .398, .401, .399]), np.array([.401, .399, .400, .402, .398]))["urteil"] == "widerlegt"
    assert aw.entscheide_p1(np.array([.3, .5, .1, .6, .2]), np.array([.35, .55, .15, .62, .3]))["urteil"] == "unentschieden"
    assert aw.entscheide_p1(np.array([.3]), np.array([.4, .5]))["urteil"] == "nicht auswertbar"


def test_gemeinsame_stufe_weicht_aus(tmp_path):
    for arm in ("A0", "A", "Ba", "Bb"):
        for seed in range(2):
            schreibe(tmp_path, arm, seed, 5.0)
            if arm != "A0":
                schreibe(tmp_path, arm, seed, 4.0)
    assert aw.gemeinsame_stufe(aw.lade_messungen(tmp_path)) == 5.0


def test_keine_gemeinsame_stufe(tmp_path):
    for arm in ("A", "Ba", "Bb"):
        schreibe(tmp_path, arm, 0, 4.0)
    assert aw.gemeinsame_stufe(aw.lade_messungen(tmp_path)) is None
    erg = aw.bericht(tmp_path)
    assert erg["stufe"] is None
    assert "keine gemeinsame Stufe" in (tmp_path / "bericht.md").read_text()


def test_nan_seed_wird_verworfen_und_gezaehlt(tmp_path):
    schreibe(tmp_path, "A0", 0, 5.0, G=float("nan"))
    schreibe(tmp_path, "A0", 1, 5.0, G=0.3)
    w, verworfen = aw.werte(aw.lade_messungen(tmp_path), "A0", 5.0, "G")
    assert list(w) == [0.3] and verworfen == 1


def test_bericht_vollstaendig(tmp_path):
    rng = np.random.default_rng(0)
    for seed in range(5):
        schreibe(tmp_path, "A0", seed, 5.0, G=0.1 + 0.01 * rng.random(), kurtosis=[0.5])
        schreibe(tmp_path, "A", seed, 5.0, G=0.5 + 0.01 * rng.random(), kurtosis=[40.0], pr_archetyp_ica=80.0, pr_strom_ica=80.0)
        schreibe(tmp_path, "Ba", seed, 5.0, G=0.5 + 0.01 * rng.random(), pr_archetyp_eigen=20.0 + rng.random())
        schreibe(tmp_path, "Bb", seed, 5.0, G=0.5 + 0.01 * rng.random(), pr_strom_eigen=20.0 + rng.random())
    erg = aw.bericht(tmp_path)
    assert erg["stufe"] == 5.0
    assert erg["manipulation"]["bestanden"] is True
    assert erg["p1"]["urteil"] == "bestätigt"
    assert erg["p2"]["Ba"]["lesbar"] and erg["p2"]["Bb"]["lesbar"]
    text = (tmp_path / "bericht.md").read_text()
    assert "P1" in text and "explorativ" in text
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `venv/bin/python -m pytest tests/test_auswertung.py -v`
Expected: FAIL mit `ImportError: cannot import name 'auswertung'`

- [ ] **Step 3: Auswertung implementieren**

`chroma/auswertung.py`:

```python
"""
chroma/auswertung.py
Entscheidungsregeln der Hauptstudie (Spec § 4.6). Einheit der Inferenz ist der Seed
(verschiedene Biographien desselben Gattungswesens). δ ist eine Festsetzung.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

from .arme import ARME


def lade_messungen(haupt):
    erg = {}
    for f in Path(haupt).glob("*/seed*/messung_stufe_*.json"):
        arm, seed = f.parent.parent.name, int(f.parent.name[4:])
        stufe = float(f.stem.rsplit("_", 1)[-1])
        erg[(arm, seed, stufe)] = json.loads(f.read_text(encoding="utf-8"))
    return erg


def gemeinsame_stufe(messungen, arme=tuple(ARME)):
    """Die niedrigste Loss-Stufe, die alle Seeds aller Arme erreicht haben; None, wenn es keine gibt."""
    pro = {}
    for arm, seed, stufe in messungen:
        pro.setdefault((arm, seed), set()).add(stufe)
    if not pro or any(a not in {k[0] for k in pro} for a in arme):
        return None
    gemeinsam = set.intersection(*pro.values())
    return min(gemeinsam) if gemeinsam else None


def _wert(m, schluessel):
    v = m.get(schluessel)
    if isinstance(v, list):
        v = float(np.mean(v)) if v else None
    return float("nan") if v is None else float(v)


def werte(messungen, arm, stufe, schluessel):
    alle = [_wert(m, schluessel) for (a, _, s), m in sorted(messungen.items()) if a == arm and s == stufe]
    w = np.array([v for v in alle if np.isfinite(v)])
    return w, len(alle) - len(w)


def entscheide_p1(g_a0, g_a, delta=0.05, alpha=0.05):
    erg = {"n_a0": int(len(g_a0)), "n_a": int(len(g_a)), "p_differenz": None, "p_tost": None}
    if len(g_a0) < 2 or len(g_a) < 2:
        return {**erg, "urteil": "nicht auswertbar"}
    p_diff = ttest_ind(g_a0, g_a, equal_var=False, alternative="less").pvalue
    p_unten = ttest_ind(g_a0 + delta, g_a, equal_var=False, alternative="greater").pvalue
    p_oben = ttest_ind(g_a0 - delta, g_a, equal_var=False, alternative="less").pvalue
    p_tost = max(p_unten, p_oben)
    urteil = "bestätigt" if p_diff < alpha else "widerlegt" if p_tost < alpha else "unentschieden"
    return {**erg, "urteil": urteil, "p_differenz": float(p_diff), "p_tost": float(p_tost)}


def entscheide_p2(pr_b, pr_a_ica, g_b, g_a, delta=0.05, alpha=0.05):
    if min(len(pr_b), len(pr_a_ica), len(g_b), len(g_a)) < 2:
        return {"lesbar": None, "p_lesbar": None, "preis_ok": None, "p_preis": None}
    p_les = ttest_ind(pr_b, pr_a_ica, equal_var=False, alternative="less").pvalue
    p_preis = ttest_ind(g_b + delta, g_a, equal_var=False, alternative="greater").pvalue
    return {"lesbar": bool(p_les < alpha), "p_lesbar": float(p_les),
            "preis_ok": bool(p_preis < alpha), "p_preis": float(p_preis)}


def bericht(haupt, delta=0.05, alpha=0.05, schwelle_a0=2.0, schwelle_a=10.0):
    haupt = Path(haupt)
    m = lade_messungen(haupt)
    stufe = gemeinsame_stufe(m)
    zeilen = ["# Auswertung Hauptstudie (Chroma)", "",
              f"δ = {delta} (Festsetzung), α = {alpha}, Einheit: Seed.", ""]
    erg = {"stufe": stufe}
    if stufe is None:
        zeilen.append("**Es gibt keine gemeinsame Stufe**, die alle Seeds aller Arme erreicht haben. "
                      "Keine Entscheidung möglich; als Einschränkung berichten.")
        (haupt / "bericht.md").write_text("\n".join(zeilen) + "\n", encoding="utf-8")
        return erg
    zeilen.append(f"Primäre Loss-Stufe: {stufe:.2f}")
    zeilen.append("")

    k_a0, _ = werte(m, "A0", stufe, "kurtosis")
    k_a, _ = werte(m, "A", stufe, "kurtosis")
    bestanden = bool(len(k_a0) and len(k_a) and k_a0.mean() < schwelle_a0 and k_a.mean() > schwelle_a)
    erg["manipulation"] = {"bestanden": bestanden, "kurtosis_a0": k_a0.tolist(), "kurtosis_a": k_a.tolist()}
    zeilen += ["## Manipulationsprüfung", "",
               f"Kurtosis A₀ Mittel {k_a0.mean() if len(k_a0) else float('nan'):.3f} (Schwelle < {schwelle_a0}), "
               f"A Mittel {k_a.mean() if len(k_a) else float('nan'):.3f} (Schwelle > {schwelle_a}): "
               f"**{'bestanden' if bestanden else 'NICHT bestanden – Vergleich nichtig'}**", ""]

    g = {arm: werte(m, arm, stufe, "G") for arm in ARME}
    erg["p1"] = entscheide_p1(g["A0"][0], g["A"][0], delta, alpha)
    zeilen += ["## P1 (Gerüstfrage)", "",
               f"G(A₀) = {g['A0'][0].round(4).tolist()} (verworfen: {g['A0'][1]}), "
               f"G(A) = {g['A'][0].round(4).tolist()} (verworfen: {g['A'][1]})",
               f"p Differenz = {erg['p1']['p_differenz']}, p TOST = {erg['p1']['p_tost']} → "
               f"**{erg['p1']['urteil']}**", ""]

    pr_a_arch, _ = werte(m, "A", stufe, "pr_archetyp_ica")
    pr_a_strom, _ = werte(m, "A", stufe, "pr_strom_ica")
    erg["p2"] = {
        "Ba": entscheide_p2(werte(m, "Ba", stufe, "pr_archetyp_eigen")[0], pr_a_arch, g["Ba"][0], g["A"][0], delta, alpha),
        "Bb": entscheide_p2(werte(m, "Bb", stufe, "pr_strom_eigen")[0], pr_a_strom, g["Bb"][0], g["A"][0], delta, alpha),
    }
    zeilen += ["## P2 (Sichtbarkeitsfrage)", ""]
    for arm, e in erg["p2"].items():
        zeilen.append(f"- {arm}: lesbar = {e['lesbar']} (p = {e['p_lesbar']}), "
                      f"Preisprüfung bestanden = {e['preis_ok']} (p = {e['p_preis']})")
    zeilen.append("")

    koph = {arm: werte(m, arm, stufe, "kophenetisch")[0] for arm in ARME}
    mittel = {arm: float(v.mean()) if len(v) else float("nan") for arm, v in koph.items()}
    erg["p3"] = mittel
    zeilen += ["## P3 (Verschachtelung, sekundär, gerichtet)", "",
               ", ".join(f"{a}: {v:.4f}" for a, v in mittel.items()), ""]

    zeilen += ["## Weitere Maße (explorativ, nicht entscheidend)", ""]
    for schl in ("G_QK", "G_top"):
        zeilen.append(f"- {schl}: " + ", ".join(
            f"{arm} {np.mean(werte(m, arm, stufe, schl)[0]):.4f}" for arm in ARME))
    (haupt / "bericht.md").write_text("\n".join(zeilen) + "\n", encoding="utf-8")
    (haupt / "auswertung.json").write_text(json.dumps(erg, indent=2, default=str), encoding="utf-8")
    return erg
```

- [ ] **Step 4: Tests laufen lassen, Erfolg prüfen**

Run: `venv/bin/python -m pytest tests/test_auswertung.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add chroma/auswertung.py tests/test_auswertung.py
git commit -m "Chroma: Auswertung über Seeds (P1–P3, Manipulationsprüfung)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- chroma/auswertung.py tests/test_auswertung.py
```

---

### Task 9: Vorstudie an GPT-2 und Pythia

**Files:**
- Create: `chroma/vorstudie.py`
- Test: `tests/test_vorstudie.py`

**Interfaces:**
- Consumes: `basis.*` (Task 4); `wortliste` (Task 2); `ers.kosinus_matrix` (Task 5).
- Produces: `PYTHIA_STEPS` (11 Revisionen), `PYTHIA_DICHT` (30 Revisionen); `messe_hf_modell(model, seqs: np.ndarray[n,T], vocab: dict[str,int], device="cpu", batch=8, n_drehungen=10, seed=0, n_ausreisser=3, n_woerter=2000) -> (dict, np.ndarray E_woerter)`; `effektiver_rang(E) -> float`; `korrelationsdimension(P, n_skalen=8) -> dict(radien, dimension)`; `lauf_vorstudie(modell_name, revisionen: list[str], seqs_pfad, aus_ordner, device) -> None` (schreibt `<aus>/<slug>/<rev>.json` und `<rev>_woerter.npy`); `vorstudie_bericht(aus_ordner, v_schwelle=1.0) -> str` (schreibt `<aus>/bericht.md`).

- [ ] **Step 1: Failing tests schreiben**

`tests/test_vorstudie.py`:

```python
import json

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from chroma import vorstudie

VOCAB = {f"Ġ{w}": i for i, w in enumerate(["haus", "baum", "katze", "hund", "stein", "fluss", "berg", "licht",
                                            "farbe", "zeit", "art", "folge"])}


def mini_gpt2():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=100, n_positions=32)).eval()


def seqs():
    return np.random.default_rng(0).integers(0, 100, size=(6, 32))


def test_messe_hf_modell_struktur():
    erg, E = vorstudie.messe_hf_modell(mini_gpt2(), seqs(), VOCAB, n_drehungen=3, n_woerter=10)
    for k in ("kurtosis", "kurtosis_ohne_pos0", "kontrolle", "v3_rho", "v3_p", "e4_rho", "effektiver_rang",
              "korrelationsdimension", "ausreisser_dims"):
        assert k in erg
    assert len(erg["kurtosis"]) == 2 and len(erg["kontrolle"]) == 2
    assert E.shape[1] == 32


def test_eingepflanzte_achse_wird_gefunden():
    m = mini_gpt2()
    with torch.no_grad():
        m.transformer.wte.weight[:, 0] += 20.0
    erg, _ = vorstudie.messe_hf_modell(m, seqs(), VOCAB, n_drehungen=3, n_woerter=10)
    assert erg["kurtosis"][0] > erg["kontrolle"][0] + 1.0
    assert 0 in erg["ausreisser_dims"]


def test_bericht(tmp_path):
    ordner = tmp_path / "mini"
    ordner.mkdir()
    for rev in ("step0", "step1"):
        erg, E = vorstudie.messe_hf_modell(mini_gpt2(), seqs(), VOCAB, n_drehungen=2, n_woerter=10)
        (ordner / f"{rev}.json").write_text(json.dumps({**erg, "revision": rev}))
        np.save(ordner / f"{rev}_woerter.npy", E)
    text = vorstudie.vorstudie_bericht(tmp_path)
    assert "V1" in text and "explorativ" in text and "E1" in text
    assert (tmp_path / "bericht.md").exists()
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `venv/bin/python -m pytest tests/test_vorstudie.py -v`
Expected: FAIL mit `ImportError: cannot import name 'vorstudie'`

- [ ] **Step 3: Vorstudie implementieren**

`chroma/vorstudie.py`:

```python
"""
chroma/vorstudie.py
Genealogie der Achsen (Spec § 3): Liegt im sedimentierten Archetyp von GPT-2 und Pythia schon
eine bevorzugte Basis, und entsteht sie durch Gewohnheit? Präregistriert: V1–V3 (PRAEREG_8).
Explorativ: E1 Nachträglichkeit, E2 fraktale Dimension, E3 Knotenlinie, E4 Reflex.
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from .korpus import wortliste
from .messung import basis

PYTHIA_STEPS = [0, 512, 1000, 2000, 4000, 8000, 16000, 33000, 66000, 100000, 143000]
PYTHIA_DICHT = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 13000,
                16000, 20000, 25000, 33000, 43000, 53000, 66000, 83000, 100000, 120000, 143000]


def effektiver_rang(E):
    lam = np.linalg.svd(E - E.mean(axis=0), compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def korrelationsdimension(P, n_skalen=8):
    """Lokale Steigung von log C(r) gegen log r (Grassberger-Procaccia), Sehnenabstand normierter Vektoren."""
    P = np.asarray(P, dtype=np.float64)
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    d = pdist(P)
    d = d[d > 0]
    r = np.quantile(d, np.geomspace(0.01, 0.5, n_skalen + 1))
    C = np.array([(d < ri).mean() for ri in r])
    ok = C > 0
    steigung = np.diff(np.log(C[ok])) / np.diff(np.log(r[ok]))
    return {"radien": r[ok].tolist(), "dimension": steigung.tolist()}


@torch.no_grad()
def messe_hf_modell(model, seqs, vocab, device="cpu", batch=8, n_drehungen=10, seed=0, n_ausreisser=3,
                    n_woerter=2000):
    rng = np.random.default_rng(seed)
    model = model.to(device).eval()
    n_layer, d = model.config.num_hidden_layers, model.config.hidden_size
    mitte = n_layer // 2
    seqs = np.asarray(seqs, dtype=np.int64)
    summe, summe0, zeilen, zeilen0 = np.zeros(n_layer), np.zeros(n_layer), 0, 0
    kontrolle = None
    mitte_strom = []
    for i in range(0, len(seqs), batch):
        x = torch.as_tensor(seqs[i:i + batch], device=device)
        hs = model(x, output_hidden_states=True).hidden_states[:n_layer]  # Eingang jedes Blocks
        for l, h in enumerate(hs):
            h = h.double().cpu().numpy()
            kz = basis.zeilen_kurtosis(h.reshape(-1, d)).reshape(h.shape[:2])
            summe[l] += kz.sum()
            summe0[l] += kz[:, 1:].sum()
            if l == mitte:
                mitte_strom.append(h.astype(np.float32))
        zeilen += x.numel()
        zeilen0 += x.shape[0] * (x.shape[1] - 1)
        if kontrolle is None:  # Drehungskontrolle auf dem ersten Batch
            kontrolle = [float(np.mean(basis.kurtosis_kontrolle(h.double().cpu().numpy().reshape(-1, d), n_drehungen, rng)))
                         for h in hs]
    strom = np.concatenate(mitte_strom).reshape(-1, d)
    tokens = seqs.reshape(-1)
    V = model.get_input_embeddings().weight.shape[0]
    freq = np.bincount(tokens, minlength=V)

    # V3: Ausreißerdimensionen (größter mittlerer Betrag in der mittleren Schicht) gegen Frequenz
    dims = np.argsort(-np.abs(strom).mean(axis=0))[:n_ausreisser]
    betrag = np.abs(strom[:, dims]).sum(axis=1)
    typen = np.unique(tokens)
    mittel_betrag = np.array([betrag[tokens == t].mean() for t in typen])
    v3 = spearmanr(mittel_betrag, freq[typen])

    # E4 (explorativ): Reflex – Participation Ratio kontextueller Repräsentationen gegen log-Frequenz
    pr = basis.participation_ratio(strom)
    haeufig = typen[freq[typen] >= 5]
    e4 = spearmanr([pr[tokens == t].mean() for t in haeufig], np.log(freq[haeufig])) if len(haeufig) >= 3 else None

    # E1/E2 (explorativ): Embedding der Wortliste
    W = model.get_input_embeddings().weight.double().cpu().numpy()
    woerter = wortliste(vocab, freq, n_woerter)
    E = W[woerter]
    erg = {
        "kurtosis": (summe / zeilen).tolist(),
        "kurtosis_ohne_pos0": (summe0 / zeilen0).tolist(),
        "kontrolle": kontrolle,
        "ausreisser_dims": dims.tolist(),
        "v3_rho": float(v3.statistic), "v3_p": float(v3.pvalue),
        "e4_rho": float(e4.statistic) if e4 is not None else None,
        "effektiver_rang": effektiver_rang(W),
        "korrelationsdimension": korrelationsdimension(E) if len(E) >= 3 else None,
        "n_woerter": len(woerter),
    }
    return erg, E


def lauf_vorstudie(modell_name, revisionen, seqs_pfad, aus_ordner, device="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seqs = np.load(seqs_pfad)["A"]
    vocab = AutoTokenizer.from_pretrained(modell_name).get_vocab()
    ordner = Path(aus_ordner) / modell_name.replace("/", "__")
    ordner.mkdir(parents=True, exist_ok=True)
    for rev in revisionen:
        model = AutoModelForCausalLM.from_pretrained(modell_name, revision=rev)
        erg, E = messe_hf_modell(model, seqs, vocab, device=device)
        (ordner / f"{rev}.json").write_text(json.dumps({**erg, "modell": modell_name, "revision": rev}, indent=2))
        np.save(ordner / f"{rev}_woerter.npy", E)
        del model


def _schritt(rev):
    return int(rev[4:]) if rev.startswith("step") else 10 ** 9


def vorstudie_bericht(aus_ordner, v_schwelle=1.0):
    aus = Path(aus_ordner)
    zeilen = ["# Vorstudie: Genealogie der Achsen", "",
              f"V-Schwelle (Festsetzung aus PRAEREG_8): {v_schwelle}", ""]
    for ordner in sorted(p for p in aus.iterdir() if p.is_dir()):
        laeufe = sorted((json.loads(f.read_text()) for f in ordner.glob("*.json")), key=lambda e: _schritt(e["revision"]))
        if not laeufe:
            continue
        zeilen += [f"## {ordner.name}", "",
                   "| Revision | Kurtosis | ohne Pos. 0 | Kontrolle | V1 | V3 ρ (p) |", "|---|---|---|---|---|---|"]
        for e in laeufe:
            k, k0, c = np.mean(e["kurtosis"]), np.mean(e["kurtosis_ohne_pos0"]), np.mean(e["kontrolle"])
            v1 = "ja" if k0 - max(e["kontrolle"]) > v_schwelle else "nein"
            zeilen.append(f"| {e['revision']} | {k:.3f} | {k0:.3f} | {c:.3f} | {v1} | {e['v3_rho']:.3f} ({e['v3_p']:.3g}) |")
        if len(laeufe) > 1:
            erst = laeufe[0]
            v2 = abs(np.mean(erst["kurtosis_ohne_pos0"]) - np.mean(erst["kontrolle"])) <= v_schwelle and any(
                np.mean(e["kurtosis_ohne_pos0"]) - np.mean(e["kontrolle"]) > v_schwelle for e in laeufe[1:])
            zeilen += ["", f"V2 (erster Checkpoint auf Kontrollniveau, danach Anstieg): **{'ja' if v2 else 'nein'}**"]
        zeilen += ["", "### Weitere Maße (explorativ, nicht präregistriert entscheidend)", ""]
        mats = [np.load(ordner / f"{e['revision']}_woerter.npy") for e in laeufe]
        for (e1, m1), (e2, m2) in zip(zip(laeufe, mats), zip(laeufe[1:], mats[1:])):
            rho = spearmanr(pdist(m1, "cosine"), pdist(m2, "cosine")).statistic
            zeilen.append(f"- E1 Nachträglichkeit {e1['revision']} → {e2['revision']}: ρ = {rho:.3f}; "
                          f"effektiver Rang {e1['effektiver_rang']:.1f} → {e2['effektiver_rang']:.1f}")
        for e in laeufe:
            kd = e["korrelationsdimension"]
            dims = ", ".join(f"{v:.2f}" for v in kd["dimension"]) if kd else "–"
            zeilen.append(f"- E2/E4 {e['revision']}: Korrelationsdimension je Skala [{dims}]; E4 ρ = {e['e4_rho']}")
        zeilen += ["- E3 Knotenlinie: Verlauf der Kurtosis-Spalte oben (dichtes Raster mit --dicht).", ""]
    text = "\n".join(zeilen) + "\n"
    (aus / "bericht.md").write_text(text, encoding="utf-8")
    return text
```

- [ ] **Step 4: Tests laufen lassen, Erfolg prüfen**

Run: `venv/bin/python -m pytest tests/test_vorstudie.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add chroma/vorstudie.py tests/test_vorstudie.py
git commit -m "Chroma: Vorstudie (Genealogie der Achsen) an HF-Modellen

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- chroma/vorstudie.py tests/test_vorstudie.py
```

---

### Task 10: CLI und Selbsttest

**Files:**
- Create: `chroma/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: alles Vorige.
- Produces: `venv/bin/python -m chroma <befehl>` mit den Befehlen `korpus`, `train`, `messen`, `auswerten`, `register`, `simlex`, `vorstudie`, `vorstudie-bericht`, `selbsttest`. `selbsttest(wurzel=None) -> None` (wirft bei Fehlern).

- [ ] **Step 1: Failing test schreiben**

`tests/test_cli.py`:

```python
from chroma.__main__ import selbsttest


def test_selbsttest(tmp_path):
    selbsttest(tmp_path)
    assert (tmp_path / "haupt" / "bericht.md").exists()
    assert (tmp_path / "haupt" / "register.md").exists()
    assert (tmp_path / "vorstudie" / "bericht.md").exists()
```

- [ ] **Step 2: Test laufen lassen, Fehlschlag prüfen**

Run: `venv/bin/python -m pytest tests/test_cli.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'chroma.__main__'`

- [ ] **Step 3: CLI implementieren**

`chroma/__main__.py`:

```python
"""
python -m chroma <befehl>

  korpus             WikiText-103 laden, Tokenizer trainieren, Splits ablegen
  train              einen Arm mit einem Seed trainieren (Loss-Stufen-Checkpoints)
  messen             alle Maße für einen Checkpoint
  auswerten          Entscheidungsregeln über alle Arme und Seeds (bericht.md)
  register           Register „Begriffsbedarf“ aus den Clustern einer Stufe
  simlex             SimLex-999 herunterladen
  vorstudie          V1–V3, E1–E4 an GPT-2/Pythia
  vorstudie-bericht  Bericht der Vorstudie
  selbsttest         Offline-Durchlauf mit winzigen Modellen (prüft nur den Code)
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def _geraet(name):
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def cmd_korpus(a):
    from .korpus import baue_korpus

    print(json.dumps(baue_korpus(a.ordner, vocab_size=a.vocab), indent=2))


def cmd_train(a):
    from .training import TrainConfig, trainiere

    stufen = tuple(float(s) for s in a.stufen.split(",")) if a.stufen else ()
    tc = TrainConfig(arm=a.arm, seed=a.seed, lr=a.lr, batch_size=a.batch, max_tokens=a.max_tokens,
                     eval_alle=a.eval_alle, loss_stufen=stufen)
    trainiere(a.korpus, Path(a.haupt) / a.arm / f"seed{a.seed}", tc, device=_geraet(a.geraet))


def cmd_messen(a):
    from .messen import MessConfig, messe_checkpoint

    ckpt = Path(a.ckpt)
    messe_checkpoint(ckpt, a.korpus, ckpt.with_name(f"messung_{ckpt.stem}.json"), MessConfig(seed=a.seed),
                     device=_geraet(a.geraet), simlex_pfad=a.simlex)


def cmd_auswerten(a):
    from .auswertung import bericht

    erg = bericht(a.haupt, delta=a.delta, schwelle_a0=a.schwelle_a0, schwelle_a=a.schwelle_a)
    print(json.dumps(erg, indent=2, default=str))


def cmd_register(a):
    from .korpus import lade_frequenz, lade_tokenizer, wortliste
    from .messung.register import schreibe_register, stabile_cluster

    name = f"messung_stufe_{float(a.stufe):.2f}_cluster.npy"
    dateien = sorted(Path(a.haupt).glob(f"*/seed*/{name}"))
    labels = [np.load(f) for f in dateien]
    tok = lade_tokenizer(a.korpus)
    woerter = wortliste(tok.get_vocab(), lade_frequenz(a.korpus), len(labels[0]))
    text = [tok.decode([w]) for w in woerter]
    schreibe_register(stabile_cluster(labels), text, Path(a.haupt) / "register.md")


def cmd_simlex(a):
    from .messung.normen import lade_simlex_herunter

    print(lade_simlex_herunter(a.ziel))


def cmd_vorstudie(a):
    from .vorstudie import PYTHIA_DICHT, PYTHIA_STEPS, lauf_vorstudie

    if a.revisionen == "pythia":
        revs = [f"step{s}" for s in PYTHIA_STEPS]
    elif a.revisionen == "pythia-dicht":
        revs = [f"step{s}" for s in PYTHIA_DICHT]
    else:
        revs = a.revisionen.split(",")
    lauf_vorstudie(a.modell, revs, a.seqs, a.aus, device=_geraet(a.geraet))


def cmd_vorstudie_bericht(a):
    from .vorstudie import vorstudie_bericht

    print(vorstudie_bericht(a.aus))


def selbsttest(wurzel=None):
    """Offline-Durchlauf: synthetisches Korpus → vier Arme × zwei Seeds → Messung → Auswertung →
    Register → Vorstudie an einem winzigen Zufalls-GPT-2. Prüft nur den Code, nichts davon wird gedeutet."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    from .arme import ARME
    from .auswertung import bericht
    from .korpus import baue_korpus, lade_frequenz, lade_tokenizer, wortliste
    from .messen import MessConfig, messe_checkpoint
    from .messung.register import schreibe_register, stabile_cluster
    from .training import TrainConfig, trainiere
    from .vorstudie import messe_hf_modell, vorstudie_bericht

    wurzel = Path(wurzel or tempfile.mkdtemp(prefix="chroma_selbsttest_"))
    rng = np.random.default_rng(0)
    woerter = ["haus", "baum", "katze", "hund", "stein", "fluss", "berg", "licht", "farbe", "zeit", "art", "folge"]
    def texte(n):
        return [" ".join(rng.choice(woerter, size=12)) + "." for _ in range(n)]
    baue_korpus(wurzel, splits={"train": texte(600), "val": texte(80), "test": texte(300)}, vocab_size=400,
                quelle="selbsttest")
    haupt = wurzel / "haupt"
    mc = MessConfig(n_woerter=10, n_paare=30, n_kontexte=4, min_kontexte=2, fenster=16, nach=4, n_anker=6,
                    k_nachbarn=3, n_fenster_strom=16, schicht_strom=1, n_cluster=3, batch=64)
    for arm in ARME:
        for seed in range(2):
            tc = TrainConfig(arm=arm, seed=seed, lr=0.3 if arm == "A0" else 3e-3, batch_size=8, block_size=32,
                             n_layer=2, n_head=2, n_embd=16, max_tokens=8 * 32 * 20, eval_alle=10, eval_batches=2,
                             loss_stufen=(100.0,))
            lauf = haupt / arm / f"seed{seed}"
            trainiere(wurzel, lauf, tc, log=lambda *x: None)
            messe_checkpoint(lauf / "stufe_100.00.pt", wurzel, lauf / "messung_stufe_100.00.json", mc)
    bericht(haupt)
    labels = [np.load(f) for f in sorted(haupt.glob("*/seed*/messung_stufe_100.00_cluster.npy"))]
    tok = lade_tokenizer(wurzel)
    ids = wortliste(tok.get_vocab(), lade_frequenz(wurzel), len(labels[0]))
    schreibe_register(stabile_cluster(labels), [tok.decode([i]) for i in ids], haupt / "register.md")

    torch.manual_seed(0)
    hf = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=100, n_positions=32)).eval()
    vocab = {f"Ġ{w}": i for i, w in enumerate(woerter)}
    vordner = wurzel / "vorstudie" / "mini"
    vordner.mkdir(parents=True)
    for rev in ("step0", "step1"):
        erg, E = messe_hf_modell(hf, rng.integers(0, 100, size=(6, 32)), vocab, n_drehungen=2, n_woerter=10)
        (vordner / f"{rev}.json").write_text(json.dumps({**erg, "revision": rev}))
        np.save(vordner / f"{rev}_woerter.npy", E)
    vorstudie_bericht(wurzel / "vorstudie")
    print(f"Selbsttest bestanden: {wurzel}")


def main():
    p = argparse.ArgumentParser(prog="python -m chroma", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="befehl", required=True)

    s = sub.add_parser("korpus")
    s.add_argument("--ordner", required=True)
    s.add_argument("--vocab", type=int, default=8192)
    s.set_defaults(f=cmd_korpus)

    s = sub.add_parser("train")
    s.add_argument("--korpus", required=True)
    s.add_argument("--haupt", required=True, help="Ordner des Laufs, z. B. runs/chroma_pilot")
    s.add_argument("--arm", required=True, choices=["A0", "A", "Ba", "Bb"])
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--lr", type=float, required=True)
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--max-tokens", type=int, default=100_000_000)
    s.add_argument("--eval-alle", type=int, default=200)
    s.add_argument("--stufen", default="", help="Loss-Stufen, z. B. 5.0,4.5,4.0")
    s.add_argument("--geraet", default="auto")
    s.set_defaults(f=cmd_train)

    s = sub.add_parser("messen")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--korpus", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--simlex", default=None)
    s.add_argument("--geraet", default="auto")
    s.set_defaults(f=cmd_messen)

    s = sub.add_parser("auswerten")
    s.add_argument("--haupt", required=True)
    s.add_argument("--delta", type=float, default=0.05)
    s.add_argument("--schwelle-a0", type=float, default=2.0)
    s.add_argument("--schwelle-a", type=float, default=10.0)
    s.set_defaults(f=cmd_auswerten)

    s = sub.add_parser("register")
    s.add_argument("--haupt", required=True)
    s.add_argument("--korpus", required=True)
    s.add_argument("--stufe", required=True)
    s.set_defaults(f=cmd_register)

    s = sub.add_parser("simlex")
    s.add_argument("--ziel", default="runs/chroma_normen")
    s.set_defaults(f=cmd_simlex)

    s = sub.add_parser("vorstudie")
    s.add_argument("--modell", required=True, help="gpt2 oder EleutherAI/pythia-160m")
    s.add_argument("--revisionen", default="main", help="main | pythia | pythia-dicht | step0,step512,...")
    s.add_argument("--seqs", required=True, help="z. B. runs/lauf_wiki/korpus/seq__gpt2__N256.npz")
    s.add_argument("--aus", default="runs/chroma_vorstudie")
    s.add_argument("--geraet", default="auto")
    s.set_defaults(f=cmd_vorstudie)

    s = sub.add_parser("vorstudie-bericht")
    s.add_argument("--aus", default="runs/chroma_vorstudie")
    s.set_defaults(f=cmd_vorstudie_bericht)

    s = sub.add_parser("selbsttest")
    s.set_defaults(f=lambda a: selbsttest())

    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Test und Selbsttest laufen lassen, Erfolg prüfen**

Run: `venv/bin/python -m pytest tests/test_cli.py -v && venv/bin/python -m chroma selbsttest`
Expected: `1 passed`, dann `Selbsttest bestanden: /…/chroma_selbsttest_…`

- [ ] **Step 5: Gesamte Testsuite**

Run: `venv/bin/python -m pytest -v`
Expected: alle Tests grün (Tasks 1–10, 50 Tests)

- [ ] **Step 6: Commit**

```bash
git add chroma/__main__.py tests/test_cli.py
git commit -m "Chroma: CLI und Selbsttest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- chroma/__main__.py tests/test_cli.py
```

---

## Nach diesem Plan (nicht Teil der Implementierung)

1. `praereg/PRAEREG_8_achsen.md` schreiben (V1–V3 mit V-Schwelle, E1–E4 als explorativ) und **vor** der Vorstudie committen.
2. Vorstudie: `python -m chroma vorstudie --modell gpt2 --seqs runs/lauf_wiki/korpus/seq__gpt2__N256.npz` und `--modell EleutherAI/pythia-160m --revisionen pythia --seqs runs/lauf_wiki/korpus/seq__EleutherAI__pythia-160m__N256.npz`.
3. Korpus bauen, Pilot (1 Seed je Arm, Lernraten-Sweep nach Validierungs-Loss), Loss-Stufen kalibrieren.
4. `praereg/PRAEREG_9_chroma.md` committen, dann Hauptlauf.
