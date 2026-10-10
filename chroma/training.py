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


def _abbruch(lauf, tc, schritt, kurve, grund):
    """Schreibt abbruch.json und wirft FloatingPointError."""
    (lauf / "abbruch.json").write_text(json.dumps({
        "schritt": schritt,
        "train": asdict(tc),
        "kurve": kurve,
        "grund": grund
    }, indent=2))
    raise FloatingPointError(grund)


def trainiere(korpus_ordner, lauf_ordner, tc, device="cpu", log=print):
    # Arm-Validierung VOR Ordnererstellung
    if tc.arm not in ARME:
        raise ValueError(f"unbekannter Arm {tc.arm!r}, erlaubt: {list(ARME)}")

    lauf = Path(lauf_ordner)

    # Prüfe auf Altlasten aus früheren Läufen
    artefakte = ["fertig.json", "abbruch.json", "ende.pt", "kurve.json"]
    stufen_dateien = list(lauf.glob("stufe_*.pt")) if lauf.exists() else []

    if (lauf / "fertig.json").exists():
        raise FileExistsError(f"{lauf} ist bereits fertig trainiert; anderen Laufordner wählen")

    if any((lauf / f).exists() for f in artefakte) or stufen_dateien:
        raise FileExistsError(f"{lauf} enthält Artefakte aus einem früheren Lauf; anderen Laufordner wählen")

    lauf.mkdir(parents=True, exist_ok=True)
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
            _abbruch(lauf, tc, schritt, kurve, f"Loss nicht endlich bei Schritt {schritt} (lr={tc.lr})")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            _abbruch(lauf, tc, schritt, kurve, f"Gradienten nicht endlich bei Schritt {schritt}")
        opt.step()
        if schritt % tc.eval_alle == 0 or schritt == max_schritte:
            v = _val_loss(model, val_batches)
            if not torch.isfinite(torch.tensor(v)):
                _abbruch(lauf, tc, schritt, kurve, f"Validierungsverlust nicht endlich bei Schritt {schritt}")
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
