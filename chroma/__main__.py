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
