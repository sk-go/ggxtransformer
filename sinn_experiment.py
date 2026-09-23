#!/usr/bin/env python3
"""
sinn_experiment.py
Die Tripelstruktur des Sinns im Transformer: Attention Sinks als verdrängte Asynchronizität.

Bedingungen (Zugleichsein ist im Kontextfenster immer gegeben):
  A   kohärent      Art ✓  Folge ✓   zusammenhängender Text
  B   gewürfelt     Art ✓  Folge ✗   Wörter von A permutiert
  B2  satzgewürfelt Art ✓  Folge ~   Sätze von A permutiert (Grammatik bleibt)
  C   disparat      Art ✗  Folge ✓   Einzelsätze aus fremden Dokumenten
  D   Rauschen      Art ✗  Folge ✗   Zufallstokens nach Häufigkeit

Befehle:
  korpus     Texte laden und in <ordner>/korpus/texte.json sichern
  messen     Sequenzen pro Tokenizer bauen, Sink-Masse/Entropie/Loss messen
  analyse    H1 und H2 für ein Modell (eine Revision) auswerten
  dynamik    H3: Verlauf über Trainings-Checkpoints (z. B. Pythia)
  selbsttest Offline-Durchlauf mit winzigem Zufallsmodell (prüft nur den Code)
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

CONDITIONS = ["A", "B", "B2", "C", "D"]
LABELS = {
    "A": "A kohärent",
    "B": "B wortgewürfelt",
    "B2": "B2 satzgewürfelt",
    "C": "C disparat",
    "D": "D Rauschen",
}
PYTHIA_STEPS = [0, 512, 1000, 2000, 4000, 8000, 16000, 33000, 66000, 100000, 143000]


# ----------------------------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------------------------

def slug(name):
    return name.replace("/", "__")


def clean(text):
    """Überschriften und Kurzzeilen entfernen, Leerraum normalisieren."""
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if len(l.split()) >= 8]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9\"„(])")


def sentences(text):
    return [s for s in SENT_SPLIT.split(text) if 5 <= len(s.split()) <= 60]


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------------
# 1. Texte
# ----------------------------------------------------------------------------

def load_texts(args):
    n_docs = args.n + args.pool
    docs = []
    if args.textordner:
        files = sorted(Path(args.textordner).glob("**/*.txt"))
        random.Random(args.seed).shuffle(files)
        for f in files:
            t = clean(f.read_text(encoding="utf-8", errors="ignore"))
            if len(t) >= args.min_zeichen:
                docs.append(t)
            if len(docs) >= n_docs:
                break
    else:
        from datasets import load_dataset
        print("Hinweis: Wikipedia-Artikel können im Training der Modelle vorgekommen sein.\n"
              "Für den Hauptlauf besser eigene Texte ab 2024 per --textordner verwenden.")
        ds = load_dataset(args.hf_datensatz, args.hf_config, split="train", streaming=True)
        ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
        for row in ds:
            t = clean(row["text"])
            if len(t) >= args.min_zeichen:
                docs.append(t)
            if len(docs) >= n_docs:
                break
    if len(docs) < n_docs:
        sys.exit(f"Nur {len(docs)} geeignete Texte gefunden, benötigt: {n_docs}.")
    return {"a_docs": docs[: args.n], "pool_docs": docs[args.n:], "seed": args.seed}


def cmd_korpus(args):
    out = Path(args.ordner) / "korpus" / "texte.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        sys.exit(f"{out} existiert bereits (--force zum Überschreiben).")
    for old in out.parent.glob("seq__*.npz"):  # alte Sequenzen passen nicht mehr zu neuen Texten
        old.unlink()
    texts = load_texts(args)
    out.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    print(f"{len(texts['a_docs'])} A-Texte + {len(texts['pool_docs'])} Pool-Texte -> {out}")


# ----------------------------------------------------------------------------
# 2. Sequenzbau (pro Tokenizer, weil GPT-2 und Pythia verschieden tokenisieren)
# ----------------------------------------------------------------------------

def _fit(ids, extras, enc, L):
    """Mit Zusatzmaterial auffüllen und exakt auf L Tokens schneiden."""
    for t in extras:
        if len(ids) >= L:
            break
        ids = ids + enc(" " + t)
    return ids[:L] if len(ids) >= L else None


def build_sequences(tok, texts, N, seed):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    L = N - 1  # Platz für das feste Start-Token
    bos = tok.eos_token_id
    special = set(tok.all_special_ids)

    def enc(s):
        return tok(s, add_special_tokens=False)["input_ids"]

    seqs = {c: [] for c in CONDITIONS}
    unigram = {}

    for doc in texts["a_docs"]:
        ids = enc(doc)
        if len(ids) < L + 20:
            continue
        a = ids[:L]
        text_a = tok.decode(a)
        rest = tok.decode(ids[L:])

        words = text_a.split()
        rng.shuffle(words)
        rest_words = rest.split()
        rng.shuffle(rest_words)
        b = _fit(enc(" ".join(words)), rest_words, enc, L)

        # Angeschnittene Satzfragmente an den Rändern verwerfen
        sents = sentences(text_a)
        if sents and not sents[-1].rstrip().endswith((".", "!", "?")):
            sents = sents[:-1]
        sents = sents or [text_a]
        rng.shuffle(sents)
        rest_sents = sentences(rest)[1:]
        rng.shuffle(rest_sents)
        b2 = _fit(enc(" ".join(sents)), rest_sents + rest_words, enc, L)

        if b is None or b2 is None:
            continue
        seqs["A"].append(a)
        seqs["B"].append(b)
        seqs["B2"].append(b2)
        for t in ids:
            if t not in special:
                unigram[t] = unigram.get(t, 0) + 1

    n = len(seqs["A"])
    if n == 0:
        sys.exit("Keine A-Sequenz gebaut, Texte zu kurz?")

    # C: Sätze aus paarweise verschiedenen Pool-Dokumenten
    by_doc = [s for s in (sentences(d) for d in texts["pool_docs"]) if s]
    for _ in range(n):
        ids = []
        for d in rng.sample(range(len(by_doc)), len(by_doc)):
            ids += enc((" " if ids else "") + rng.choice(by_doc[d]))
            if len(ids) >= L:
                break
        if len(ids) < L:
            sys.exit("Pool zu klein für Bedingung C, --pool erhöhen.")
        seqs["C"].append(ids[:L])

    # D: Zufallstokens nach Unigramm-Häufigkeit der A-Texte
    vocab = np.array(list(unigram.keys()))
    p = np.array(list(unigram.values()), dtype=float)
    p /= p.sum()
    for _ in range(n):
        seqs["D"].append(nrng.choice(vocab, size=L, p=p).tolist())

    return {c: np.array([[bos] + s for s in seqs[c]], dtype=np.int64) for c in CONDITIONS}


def get_sequences(root, model_name, tok, N, seed):
    kdir = Path(root) / "korpus"
    path = kdir / f"seq__{slug(model_name)}__N{N}.npz"
    if path.exists():
        data = np.load(path)
        return {c: data[c] for c in CONDITIONS}
    texts_path = kdir / "texte.json"
    if not texts_path.exists():
        sys.exit(f"{texts_path} fehlt, zuerst 'korpus' ausführen.")
    texts = json.loads(texts_path.read_text(encoding="utf-8"))
    seqs = build_sequences(tok, texts, N, seed)
    np.savez_compressed(path, **seqs)
    with open(kdir / f"beispiele__{slug(model_name)}.txt", "w", encoding="utf-8") as f:
        for c in CONDITIONS:
            for row in seqs[c][:2]:
                f.write(f"=== {LABELS[c]} ===\n{tok.decode(row[1:])[:600]}\n\n")
    print(f"{len(seqs['A'])} Sequenzen pro Bedingung gebaut -> {path}")
    return seqs


# ----------------------------------------------------------------------------
# 3. Messung
# ----------------------------------------------------------------------------

def measure(model, x_all, skip, batch, device):
    """Liefert Sink-Masse [n, Schichten, Köpfe], Ausgabe-Entropie [n], Loss [n]."""
    import torch
    sinks, ents, losses = [], [], []
    with torch.no_grad():
        for i in range(0, len(x_all), batch):
            x = torch.as_tensor(x_all[i:i + batch], device=device)
            o = model(input_ids=x, output_attentions=True)
            if not o.attentions or o.attentions[0] is None:
                raise RuntimeError("Keine Attention-Gewichte, Modell mit attn_implementation='eager' laden.")
            # Anteil der Aufmerksamkeit auf Token 0, gemittelt über Query-Positionen >= skip
            sink = torch.stack([a[:, :, skip:, 0].float().mean(-1) for a in o.attentions], dim=1)
            logp = o.logits.float().log_softmax(-1)
            ent = -(logp.exp() * logp).sum(-1)[:, skip:-1].mean(-1)
            nll = -logp[:, :-1].gather(-1, x[:, 1:, None]).squeeze(-1)[:, skip:].mean(-1)
            sinks.append(sink.cpu().numpy())
            ents.append(ent.cpu().numpy())
            losses.append(nll.cpu().numpy())
            del o, logp
    return np.concatenate(sinks), np.concatenate(ents), np.concatenate(losses)


def run_measurement(model, seqs, out_path, meta, skip, batch, device):
    arrays = {}
    for c in CONDITIONS:
        t0 = time.time()
        s, e, l = measure(model, seqs[c], skip, batch, device)
        arrays[f"sink_{c}"], arrays[f"ent_{c}"], arrays[f"loss_{c}"] = s, e, l
        print(f"  {LABELS[c]:18s} Sink {s.mean():.3f}  Entropie {e.mean():.2f}  "
              f"Loss {l.mean():.2f}  ({time.time() - t0:.0f}s)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
    print(f"  -> {out_path}")


def cmd_messen(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = args.geraet or pick_device()
    tok = AutoTokenizer.from_pretrained(args.modell)
    seqs = get_sequences(args.ordner, args.modell, tok, args.N, args.seed)

    revs = args.revisionen or [None]
    if revs == ["pythia"]:
        revs = [f"step{s}" for s in PYTHIA_STEPS]
    for rev in revs:
        out = Path(args.ordner) / "ergebnisse" / f"{slug(args.modell)}__{rev or 'final'}.npz"
        if out.exists() and not args.force:
            print(f"Überspringe {out.name} (existiert)")
            continue
        print(f"{args.modell} @ {rev or 'final'} auf {device}")
        kw = {"attn_implementation": "eager", "dtype": torch.float32}
        if rev:
            kw["revision"] = rev
        model = AutoModelForCausalLM.from_pretrained(args.modell, **kw).to(device).eval()
        meta = {"modell": args.modell, "revision": rev, "N": args.N, "skip": args.skip,
                "n": int(len(seqs["A"])), "seed": args.seed, "datum": time.strftime("%Y-%m-%d %H:%M"),
                "torch": torch.__version__, "transformers": transformers.__version__}
        run_measurement(model, seqs, out, meta, args.skip, args.batch, device)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# 4. Statistik
# ----------------------------------------------------------------------------

def perm_test(x, y, n_perm, rng, chunk=500):
    """Einseitiger Permutationstest H: mean(y) > mean(x). x [n1,k], y [n2,k] -> p [k]."""
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)
    data = np.concatenate([x, y])
    n1, n2, n = len(x), len(y), len(x) + len(y)
    obs = y.mean(0) - x.mean(0)
    hits = np.zeros(data.shape[1])
    done = 0
    while done < n_perm:
        P = min(chunk, n_perm - done)
        idx = np.argsort(rng.random((P, n)), axis=1)
        M = np.full((P, n), -1.0 / n1)
        np.put_along_axis(M, idx[:, :n2], 1.0 / n2, axis=1)
        hits += (M @ data >= obs - 1e-12).sum(0)
        done += P
    return (1 + hits) / (1 + n_perm)


def bh(p, alpha=0.05):
    """Benjamini-Hochberg: boolsche Maske der signifikanten Tests."""
    p = np.asarray(p).ravel()
    order = np.argsort(p)
    thresh = alpha * np.arange(1, len(p) + 1) / len(p)
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    mask = np.zeros(len(p), bool)
    mask[order[:k]] = True
    return mask


def boot_ci(stat_fn, arrays, n_boot, rng):
    """Perzentil-Bootstrap über Sequenzen, jede Bedingung unabhängig resampelt."""
    vals = []
    for _ in range(n_boot):
        vals.append(stat_fn(*[a[rng.integers(0, len(a), len(a))] for a in arrays]))
    return np.nanpercentile(vals, [2.5, 97.5])


def centroid(delta):
    """Schichtgewichteter Schwerpunkt der positiven Anteile einer Differenzkarte."""
    w = np.clip(delta, 0, None).sum(1)
    return float((np.arange(len(w)) * w).sum() / w.sum()) if w.sum() > 0 else np.nan


def load_result(root, model_name, rev):
    path = Path(root) / "ergebnisse" / f"{slug(model_name)}__{rev or 'final'}.npz"
    if not path.exists():
        sys.exit(f"{path} fehlt, zuerst 'messen' ausführen.")
    d = np.load(path)
    return {k: d[k] for k in d.files if k != "meta"}, json.loads(str(d["meta"]))


# ----------------------------------------------------------------------------
# 5. Analyse H1 und H2
# ----------------------------------------------------------------------------

def analyse(res, meta, out_dir, n_perm, n_boot, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    S = {c: res[f"sink_{c}"] for c in CONDITIONS}          # [n, L, H]
    L, H = S["A"].shape[1:]
    lines = [f"# Auswertung: {meta['modell']} @ {meta.get('revision') or 'final'}", "",
             f"n = {meta['n']} Sequenzen pro Bedingung, N = {meta['N']}, skip = {meta['skip']}, "
             f"{L} Schichten × {H} Köpfe, {n_perm} Permutationen, {n_boot} Bootstraps.", ""]

    # Kontrolle und Übersicht
    lines += ["## Übersicht", "", "| Bedingung | Sink-Masse (95%-KI) | Ausgabe-Entropie | Loss |",
              "|---|---|---|---|"]
    for c in CONDITIONS:
        m = S[c].mean(axis=(1, 2))
        lo, hi = boot_ci(lambda a: a.mean(), [m], n_boot, rng)
        lines.append(f"| {LABELS[c]} | {m.mean():.4f} ({lo:.4f}–{hi:.4f}) | "
                     f"{res[f'ent_{c}'].mean():.3f} | {res[f'loss_{c}'].mean():.3f} |")
    lines += ["", "Kontrolle: Der Loss sollte in A am niedrigsten und in D am höchsten sein. "
              "Ein auffällig niedriger Loss in A deutet auf auswendig gelernte Texte hin.", ""]

    # H1: Sink-Masse steigt mit Asynchronizität
    lines += ["## H1: Asynchronizität erhöht die Sink-Masse", "",
              "| Vergleich | Differenz | 95%-KI | p (einseitig) |", "|---|---|---|---|"]
    for a, b in [("A", "B"), ("A", "B2"), ("A", "C"), ("A", "D"), ("B", "D"), ("C", "D")]:
        x, y = S[a].mean(axis=(1, 2)), S[b].mean(axis=(1, 2))
        p = perm_test(x[:, None], y[:, None], n_perm, rng)[0]
        lo, hi = boot_ci(lambda u, v: v.mean() - u.mean(), [x, y], n_boot, rng)
        lines.append(f"| {b} > {a} | {y.mean() - x.mean():+.4f} | {lo:+.4f} bis {hi:+.4f} | {p:.4f} |")
    lines.append("")

    # Kopfweise Differenzkarten mit BH-Korrektur
    mean = {c: S[c].mean(0) for c in CONDITIONS}
    deltas, sig = {}, {}
    rows = ["bedingung,schicht,kopf,delta,p,signifikant_bh"]
    lines += ["## Kopfweise Differenzen zu A (Benjamini-Hochberg, α = 0,05)", ""]
    for c in ["B", "B2", "C", "D"]:
        deltas[c] = mean[c] - mean["A"]
        p = perm_test(S["A"], S[c], n_perm, rng).reshape(L, H)
        sig[c] = bh(p).reshape(L, H)
        lines.append(f"- {LABELS[c]}: {int(sig[c].sum())} von {L * H} Köpfen signifikant erhöht")
        for l in range(L):
            for h in range(H):
                rows.append(f"{c},{l},{h},{deltas[c][l, h]:.6f},{p[l, h]:.6f},{int(sig[c][l, h])}")
    (out_dir / "koepfe.csv").write_text("\n".join(rows), encoding="utf-8")
    lines.append("")

    # H2: Dissoziation von Folge (B) und Art (C)
    def stats(a, b, c):
        db, dc = b.mean(0) - a.mean(0), c.mean(0) - a.mean(0)
        return centroid(db), centroid(dc), np.corrcoef(db.ravel(), dc.ravel())[0, 1]

    cb, cc, r = stats(S["A"], S["B"], S["C"])
    ci_diff = boot_ci(lambda a, b, c: (lambda s: s[1] - s[0])(stats(a, b, c)),
                      [S["A"], S["B"], S["C"]], n_boot, rng)
    ci_r = boot_ci(lambda a, b, c: stats(a, b, c)[2], [S["A"], S["B"], S["C"]], n_boot, rng)
    lines += ["## H2: Folge und Art treiben verschiedene Köpfe in den Sink", "",
              f"- Schichtschwerpunkt ΔB (Folge zerstört): {cb:.2f}",
              f"- Schichtschwerpunkt ΔC (Art zerstört): {cc:.2f}",
              f"- Differenz C − B: {cc - cb:+.2f} (95%-KI {ci_diff[0]:+.2f} bis {ci_diff[1]:+.2f}); "
              "Vorhersage: positiv",
              f"- Korrelation der Karten ΔB und ΔC: r = {r:.2f} (95%-KI {ci_r[0]:.2f} bis {ci_r[1]:.2f}); "
              "Vorhersage: schwach", "",
              "Schichten sind ab 0 gezählt. Das Bootstrap resampelt A und B unabhängig, obwohl B aus A "
              "abgeleitet ist; die Intervalle sind daher eher konservativ.", ""]

    # Grafiken
    vmax = max(m.max() for m in mean.values())
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(3.2 * len(CONDITIONS), 3.4), squeeze=False)
    for ax, c in zip(axes[0], CONDITIONS):
        im = ax.imshow(mean[c], vmin=0, vmax=vmax, cmap="viridis", aspect="auto")
        ax.set_title(LABELS[c], fontsize=9)
        ax.set_xlabel("Kopf")
        ax.set_ylabel("Schicht")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="Sink-Masse")
    fig.savefig(out_dir / "sink_karten.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    dmax = max(np.abs(d).max() for d in deltas.values()) or 1.0
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), squeeze=False)
    for ax, c in zip(axes[0], ["B", "B2", "C", "D"]):
        im = ax.imshow(deltas[c], vmin=-dmax, vmax=dmax, cmap="RdBu_r", aspect="auto")
        yy, xx = np.nonzero(sig[c])
        ax.scatter(xx, yy, s=6, c="k", marker="o")
        ax.set_title(f"Δ {LABELS[c]} − A", fontsize=9)
        ax.set_xlabel("Kopf")
        ax.set_ylabel("Schicht")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="Differenz der Sink-Masse")
    fig.savefig(out_dir / "differenzkarten.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    lines += ["## Grafiken", "", "![Sink-Karten](sink_karten.png)", "",
              "![Differenzkarten](differenzkarten.png)", "",
              "Punkte markieren Köpfe, die nach BH-Korrektur signifikant erhöht sind."]
    (out_dir / "bericht.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Bericht -> {out_dir / 'bericht.md'}")


def cmd_analyse(args):
    res, meta = load_result(args.ordner, args.modell, args.revision)
    out = Path(args.ordner) / "analyse" / f"{slug(args.modell)}__{args.revision or 'final'}"
    analyse(res, meta, out, args.permutationen, args.bootstraps, args.seed)


# ----------------------------------------------------------------------------
# 6. Dynamik H3
# ----------------------------------------------------------------------------

def dynamik(root, model_name, n_boot, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    files = list((Path(root) / "ergebnisse").glob(f"{slug(model_name)}__step*.npz"))
    if not files:
        sys.exit("Keine Checkpoint-Ergebnisse gefunden, zuerst 'messen --revisionen pythia'.")
    steps = sorted((int(f.stem.split("__step")[-1]), f) for f in files)
    rows = ["schritt," + ",".join(f"sink_{c}" for c in CONDITIONS) + ",D_minus_A,ki_unten,ki_oben"]
    curves = {c: [] for c in CONDITIONS}
    diff, lo_hi, xs = [], [], []
    for step, f in steps:
        d = np.load(f)
        m = {c: d[f"sink_{c}"].mean(axis=(1, 2)) for c in CONDITIONS}
        for c in CONDITIONS:
            curves[c].append(m[c].mean())
        dd = m["D"].mean() - m["A"].mean()
        ci = boot_ci(lambda a, b: b.mean() - a.mean(), [m["A"], m["D"]], n_boot, rng)
        diff.append(dd)
        lo_hi.append(ci)
        xs.append(step)
        rows.append(f"{step}," + ",".join(f"{curves[c][-1]:.6f}" for c in CONDITIONS)
                    + f",{dd:.6f},{ci[0]:.6f},{ci[1]:.6f}")

    out = Path(root) / "analyse" / f"{slug(model_name)}__dynamik"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dynamik.csv").write_text("\n".join(rows), encoding="utf-8")

    xplot = [max(x, 1) for x in xs]  # Schritt 0 auf der Log-Achse bei 1 zeigen
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for c in CONDITIONS:
        ax[0].plot(xplot, curves[c], marker="o", ms=3, label=LABELS[c])
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Trainingsschritt (0 bei 1 gezeigt)")
    ax[0].set_ylabel("mittlere Sink-Masse")
    ax[0].set_title("Genese des Sinks")
    ax[0].legend(fontsize=8)
    lo_hi = np.array(lo_hi)
    ax[1].plot(xplot, diff, marker="o", ms=3, color="k")
    ax[1].fill_between(xplot, lo_hi[:, 0], lo_hi[:, 1], alpha=0.25, color="k")
    ax[1].axhline(0, lw=0.8, color="grey")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Trainingsschritt (0 bei 1 gezeigt)")
    ax[1].set_ylabel("Sink(D) − Sink(A)")
    ax[1].set_title("Empfindlichkeit für Asynchronizität")
    fig.tight_layout()
    fig.savefig(out / "dynamik.png", dpi=150)
    plt.close(fig)
    print(f"Dynamik -> {out}")


def cmd_dynamik(args):
    dynamik(args.ordner, args.modell, args.bootstraps, args.seed)


# ----------------------------------------------------------------------------
# 7. Selbsttest (offline, winziges Zufallsmodell; prüft nur den Code)
# ----------------------------------------------------------------------------

def cmd_selbsttest(args):
    import tempfile
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast

    rng = random.Random(0)
    syll = ["ka", "lo", "mi", "ne", "ru", "ta", "so", "vi", "de", "pa", "gu", "fe"]
    words = ["".join(rng.choice(syll) for _ in range(rng.randint(1, 3))) for _ in range(400)]

    def doc():
        return " ".join(
            " ".join(rng.choice(words) for _ in range(rng.randint(8, 16))).capitalize() + "."
            for _ in range(60))

    docs = [doc() for _ in range(60)]
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(docs, trainers.BpeTrainer(vocab_size=600, special_tokens=["<|endoftext|>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<|endoftext|>")

    root = Path(tempfile.mkdtemp(prefix="sinn_selbsttest_"))
    (root / "korpus").mkdir()
    (root / "korpus" / "texte.json").write_text(
        json.dumps({"a_docs": docs[:12], "pool_docs": docs[12:], "seed": 0}), encoding="utf-8")

    N, skip = 64, 4
    seqs = get_sequences(root, "selbsttest", tok, N, 0)
    for c in CONDITIONS:
        assert seqs[c].shape[1] == N and (seqs[c][:, 0] == tok.eos_token_id).all(), c

    cfg = GPT2Config(vocab_size=len(tok), n_positions=128, n_embd=32, n_layer=2, n_head=2,
                     bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    for i, rev in enumerate(["step0", "step1000", None]):
        torch.manual_seed(i)
        model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").eval()
        meta = {"modell": "selbsttest", "revision": rev, "N": N, "skip": skip,
                "n": int(len(seqs["A"])), "seed": 0}
        out = root / "ergebnisse" / f"selbsttest__{rev or 'final'}.npz"
        run_measurement(model, seqs, out, meta, skip, 4, "cpu")

    res, meta = load_result(root, "selbsttest", None)
    analyse(res, meta, root / "analyse" / "selbsttest__final", 200, 100, 0)
    dynamik(root, "selbsttest", 100, 0)
    print(f"\nSelbsttest erfolgreich. Ausgaben in {root}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ordner", default="lauf", help="Arbeitsordner (Pilot und Hauptlauf getrennt halten)")
    ap.add_argument("--seed", type=int, default=1234)
    sub = ap.add_subparsers(dest="befehl", required=True)

    k = sub.add_parser("korpus")
    k.add_argument("--n", type=int, default=300, help="Sequenzen pro Bedingung")
    k.add_argument("--pool", type=int, default=600, help="zusätzliche Dokumente für Bedingung C")
    k.add_argument("--textordner", help="Ordner mit .txt-Dateien (ein Dokument pro Datei)")
    k.add_argument("--hf-datensatz", default="wikimedia/wikipedia")
    k.add_argument("--hf-config", default="20231101.en")
    k.add_argument("--min-zeichen", type=int, default=3000)
    k.add_argument("--force", action="store_true")
    k.set_defaults(func=cmd_korpus)

    m = sub.add_parser("messen")
    m.add_argument("--modell", default="gpt2")
    m.add_argument("--revisionen", nargs="*", help="z. B. step0 step512 ... oder 'pythia'")
    m.add_argument("--N", type=int, default=256)
    m.add_argument("--skip", type=int, default=8)
    m.add_argument("--batch", type=int, default=4)
    m.add_argument("--geraet", choices=["cpu", "cuda", "mps"])
    m.add_argument("--force", action="store_true")
    m.set_defaults(func=cmd_messen)

    a = sub.add_parser("analyse")
    a.add_argument("--modell", default="gpt2")
    a.add_argument("--revision")
    a.add_argument("--permutationen", type=int, default=5000)
    a.add_argument("--bootstraps", type=int, default=2000)
    a.set_defaults(func=cmd_analyse)

    d = sub.add_parser("dynamik")
    d.add_argument("--modell", default="EleutherAI/pythia-160m")
    d.add_argument("--bootstraps", type=int, default=2000)
    d.set_defaults(func=cmd_dynamik)

    s = sub.add_parser("selbsttest")
    s.set_defaults(func=cmd_selbsttest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
