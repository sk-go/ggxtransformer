#!/usr/bin/env python3
"""
gewoehnung_experiment.py
Ontogenese des Sinns: Entstehung der Tripelstruktur über das Training (PRAEREG_7_gewoehnung.md).
Misst den Vorhersage-Loss, Top-1 und die Superadditivität H_synthese über 11 Pythia-160m Checkpoints.
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from tripel_experiment import patch_pythia, pick_device, bootstrap_ci, perm_test_paired, load_corpus, gleiche_bedingungen


CHECKPOINTS = [
    "step0",
    "step512",
    "step1000",
    "step2000",
    "step4000",
    "step8000",
    "step16000",
    "step33000",
    "step66000",
    "step100000",
    "step143000"
]

CONDITIONS_TRIPEL = [
    ("T111", 1, 1, 1, "Voller Sinn"),
    ("T011", 0, 1, 1, "Ohne Art"),
    ("T101", 1, 0, 1, "Ohne Folge"),
    ("T110", 1, 1, 0, "Ohne Zugleichsein"),
    ("T100", 1, 0, 0, "Nur Art"),
    ("T010", 0, 1, 0, "Nur Folge"),
    ("T001", 0, 0, 1, "Nur Zugleichsein"),
    ("T000", 0, 0, 0, "Vollständige Leere")
]


def measure_checkpoint(model, tok, docs, device, batch_size=8, N=128):
    """Misst alle 8 Bedingungen für ein geladenes Modell."""
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    # Vor-Tokenisierung
    all_input_ids = []
    for d in docs:
        t = tok(d, return_tensors="pt", max_length=N, truncation=True)
        ids = t.input_ids[0]
        if len(ids) == N:
            all_input_ids.append(ids)
    all_input_ids = torch.stack(all_input_ids).to(device)
    n_seq = len(all_input_ids)
    
    results = {}
    
    for tag, art_on, folge_on, zugleich_on, desc in CONDITIONS_TRIPEL:
        restore = patch_pythia(model, art=bool(art_on), folge=bool(folge_on), zugleich=bool(zugleich_on))
        
        doc_losses = []
        doc_top1 = []
        
        with torch.no_grad():
            for i in range(0, n_seq, batch_size):
                b_ids = all_input_ids[i:i+batch_size]
                labels = b_ids[:, 1:].clone()
                inp = b_ids[:, :-1]
                
                out = model(input_ids=inp)
                logits = out.logits
                
                loss_per_token = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss_per_seq = loss_per_token.view(b_ids.size(0), -1).mean(dim=1)
                doc_losses.extend(loss_per_seq.cpu().numpy().tolist())
                
                preds = logits.argmax(dim=-1)
                top1_per_seq = (preds == labels).float().mean(dim=1)
                doc_top1.extend(top1_per_seq.cpu().numpy().tolist())
                
        restore()
        
        results[tag] = {
            "loss_mean": float(np.mean(doc_losses)),
            "top1_mean": float(np.mean(doc_top1)),
            "doc_losses": doc_losses,
            "doc_top1": doc_top1
        }
        
    # Extra: Sink-Masse bei T111 messen
    sink_vals = []
    with torch.no_grad():
        for i in range(0, min(n_seq, 20), batch_size): # 20 Sequenzen reichen für robusten Sink-Mittelwert
            b_ids = all_input_ids[i:i+batch_size]
            out = model(input_ids=b_ids, output_attentions=True)
            # Attention auf Token 0 ab Position 4 gemittelt über Köpfe und Schichten.
            # Zufallsniveau (uniform kausal) ist daher mean(1/(i+1)), i = 4..N-1, nicht 1/N.
            for layer_attn in out.attentions:
                # layer_attn: [B, H, L, L]
                s = layer_attn[:, :, 4:, 0].mean().item()
                sink_vals.append(s)
    results["sink_token0"] = float(np.mean(sink_vals)) if sink_vals else 0.0
    
    return results


def run_gewoehnung(n_docs=100, N=128, out_dir="lauf_gewoehnung", device=None):
    device = device or pick_device()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "ergebnisse").mkdir(parents=True, exist_ok=True)
    (out_path / "analyse").mkdir(parents=True, exist_ok=True)
    
    docs = load_corpus(n_docs=n_docs, N=N)
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    
    summary = []
    
    print("\n" + "="*70)
    print("ONTOGENESE DES SINNS: PYTHIA-160M ÜBER 11 CHECKPOINTS")
    print(f"Device: {device} | Docs: {len(docs)} | Token-Länge: {N}")
    print("="*70)
    
    for ckpt in CHECKPOINTS:
        t0 = time.time()
        print(f"\n>>> Lade Checkpoint: {ckpt} ...")
        kw = {"attn_implementation": "eager", "dtype": torch.float32, "revision": ckpt}
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m", **kw).to(device).eval()
        
        res = measure_checkpoint(model, tok, docs, device, batch_size=8, N=N)
        dt = time.time() - t0
        
        # Synthese-Berechnung
        # Delta_voll = L(T000) - L(T111)
        # Summe_iso = (L(T000) - L(T100)) + (L(T000) - L(T010)) + (L(T000) - L(T001))
        # H_syn = Delta_voll - Summe_iso
        l_000 = res["T000"]["loss_mean"]
        l_111 = res["T111"]["loss_mean"]
        l_100 = res["T100"]["loss_mean"]
        l_010 = res["T010"]["loss_mean"]
        l_001 = res["T001"]["loss_mean"]
        
        delta_voll = l_000 - l_111
        g_art = l_000 - l_100
        g_folge = l_000 - l_010
        g_zugleich = l_000 - l_001
        sum_iso = g_art + g_folge + g_zugleich
        h_syn = delta_voll - sum_iso
        
        # Per-Doc H_syn für Konfidenzintervall
        doc_h = []
        for i in range(len(res["T111"]["doc_losses"])):
            d_000 = res["T000"]["doc_losses"][i]
            d_111 = res["T111"]["doc_losses"][i]
            d_100 = res["T100"]["doc_losses"][i]
            d_010 = res["T010"]["doc_losses"][i]
            d_001 = res["T001"]["doc_losses"][i]
            v = (d_000 - d_111) - ((d_000 - d_100) + (d_000 - d_010) + (d_000 - d_001))
            doc_h.append(v)
            
        m, lo, hi = bootstrap_ci(doc_h)
        p_val = perm_test_paired(doc_h, alternative="greater")
        
        entry = {
            "checkpoint": ckpt,
            "step": int(ckpt.replace("step", "")),
            "loss_T111": l_111,
            "loss_T011": res["T011"]["loss_mean"],
            "loss_T101": res["T101"]["loss_mean"],
            "loss_T110": res["T110"]["loss_mean"],
            "loss_T100": l_100,
            "loss_T010": l_010,
            "loss_T001": l_001,
            "loss_T000": l_000,
            "doc_h": doc_h,
            "gleiche": gleiche_bedingungen({k: res[k] for k, *_ in CONDITIONS_TRIPEL}, key="doc_losses"),
            "top1_T111": res["T111"]["top1_mean"],
            "top1_T000": res["T000"]["top1_mean"],
            "delta_voll": delta_voll,
            "sum_iso": sum_iso,
            "H_synthese": h_syn,
            "H_syn_ci_lo": lo,
            "H_syn_ci_hi": hi,
            "H_syn_p": p_val,
            "sink_token0": res["sink_token0"],
            "dauer_s": dt
        }
        summary.append(entry)
        
        print(f"[{ckpt}] Loss T111: {l_111:.3f} | Top1: {res['T111']['top1_mean']*100:.1f}% | "
              f"H_syn: {h_syn:+.3f} (95%-KI [{lo:+.3f}, {hi:+.3f}], p={p_val:.4f}) | "
              f"Sink0: {res['sink_token0']:.3f} ({dt:.1f}s)")
              
        # Detailergebnis des Checkpoints speichern
        (out_path / "ergebnisse" / f"{ckpt}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        
        del model
        if device == "mps":
            torch.mps.empty_cache()
            
    # Gesamt-Zusammenfassung speichern
    (out_path / "analyse" / "dynamik_tripel.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    # CSV schreiben
    csv_lines = ["step,checkpoint,loss_T111,loss_T011,loss_T101,loss_T110,loss_T000,top1_T111,delta_voll,sum_iso,H_synthese,ci_lo,ci_hi,sink_token0"]
    for s in summary:
        csv_lines.append(f"{s['step']},{s['checkpoint']},{s['loss_T111']:.4f},{s['loss_T011']:.4f},"
                         f"{s['loss_T101']:.4f},{s['loss_T110']:.4f},{s['loss_T000']:.4f},"
                         f"{s['top1_T111']:.4f},{s['delta_voll']:.4f},{s['sum_iso']:.4f},"
                         f"{s['H_synthese']:.4f},{s['H_syn_ci_lo']:.4f},{s['H_syn_ci_hi']:.4f},{s['sink_token0']:.4f}")
    (out_path / "analyse" / "dynamik_tripel.csv").write_text("\n".join(csv_lines), encoding="utf-8")
    
    # Plot erstellen
    generate_plot(summary, out_path / "analyse" / "ontogenese_plot.png")
    
    # Bericht erstellen
    generate_report(summary, out_path / "analyse" / "bericht.md", N=N)
    print(f"\nFertig! Bericht und Plots in {out_path / 'analyse'}")


def generate_plot(summary, out_file):
    steps = [s["step"] for s in summary]
    h_syn = [s["H_synthese"] for s in summary]
    ci_lo = [s["H_syn_ci_lo"] for s in summary]
    ci_hi = [s["H_syn_ci_hi"] for s in summary]
    loss_voll = [s["loss_T111"] for s in summary]
    sink = [s["sink_token0"] for s in summary]
    
    x = np.arange(len(steps))
    labels = [s["checkpoint"] for s in summary]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. H_synthese
    axes[0].plot(x, h_syn, 'o-', color='#1f77b4', lw=2.5, label='Superadditivität $H_{synthese}$ (Nats)')
    axes[0].fill_between(x, ci_lo, ci_hi, color='#1f77b4', alpha=0.2, label='95%-Bootstrap-KI')
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.7)
    axes[0].set_ylabel("Sinn-Synthese $H_{synthese}$ (Nats)", fontsize=11)
    axes[0].set_title("Ontogenese des Sinns: Entstehung der Superadditivität durch Gewohnheit", fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")
    
    # 2. Loss & Ablationen
    axes[1].plot(x, loss_voll, 's-', color='#2ca02c', lw=2, label='Voller Sinn ($T111$)')
    axes[1].plot(x, [s["loss_T011"] for s in summary], '--', color='#d62728', label='Ohne Art ($T011$)')
    axes[1].plot(x, [s["loss_T101"] for s in summary], '--', color='#ff7f0e', label='Ohne Folge ($T101$)')
    axes[1].plot(x, [s["loss_T110"] for s in summary], '--', color='#9467bd', label='Ohne Zugleichsein ($T110$)')
    axes[1].set_ylabel("Cross-Entropy Loss (Nats)", fontsize=11)
    axes[1].set_title("Entwicklung des Vorhersage-Loss unter Ablationen", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")
    
    # 3. Attention Sink
    axes[2].plot(x, sink, 'd-', color='#8c564b', lw=2, label='Attention Sink Masse (Token 0)')
    axes[2].set_ylabel("Sink-Masse (Token 0)", fontsize=11)
    axes[2].set_xlabel("Trainings-Checkpoint (EleutherAI/pythia-160m)", fontsize=11)
    axes[2].set_title("Kondensation des Attention Sinks als sedimentierte Entlastung", fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper left")
    
    plt.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def generate_report(summary, out_file, N=128):
    s0 = summary[0]
    s_end = summary[-1]
    zufall_sink = float(np.mean(1.0 / (np.arange(4, N) + 1)))

    def ki_text(lo, hi):
        return "schließt 0 aus" if (lo > 0 or hi < 0) else "enthält 0"

    # H2 präregistriert: H(143000) > H(0); gepaart über dieselben Dokumente
    d_h2 = np.array(s_end["doc_h"]) - np.array(s0["doc_h"])
    m2, lo2, hi2 = bootstrap_ci(d_h2)
    p2 = perm_test_paired(d_h2, alternative="greater")
    h2_ok = lo2 > 0 and p2 < 0.05

    lines = [
        "# Analysebericht: Gewöhnung über 11 Checkpoints (Pythia-160m)",
        "",
        "**Präregistrierung:** `PRAEREG_7_gewoehnung.md` (erst zusammen mit den Ergebnissen committet, "
        "daher nicht nachweislich vor dem Lauf festgelegt).",
        f"**Checkpoints:** {len(summary)} Stufen von `{s0['checkpoint']}` bis `{s_end['checkpoint']}`",
        "",
        "**Hinweis zur Konstruktion:** " + s_end["gleiche"] + " (Stand letzter Checkpoint). "
        "Weil mehrere Bedingungen bauartbedingt zusammenfallen, reduziert sich $H_{synthese}$ "
        "algebraisch auf weniger Terme; ein positiver Wert ist dann keine unabhängige Evidenz für Synthese.",
        "",
        "## 1. Präregistrierte Hypothesen",
        "",
        "### H-Gewohnheit 1: $H_{synthese}(\\text{step 0}) \\approx 0$",
        f"- $H_{{synthese}} = {s0['H_synthese']:+.4f}$, 95%-KI [{s0['H_syn_ci_lo']:+.4f}, {s0['H_syn_ci_hi']:+.4f}] "
        f"({ki_text(s0['H_syn_ci_lo'], s0['H_syn_ci_hi'])}), p = {s0['H_syn_p']:.4f}.",
        "- Die Präregistrierung nennt keine Schwelle für „≈ 0“; das Urteil ist deshalb nicht formal entscheidbar. "
        f"Zum Vergleich: Loss T111 = {s0['loss_T111']:.3f}, ln V ≈ 10,83 — bei Zufallsgewichten liegen alle Bedingungen am Zufallsniveau.",
        "",
        "### H-Gewohnheit 2: $H_{synthese}(\\text{step 143000}) > H_{synthese}(\\text{step 0})$",
        f"- $H_{{synthese}}(\\text{{{s_end['checkpoint']}}}) = {s_end['H_synthese']:+.4f}$, 95%-KI [{s_end['H_syn_ci_lo']:+.4f}, {s_end['H_syn_ci_hi']:+.4f}].",
        f"- Gepaarte Differenz: {m2:+.4f} (95%-KI {lo2:+.4f} bis {hi2:+.4f}), p = {p2:.4f} → "
        f"{'erfüllt' if h2_ok else 'nicht erfüllt'}.",
        "- Monotonie wurde präregistriert im Titel, aber ohne Test; siehe Verlaufstabelle.",
        "",
        "### H-Gewohnheit 3: Reihenfolge der Ablationsschäden (deskriptiv, keine Entscheidungsregel präregistriert)",
        "",
        "Δ = Loss(Ablation) − Loss(T111) in Nats. 3a erwartet, dass ΔZugleich von Beginn an dominiert; "
        "3b, dass ΔFolge und ΔArt erst ab mittleren Schritten wachsen.",
        "",
        "| Checkpoint | ΔArt (T011) | ΔFolge (T101) | ΔZugleich (T110) | größter Schaden |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for s in summary:
        d = {"Art": s["loss_T011"] - s["loss_T111"],
             "Folge": s["loss_T101"] - s["loss_T111"],
             "Zugleich": s["loss_T110"] - s["loss_T111"]}
        groesster = max(d, key=d.get) if max(d.values()) > 0.01 else "—"
        lines.append(f"| `{s['checkpoint']}` | {d['Art']:+.3f} | {d['Folge']:+.3f} | {d['Zugleich']:+.3f} | {groesster} |")

    lines += [
        "",
        "### H-Gewohnheit 4: Sink-Genese (step 0 ≈ Zufall, später > 0,10)",
        f"- Zufallsniveau bei uniformer kausaler Aufmerksamkeit über die gemessenen Positionen 4…{N-1}: **{zufall_sink:.4f}** "
        "(nicht 1/N).",
        f"- step 0: **{s0['sink_token0']:.4f}**; {s_end['checkpoint']}: **{s_end['sink_token0']:.4f}**.",
        f"- Erster Checkpoint über 0,10: "
        + next((f"`{s['checkpoint']}` ({s['sink_token0']:.3f})" for s in summary if s["sink_token0"] > 0.10), "keiner"),
        "",
        "## 2. Checkpoint-Verlaufstabelle",
        "",
        "| Checkpoint | Loss T111 | Top-1 T111 | $H_{synthese}$ (Nats) | 95%-Bootstrap-KI | p-Wert | Sink Token 0 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in summary:
        lines.append(f"| `{s['checkpoint']}` | {s['loss_T111']:.3f} | {s['top1_T111']*100:.1f}% | **{s['H_synthese']:+.3f}** | [{s['H_syn_ci_lo']:+.3f}, {s['H_syn_ci_hi']:+.3f}] | {s['H_syn_p']:.4f} | {s['sink_token0']:.3f} |")

    lines += ["", "## 3. Grafik", "", "![Verlauf](ontogenese_plot.png)", ""]
    out_file.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gewöhnungsexperiment: Ontogenese des Sinns über Pythia Checkpoints")
    parser.add_argument("--docs", type=int, default=100, help="Anzahl Dokumente (Default: 100)")
    parser.add_argument("--len", type=int, default=128, help="Token-Länge (Default: 128)")
    parser.add_argument("--out", type=str, default="lauf_gewoehnung", help="Ausgabeordner")
    args = parser.parse_args()
    
    run_gewoehnung(n_docs=args.docs, N=args.len, out_dir=args.out)
