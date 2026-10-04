#!/usr/bin/env python3
"""
tripel_experiment.py
Vollfaktorielle Ablations-Triade (PRAEREG_6_tripel.md).
Misst den Vorhersage-Loss und die Top-1/Top-5-Genauigkeit unter allen 8 Kombinationen
von Art (Q*K), Folge (Pos/Maske) und Zugleichsein (Value-Mischung) in GPT-2 und Pythia.
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bootstrap_ci(diffs, n_boot=2000, alpha=0.05, seed=1234):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = rng.choice(diffs, size=(n_boot, n), replace=True).mean(axis=1)
    lo = np.percentile(boots, 100 * (alpha / 2))
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(np.mean(diffs)), float(lo), float(hi)


def perm_test_paired(diffs, n_perm=5000, seed=1234, alternative="greater"):
    rng = np.random.default_rng(seed)
    diffs = np.array(diffs)
    obs = np.mean(diffs)
    n = len(diffs)
    signs = rng.choice([-1, 1], size=(n_perm, n))
    perm_means = (signs * diffs).mean(axis=1)
    if alternative == "greater":
        p = (np.sum(perm_means >= obs) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(perm_means <= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(perm_means) >= np.abs(obs)) + 1) / (n_perm + 1)
    return float(p)


def load_corpus(n_docs=100, N=128):
    path = Path("lauf_wiki/korpus/texte.json")
    if not path.exists():
        path = Path("lauf_c0/korpus/texte.json")
    texts = json.loads(path.read_text(encoding="utf-8"))
    docs = texts["a_docs"][:n_docs]
    return docs


def patch_gpt2(model, art=True, folge=True, zugleich=True):
    orig_forwards = []
    orig_wpe = model.transformer.wpe.weight.data.clone()
    if not folge:
        model.transformer.wpe.weight.data.zero_()
        
    for block in model.transformer.h:
        attn = block.attn
        orig_fn = attn.forward
        orig_forwards.append((attn, orig_fn))
        
        def make_forward(orig_f, attn_mod):
            def custom_forward(hidden_states, *args, **kwargs):
                query, key, value = attn_mod.c_attn(hidden_states).split(attn_mod.split_size, dim=2)
                B, L, _ = hidden_states.shape
                H = attn_mod.num_heads
                D = attn_mod.head_dim
                
                q = query.view(B, L, H, D).transpose(1, 2)
                k = key.view(B, L, H, D).transpose(1, 2)
                v = value.view(B, L, H, D).transpose(1, 2)
                
                # 1. Art
                if art:
                    scores = torch.matmul(q, k.transpose(-1, -2)) / (D ** 0.5)
                else:
                    scores = torch.zeros((B, H, L, L), device=q.device)
                    
                # 2. Folge (Kausale Maske)
                mask = torch.tril(torch.ones((L, L), device=q.device)).view(1, 1, L, L)
                scores = scores.masked_fill(mask == 0, -1e9)
                w = torch.softmax(scores, dim=-1)
                
                # 3. Zugleichsein
                if not zugleich:
                    w = torch.eye(L, device=q.device).view(1, 1, L, L).expand(B, H, L, L)
                    
                out = torch.matmul(w, v)
                out = out.transpose(1, 2).contiguous().view(B, L, H * D)
                out = attn_mod.c_proj(out)
                out = attn_mod.resid_dropout(out)
                return out, w
            return custom_forward
            
        attn.forward = make_forward(orig_fn, attn)

    def restore():
        for attn, orig_f in orig_forwards:
            attn.forward = orig_f
        model.transformer.wpe.weight.data.copy_(orig_wpe)
        
    return restore


def patch_pythia(model, art=True, folge=True, zugleich=True):
    orig_forwards = []
    hooks = []
    
    for layer in model.gpt_neox.layers:
        attn = layer.attention
        orig_fn = attn.forward
        orig_forwards.append((attn, orig_fn))
        
        if not art:
            def hook_qkv(m, inp, out):
                q, k, v = out.chunk(3, dim=-1)
                q = torch.zeros_like(q)
                return torch.cat([q, k, v], dim=-1)
            hooks.append(attn.query_key_value.register_forward_hook(hook_qkv))
            
        def make_forward(orig_f, attn_mod):
            def custom_forward(hidden_states, *args, **kwargs):
                if not folge and 'position_embeddings' in kwargs and kwargs['position_embeddings'] is not None:
                    cos, sin = kwargs['position_embeddings']
                    kwargs['position_embeddings'] = (torch.ones_like(cos), torch.zeros_like(sin))
                    
                if not zugleich:
                    qkv = attn_mod.query_key_value(hidden_states)
                    _, _, v = qkv.chunk(3, dim=-1)
                    out = attn_mod.dense(v)
                    return out, None
                    
                return orig_f(hidden_states, *args, **kwargs)
            return custom_forward
            
        attn.forward = make_forward(orig_fn, attn)

    def restore():
        for h in hooks:
            h.remove()
        for attn, orig_f in orig_forwards:
            attn.forward = orig_f
            
    return restore



CONDITIONS_TRIPEL = [
    ("T111", 1, 1, 1, "Voller Sinn (Art + Folge + Zugleichsein)"),
    ("T011", 0, 1, 1, "Ohne Art (Folge + Zugleichsein)"),
    ("T101", 1, 0, 1, "Ohne Folge (Art + Zugleichsein)"),
    ("T110", 1, 1, 0, "Ohne Zugleichsein (Art + Folge)"),
    ("T100", 1, 0, 0, "Nur Art"),
    ("T010", 0, 1, 0, "Nur Folge"),
    ("T001", 0, 0, 1, "Nur Zugleichsein"),
    ("T000", 0, 0, 0, "Vollständige Leere")
]


def run_experiment(model_name, n_docs=100, N=128, out_dir="lauf_tripel", device=None):
    device = device or pick_device()
    print(f"\n=======================================================")
    print(f"Tripel-Ablation: {model_name} auf {device}")
    print(f"=======================================================")

    tok = AutoTokenizer.from_pretrained(model_name)
    kw = {"attn_implementation": "eager", "dtype": torch.float32}
    if "pythia" in model_name:
        kw["revision"] = "step143000"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw).to(device).eval()

    docs = load_corpus(n_docs, N)
    print(f"Korpus geladen: {len(docs)} Texte, Länge N = {N}")

    # Sequenzen vorbereiten
    seq_list = []
    for doc in docs:
        ids = tok.encode(doc, add_special_tokens=False)
        if len(ids) >= N:
            seq_list.append(ids[:N])
        if len(seq_list) >= n_docs:
            break

    print(f"Evaluierungs-Sequenzen: {len(seq_list)}")

    results = {code: {"loss": [], "top1": [], "top5": []} for code, *_ in CONDITIONS_TRIPEL}

    patch_fn = patch_gpt2 if "gpt2" in model_name else patch_pythia

    for code, a, f, z, label in CONDITIONS_TRIPEL:
        t0 = time.time()
        restore = patch_fn(model, art=bool(a), folge=bool(f), zugleich=bool(z))
        
        try:
            with torch.no_grad():
                for seq in seq_list:
                    x = torch.tensor([seq], device=device)
                    targets = x[:, 1:]
                    
                    o = model(x)
                    logits = o.logits[:, :-1, :]
                    
                    # NLL
                    logp = logits.float().log_softmax(-1)
                    nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean().item()
                    
                    # Top-1
                    preds = logits.argmax(-1)
                    top1 = (preds == targets).float().mean().item()
                    
                    # Top-5
                    top5_idx = logits.topk(5, dim=-1).indices
                    top5 = (top5_idx == targets.unsqueeze(-1)).any(-1).float().mean().item()
                    
                    results[code]["loss"].append(nll)
                    results[code]["top1"].append(top1)
                    results[code]["top5"].append(top5)
        finally:
            restore()
            
        m_loss = np.mean(results[code]["loss"])
        m_top1 = np.mean(results[code]["top1"]) * 100
        print(f"  {code:5s} ({label:38s}) -> Loss: {m_loss:5.2f} | Top-1: {m_top1:4.1f}% ({time.time()-t0:.1f}s)")

    # Statistische Auswertung
    report = []
    report.append(f"# Auswertung: {model_name} (Tripelstruktur-Ablation)")
    report.append(f"\nDatum: {time.strftime('%Y-%m-%d %H:%M')}, n = {len(seq_list)} Sequenzen, N = {N}, Gerät: {device}\n")
    report.append("## 1. Vollfaktorielle Matrix ($2 \\times 2 \\times 2$)\n")
    report.append("| Bedingung | Art ($Q K^T$) | Folge (Pos) | Zugleichsein ($\\sum V$) | Loss (NLL) | Top-1 Acc | Top-5 Acc | Perplexität |")
    report.append("|---|---|---|---|---|---|---|---|")
    
    for code, a, f, z, label in CONDITIONS_TRIPEL:
        l = np.mean(results[code]["loss"])
        t1 = np.mean(results[code]["top1"]) * 100
        t5 = np.mean(results[code]["top5"]) * 100
        ppl = np.exp(l)
        report.append(f"| **{code}** | {a} | {f} | {z} | **{l:.2f}** | {t1:.1f}% | {t5:.1f}% | {ppl:.1f} |")

    report.append("\n## 2. Prüfung der Hypothesen (PRAEREG_6_tripel.md)\n")
    report.append("| Hypothese | Test / Differenz | 95%-KI | p-Wert | Bestätigt? |")
    report.append("|---|---|---|---|---|")

    # H-Art: T011 - T111
    d_art = np.array(results["T011"]["loss"]) - np.array(results["T111"]["loss"])
    m_art, lo_art, hi_art = bootstrap_ci(d_art)
    p_art = perm_test_paired(d_art, alternative="greater")
    h_art_ok = lo_art > 0 and p_art < 0.05
    report.append(f"| **H-Art (Inhärenz fehlt)** | T011 − T111 | {m_art:+.2f} ({lo_art:+.2f} bis {hi_art:+.2f}) | p = {p_art:.4f} | {'✓ JA' if h_art_ok else '✗ NEIN'} |")

    # H-Folge: T101 - T111
    d_folge = np.array(results["T101"]["loss"]) - np.array(results["T111"]["loss"])
    m_folge, lo_folge, hi_folge = bootstrap_ci(d_folge)
    p_folge = perm_test_paired(d_folge, alternative="greater")
    h_folge_ok = lo_folge > 0 and p_folge < 0.05
    report.append(f"| **H-Folge (Sukzession fehlt)** | T101 − T111 | {m_folge:+.2f} ({lo_folge:+.2f} bis {hi_folge:+.2f}) | p = {p_folge:.4f} | {'✓ JA' if h_folge_ok else '✗ NEIN'} |")

    # H-Zugleichsein: T110 - T111
    d_zugleich = np.array(results["T110"]["loss"]) - np.array(results["T111"]["loss"])
    m_zugleich, lo_zugleich, hi_zugleich = bootstrap_ci(d_zugleich)
    p_zugleich = perm_test_paired(d_zugleich, alternative="greater")
    h_zugleich_ok = lo_zugleich > 0 and p_zugleich < 0.05
    report.append(f"| **H-Zugleichsein (Gemeinschaft fehlt)** | T110 − T111 | {m_zugleich:+.2f} ({lo_zugleich:+.2f} bis {hi_zugleich:+.2f}) | p = {p_zugleich:.4f} | {'✓ JA' if h_zugleich_ok else '✗ NEIN'} |")

    # Rangordnung
    h_rang_ok = m_zugleich > m_folge > m_art
    report.append(f"| **Rangordnung der Schäden** | T110 > T101 > T011 | {m_zugleich:.2f} > {m_folge:.2f} > {m_art:.2f} | - | {'✓ JA' if h_rang_ok else '✗ NEIN'} |")

    # H-Synthese: Superadditivität
    # Wie viel Loss gewinnt man durch den vollen Sinn gegenüber der Leere T000?
    gewinn_voll = np.array(results["T000"]["loss"]) - np.array(results["T111"]["loss"])
    # Summe der Gewinne isolierter Komponenten
    gewinn_art_iso = np.array(results["T000"]["loss"]) - np.array(results["T100"]["loss"])
    gewinn_folge_iso = np.array(results["T000"]["loss"]) - np.array(results["T010"]["loss"])
    gewinn_zugleich_iso = np.array(results["T000"]["loss"]) - np.array(results["T001"]["loss"])
    summe_iso = gewinn_art_iso + gewinn_folge_iso + gewinn_zugleich_iso
    
    synergie = gewinn_voll - summe_iso
    m_syn, lo_syn, hi_syn = bootstrap_ci(synergie)
    p_syn = perm_test_paired(synergie, alternative="greater")
    h_syn_ok = lo_syn > 0 and p_syn < 0.05
    report.append(f"| **H-Synthese (Superadditivität)** | Voll − Summe(Einzel) | {m_syn:+.2f} ({lo_syn:+.2f} bis {hi_syn:+.2f}) | p = {p_syn:.4f} | {'✓ JA' if h_syn_ok else '✗ NEIN'} |")

    rep_text = "\n".join(report)
    slug_name = model_name.replace("/", "__")
    out_path = Path(out_dir) / "analyse" / slug_name / "bericht.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rep_text, encoding="utf-8")
    
    raw_path = Path(out_dir) / "ergebnisse" / f"{slug_name}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    print(f"\nBericht -> {out_path}")
    print("\n" + rep_text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modell", default="gpt2")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--ordner", default="lauf_tripel")
    args = ap.parse_args()
    run_experiment(args.modell, n_docs=args.n, N=args.N, out_dir=args.ordner)
