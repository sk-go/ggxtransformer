#!/usr/bin/env python3
"""
valenz_experiment.py
Führt das Valenz-Inkongruenz-Experiment (PRAEREG_5_valenz.md) durch.
Misst MLP-Aktivierungsnormen, Residual-Stream-Kosinusdrift, Sink-Masse und Loss
an den Minimalpaaren K1 (Konsistent), K2 (Paradox) und K3 (Neutral).
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
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


def perm_test_paired(diffs, n_perm=5000, seed=1234, alternative="two-sided"):
    """Paired permutation test (Vorzeichen-Permutation)."""
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
    else: # two-sided
        p = (np.sum(np.abs(perm_means) >= np.abs(obs)) + 1) / (n_perm + 1)
    return float(p)


def run_valenz_experiment(model_name, stimuli_file="korpus_valenz/stimuli.json", out_dir="lauf_valenz", device=None):
    device = device or pick_device()
    print(f"\n=======================================================")
    print(f"Starte Valenz-Experiment: {model_name} auf {device}")
    print(f"=======================================================")

    tok = AutoTokenizer.from_pretrained(model_name)
    kw = {"attn_implementation": "eager", "dtype": torch.float32}
    if "pythia" in model_name:
        kw["revision"] = "step143000"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw).to(device).eval()

    stimuli = json.loads(Path(stimuli_file).read_text(encoding="utf-8"))
    n_items = len(stimuli)
    print(f"Geladene Stimuli: {n_items} Minimalpaare")

    # Hook für MLP-Zwischenschichten
    mlp_activations = []
    def hook_fn(mod, inp, out):
        mlp_activations.append(out.detach())

    hooks = []
    if "gpt2" in model_name:
        for block in model.transformer.h:
            hooks.append(block.mlp.act.register_forward_hook(hook_fn))
    else:
        for block in model.gpt_neox.layers:
            hooks.append(block.mlp.act.register_forward_hook(hook_fn))

    conditions = ["K1", "K2", "K3"]
    # Messdaten-Container
    data = {c: {
        "mlp_norm_mid": [],      # Mittlere L2-Norm der MLPs Schichten 4-8 am Target
        "mlp_norm_all": [],      # Mittlere L2-Norm über alle Schichten
        "mlp_sparsity_mid": [],  # Sparsity (< 1e-4) in Schichten 4-8
        "residual_cos": [],      # Kosinus-Ähnlichkeit Schicht 11: h(t_start-1) vs h(t_end-1)
        "residual_cos_mid": [],  # Kosinus-Ähnlichkeit mittlere Schichten (4-8)
        "sink_target": [],       # Sink-Attention am Target-Token
        "sink_after": [],        # Sink-Attention am Token unmittelbar nach dem Target
        "loss_target": [],       # NLL des Target-Worts
        "ent_after": []          # Ausgabe-Entropie unmittelbar nach dem Target
    } for c in conditions}

    mid_layers = [4, 5, 6, 7, 8]

    for item in stimuli:
        prefix = item["prefix"]
        prefix_ids = tok.encode(prefix, add_special_tokens=False)
        t_start = len(prefix_ids)

        for c in conditions:
            sent = item["sentences"][c]
            toks = tok(sent, return_tensors="pt").to(device)
            input_ids = toks["input_ids"][0]
            
            # Finde Target-Token-Bereich
            target_word = item["targets"][{"K1": "K1_konsistent", "K2": "K2_paradox", "K3": "K3_neutral"}[c]]
            # Target mit Leerzeichen kodieren
            target_ids = tok.encode(" " + target_word, add_special_tokens=False)
            t_len = len(target_ids)
            t_end = t_start + t_len

            mlp_activations.clear()
            with torch.no_grad():
                o = model(**toks, output_attentions=True, output_hidden_states=True)

            # 1. MLP-Aktivierungsnorm am Target
            # mlp_activations ist Liste von 12 Tensoren [1, seq_len, hidden_mlp]
            mlp_mats = torch.stack(mlp_activations, dim=1)[0] # [layers, seq_len, hidden]
            target_mlp = mlp_mats[:, t_start:t_end, :] # [layers, t_len, hidden]
            target_l2 = torch.norm(target_mlp, p=2, dim=-1).mean(dim=-1).cpu().numpy() # [layers]
            
            norm_mid = float(target_l2[mid_layers].mean())
            norm_all = float(target_l2.mean())
            
            sparsity_mid = float((target_mlp[mid_layers] < 1e-4).float().mean().cpu().numpy())

            # 2. Residual Stream Kosinus-Drift
            # hidden_states: tuple von 13 Tensoren [1, seq_len, d_model] (inkl. Embedding)
            h_states = torch.stack(o.hidden_states[1:], dim=1)[0] # [12, seq_len, d_model]
            vec_before = h_states[:, t_start - 1, :]
            vec_after = h_states[:, t_end - 1, :]
            cos_sims = torch.cosine_similarity(vec_before, vec_after, dim=-1).cpu().numpy() # [12]
            
            cos_final = float(cos_sims[-1])
            cos_mid = float(cos_sims[mid_layers].mean())

            # 3. Sink Attention
            # attentions: 12 Tensoren [1, H, seq_len, seq_len]
            attns = torch.stack([a[0] for a in o.attentions], dim=0) # [12, H, seq_len, seq_len]
            # Sink-Attention auf pos 0
            sink_all_pos = attns[:, :, :, 0].mean(dim=(0, 1)).cpu().numpy() # [seq_len]
            
            sink_target = float(sink_all_pos[t_start:t_end].mean())
            sink_after = float(sink_all_pos[t_end]) if t_end < len(sink_all_pos) else float(sink_all_pos[-1])

            # 4. Loss & Entropie
            logp = o.logits[0].float().log_softmax(-1)
            # Loss für Target-Token
            target_nll = -logp[t_start - 1 : t_end - 1, :].gather(-1, input_ids[t_start:t_end].unsqueeze(-1)).squeeze(-1)
            loss_t = float(target_nll.mean().cpu().numpy())
            
            # Entropie nach Target
            ent = -(logp.exp() * logp).sum(-1)
            ent_after = float(ent[t_end - 1].cpu().numpy())

            data[c]["mlp_norm_mid"].append(norm_mid)
            data[c]["mlp_norm_all"].append(norm_all)
            data[c]["mlp_sparsity_mid"].append(sparsity_mid)
            data[c]["residual_cos"].append(cos_final)
            data[c]["residual_cos_mid"].append(cos_mid)
            data[c]["sink_target"].append(sink_target)
            data[c]["sink_after"].append(sink_after)
            data[c]["loss_target"].append(loss_t)
            data[c]["ent_after"].append(ent_after)

    for h in hooks:
        h.remove()

    # Statistische Auswertung
    report_lines = []
    report_lines.append(f"# Auswertung: {model_name} (Valenz-Inkongruenz)")
    report_lines.append(f"\nDatum: {time.strftime('%Y-%m-%d %H:%M')}, n = {n_items} Minimalpaare, Gerät: {device}\n")
    
    report_lines.append("## 1. Übersicht der Mittelwerte\n")
    report_lines.append("| Metrik | K1 (Konsistent) | K2 (Paradox) | K3 (Neutral) |")
    report_lines.append("|---|---|---|---|")
    
    metrics = [
        ("Loss am Target (Surprise)", "loss_target", "{:.2f}"),
        ("MLP-Norm Mid (Schichten 4–8)", "mlp_norm_mid", "{:.2f}"),
        ("MLP-Sparsity Mid (< 1e-4)", "mlp_sparsity_mid", "{:.3f}"),
        ("Residual Cosine Drift (Mid)", "residual_cos_mid", "{:.4f}"),
        ("Residual Cosine Drift (Final)", "residual_cos", "{:.4f}"),
        ("Sink-Masse am Target", "sink_target", "{:.4f}"),
        ("Sink-Masse nach Target (t+1)", "sink_after", "{:.4f}"),
        ("Ausgabe-Entropie (t+1)", "ent_after", "{:.2f}")
    ]
    
    for label, key, fmt in metrics:
        v1 = fmt.format(np.mean(data["K1"][key]))
        v2 = fmt.format(np.mean(data["K2"][key]))
        v3 = fmt.format(np.mean(data["K3"][key]))
        report_lines.append(f"| {label} | {v1} | {v2} | {v3} |")

    report_lines.append("\n## 2. Prüfung der Hypothesen (PRAEREG_5_valenz.md)\n")
    report_lines.append("| Hypothese | Vergleich | Differenz (95%-KI) | p-Wert | Bestätigt? |")
    report_lines.append("|---|---|---|---|---|")

    # V1: MLP Mid-Norm K2 vs K1
    diff_mlp = np.array(data["K2"]["mlp_norm_mid"]) - np.array(data["K1"]["mlp_norm_mid"])
    mean_d, lo_d, hi_d = bootstrap_ci(diff_mlp)
    p_mlp = perm_test_paired(diff_mlp, alternative="two-sided")
    v1_ok = (lo_d > 0 or hi_d < 0) and p_mlp < 0.05
    report_lines.append(f"| **V1 (MLP-Norm Mid)** | K2 − K1 | {mean_d:+.2f} ({lo_d:+.2f} bis {hi_d:+.2f}) | p = {p_mlp:.4f} | {'✓ JA' if v1_ok else '✗ NEIN'} |")

    # V2: Residual Cosine Drift K2 vs K1
    diff_cos = np.array(data["K2"]["residual_cos_mid"]) - np.array(data["K1"]["residual_cos_mid"])
    mean_c, lo_c, hi_c = bootstrap_ci(diff_cos)
    p_cos = perm_test_paired(diff_cos, alternative="less")
    v2_ok = hi_c < 0 and p_cos < 0.05
    report_lines.append(f"| **V2 (Residual Cos Drift)** | K2 − K1 | {mean_c:+.4f} ({lo_c:+.4f} bis {hi_c:+.4f}) | p = {p_cos:.4f} | {'✓ JA' if v2_ok else '✗ NEIN'} |")

    # V3: Sink-Masse t+1 K2 vs K1
    diff_sink = np.array(data["K2"]["sink_after"]) - np.array(data["K1"]["sink_after"])
    mean_s, lo_s, hi_s = bootstrap_ci(diff_sink)
    p_sink = perm_test_paired(diff_sink, alternative="greater")
    v3_ok = lo_s > 0 and p_sink < 0.05
    report_lines.append(f"| **V3 (Sink-Masse t+1)** | K2 > K1 | {mean_s:+.4f} ({lo_s:+.4f} bis {hi_s:+.4f}) | p = {p_sink:.4f} | {'✓ JA' if v3_ok else '✗ NEIN'} |")

    # V4: Dissoziation K2 vs K3 in MLP
    diff_v4 = np.abs(diff_mlp) - np.abs(np.array(data["K3"]["mlp_norm_mid"]) - np.array(data["K1"]["mlp_norm_mid"]))
    mean_v4, lo_v4, hi_v4 = bootstrap_ci(diff_v4)
    p_v4 = perm_test_paired(diff_v4, alternative="greater")
    v4_ok = lo_v4 > 0 and p_v4 < 0.05
    report_lines.append(f"| **V4 (|K2−K1| > |K3−K1|)** | |K2| − |K3| | {mean_v4:+.2f} ({lo_v4:+.2f} bis {hi_v4:+.2f}) | p = {p_v4:.4f} | {'✓ JA' if v4_ok else '✗ NEIN'} |")

    report_text = "\n".join(report_lines)
    
    slug_name = model_name.replace("/", "__")
    out_path = Path(out_dir) / "analyse" / slug_name / "bericht.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    
    # Rohe Daten speichern
    raw_path = Path(out_dir) / "ergebnisse" / f"{slug_name}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    
    print(f"\nBericht gespeichert in {out_path}")
    print(f"Rohdaten gespeichert in {raw_path}")
    print("\n" + report_text)
    return report_text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modell", default="gpt2")
    ap.add_argument("--ordner", default="lauf_valenz")
    args = ap.parse_args()
    run_valenz_experiment(args.modell, out_dir=args.ordner)
