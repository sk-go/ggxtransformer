"""
chroma_transformer/visualizer.py
================================
Visualisiert die chromatische Entwicklung von Begriffen (z. B. "Dinosaurier")
über die Schichten des Transformers im Terminal (ANSI 24-bit TrueColor)
und als grafischen Plot.
"""

from typing import List, Optional
import torch
from transformers import AutoTokenizer

from chroma_transformer.model import ChromaTransformer, ChromaConfig


def rgb_to_ansi(r: float, g: float, b: float, text: str = "  ") -> str:
    """
    Erzeugt einen 24-Bit TrueColor ANSI-Farbblock für das Terminal.
    r, g, b liegen im Bereich [0.0, 1.0].
    """
    ri = int(min(max(r, 0.0), 1.0) * 255)
    gi = int(min(max(g, 0.0), 1.0) * 255)
    bi = int(min(max(b, 0.0), 1.0) * 255)
    # Kontrastreiche Textfarbe (weiß oder schwarz)
    luminance = 0.299 * ri + 0.587 * gi + 0.114 * bi
    fg = 0 if luminance > 128 else 255
    return f"\033[48;2;{ri};{gi};{bi}m\033[38;2;{fg};{fg};{fg}m {text} \033[0m"


def print_chromatic_trace(
    model: ChromaTransformer,
    tokenizer,
    prompt: str,
    target_token: Optional[str] = None,
):
    """
    Zeigt im Terminal, wie sich die Farben der einzelnen Tokens durch die Schichten entwickeln.
    """
    model.eval()
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    tokens = [tokenizer.decode([tok_id]) for tok_id in input_ids[0].cpu()]

    with torch.no_grad():
        out = model(input_ids, return_color_trace=True)
        # Form: (n_layer + 1, 1, T, 3)
        trace = out["color_trace"].squeeze(1)

    n_layers_plus_1, seq_len, _ = trace.shape

    print("\n" + "=" * 70)
    print(f"CHROMATISCHE ENTWICKLUNG: „{prompt}“")
    print("=" * 70)

    # Schichtweise Ausgabe
    for layer_idx in range(n_layers_plus_1):
        layer_label = "L0 (Archetyp)" if layer_idx == 0 else f"L{layer_idx} (Residual)"
        line_str = f"{layer_label:14} | "
        for t_idx in range(seq_len):
            rgb = trace[layer_idx, t_idx].tolist()
            tok_repr = tokens[t_idx].strip()
            if not tok_repr:
                tok_repr = "_"
            line_str += rgb_to_ansi(rgb[0], rgb[1], rgb[2], tok_repr) + " "
        print(line_str)
    print("=" * 70 + "\n")


def plot_chromatic_spectrum(
    model: ChromaTransformer,
    tokenizer,
    prompt: str,
    output_path: str = "chroma_trace.png",
):
    """
    Speichert eine visuelle Farbmatrix als Grafik.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Matplotlib nicht installiert; Plot übersprungen.")
        return

    model.eval()
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    tokens = [tokenizer.decode([tok_id]).strip() for tok_id in input_ids[0].cpu()]

    with torch.no_grad():
        out = model(input_ids, return_color_trace=True)
        # Form: (n_layer + 1, T, 3)
        trace = out["color_trace"].squeeze(1).cpu().numpy()

    n_layers, seq_len, _ = trace.shape

    fig, ax = plt.subplots(figsize=(max(8, seq_len * 1.2), n_layers * 0.8))
    ax.imshow(trace, aspect="auto", origin="lower")

    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=10)

    layer_labels = ["L0: Archetyp"] + [f"L{i}: Residual" for i in range(1, n_layers)]
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_labels, fontsize=10)

    ax.set_title(f"Chromatische Kristallisation des Sinns\n„{prompt}“", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Farbspektrum gespeichert unter: {output_path}")


if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    cfg = ChromaConfig(vocab_size=len(tok), block_size=64, n_layer=4, n_head=4, n_embd=64)
    m = ChromaTransformer(cfg)

    prompt = "Der Dinosaurier fand ein uraltes Fossil im Museum"
    print_chromatic_trace(m, tok, prompt)
