"""
chroma_transformer/train.py
===========================
Trainiert den Chromatischen Transformer (Gewöhnung über Backpropagation).
Demonstriert die Entstehung der Synthese (Haeckels Gesetz / Gewohnheit).
"""

import time
from pathlib import Path
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

from chroma_transformer.model import ChromaTransformer, ChromaConfig
from chroma_transformer.ablation_test import evaluate_conditions, print_ablation_report
from chroma_transformer.visualizer import print_chromatic_trace


def train_mini(
    texts: list,
    steps: int = 200,
    lr: float = 3e-3,
    device: str = "cpu",
):
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    cfg = ChromaConfig(
        vocab_size=len(tok),
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=64,
        dropout=0.0,
    )
    model = ChromaTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Daten vorbereiten
    all_tokens = []
    for t in texts:
        ids = tok.encode(t)
        all_tokens.extend(ids)
    data = torch.tensor(all_tokens, dtype=torch.long, device=device)

    print("\n" + "=" * 70)
    print(f"TRAINING DES CHROMATISCHEN TRANSFORMERS (Gewöhnung)")
    print(f"Parameter: {sum(p.numel() for p in model.parameters()):,} | Steps: {steps}")
    print("=" * 70)

    # Vor dem Training (step 0)
    print("\n>>> ZUSTAND VOR DEM TRAINING (Step 0 - Tabula Rasa):")
    sample_seq = data[:min(32, len(data))].unsqueeze(0)
    print_ablation_report(evaluate_conditions(model, sample_seq))

    model.train()
    block_size = cfg.block_size
    batch_size = 4

    t0 = time.time()
    for step in range(1, steps + 1):
        # Zufällige Batches ziehen
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])

        out = model(x, targets=y)
        loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps:
            dt = time.time() - t0
            print(f"Step {step:4d}/{steps} | Loss: {loss.item():.4f} | Zeit: {dt:.1f}s")

    print("\n>>> ZUSTAND NACH DEM TRAINING (Step", steps, "- Sedimentierter Sinn):")
    print_ablation_report(evaluate_conditions(model, sample_seq))

    # Farbspur demonstrieren
    test_phrase = "Der Dinosaurier im Museum"
    print_chromatic_trace(model, tok, test_phrase)

    return model, tok


if __name__ == "__main__":
    beispiel_texte = [
        "Der Dinosaurier ist ein prähistorisches Reptil und ausgestorben.",
        "Im Museum steht das Skelett eines riesigen Dinosauriers aus Stein und Knochen.",
        "Tyrannosaurus Rex war ein Fleischfresser, während der Brachiosaurus Pflanzen fraß.",
        "Die Paläontologie erforscht Fossilien aus der Kreidezeit und dem Jura.",
    ] * 20  # Wiederholung zur Gewöhnung

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    train_mini(beispiel_texte, steps=100, lr=2e-3, device=device)
