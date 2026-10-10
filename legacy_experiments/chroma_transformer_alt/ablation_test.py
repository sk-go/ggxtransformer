"""
chroma_transformer/ablation_test.py
===================================
Vollfaktorielle 2x2x2 Ablations-Triade auf dem Chromatischen Transformer:
Prüft die Bedingungen T111 bis T000 und misst den Kollaps der Sinnkonstitution.
"""

from typing import Dict, List
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from chroma_transformer.model import ChromaTransformer, ChromaConfig


CONDITIONS = [
    ("T111", True, True, True, "Voller Sinn (Art + Folge + Zugleichsein)"),
    ("T011", False, True, True, "Ohne Art (Inhaltsblind, Folge + Zugleichsein)"),
    ("T101", True, False, True, "Ohne Folge (Zeitlose Menge, Art + Zugleichsein)"),
    ("T110", True, True, False, "Ohne Zugleichsein (Monadisch, Art + Folge)"),
    ("T100", True, False, False, "Nur Art (Isoliertes Merkmal)"),
    ("T010", False, True, False, "Nur Folge (Reine Positionsmarke)"),
    ("T001", False, False, True, "Nur Zugleichsein (Ungerichtetes Rauschen)"),
    ("T000", False, False, False, "Vollständige Leere (Nullpunkt)"),
]


def evaluate_conditions(
    model: ChromaTransformer,
    input_ids: torch.Tensor,
) -> List[Dict]:
    """
    Evaluiert eine Sequenz unter allen 8 Bedingungen des Sinntripels.
    """
    model.eval()
    targets = input_ids[:, 1:].contiguous()
    inputs = input_ids[:, :-1].contiguous()

    results = []

    with torch.no_grad():
        for code, art, folge, zugleich, desc in CONDITIONS:
            out = model(inputs, targets=targets, art=art, folge=folge, zugleich=zugleich)
            loss = out["loss"].item()
            logits = out["logits"]
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            # Top-1 Genauigkeit
            acc = (preds == targets).float().mean().item()

            # Mittlere Entropie der Vorhersage
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean().item()

            results.append({
                "code": code,
                "art": int(art),
                "folge": int(folge),
                "zugleich": int(zugleich),
                "loss": loss,
                "acc": acc,
                "entropy": entropy,
                "description": desc,
            })

    return results


def print_ablation_report(results: List[Dict]):
    print("\n" + "=" * 80)
    print("VOLLFAKTORIELLE TRIPEL-ABLATION (SYNTHESE-BEWEIS)")
    print("=" * 80)
    print(f"| {'Code':<6} | {'Art':<3} | {'Folge':<5} | {'Zugleich':<8} | {'Loss':<7} | {'Top-1':<7} | {'Entropie':<8} | Beschreibung")
    print("|" + "-" * 8 + "|" + "-" * 5 + "|" + "-" * 7 + "|" + "-" * 10 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 10 + "|" + "-" * 25)

    base_loss = results[0]["loss"]
    for r in results:
        delta = r["loss"] - base_loss
        delta_str = f" (+{delta:.2f})" if delta > 0.001 else " (Basis)"
        print(
            f"| {r['code']:<6} | {r['art']:^3} | {r['folge']:^5} | {r['zugleich']:^8} | "
            f"{r['loss']:>6.2f} | {r['acc']*100:>5.1f}% | {r['entropy']:>8.2f} | {r['description']}{delta_str}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    cfg = ChromaConfig(vocab_size=len(tok), block_size=64, n_layer=4, n_head=4, n_embd=64)
    model = ChromaTransformer(cfg)

    test_text = "Die Naturwissenschaften erklären die Welt durch Kausalität, die Geisteswissenschaften verstehen sie durch Sinn."
    encoded = tok(test_text, return_tensors="pt").input_ids
    results = evaluate_conditions(model, encoded)
    print_ablation_report(results)
