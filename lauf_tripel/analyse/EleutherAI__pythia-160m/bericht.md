# Auswertung: EleutherAI/pythia-160m (Tripelstruktur-Ablation)

Datum: 2026-09-27 10:50, n = 100 Sequenzen, N = 128, Gerät: mps

## 1. Vollfaktorielle Matrix ($2 \times 2 \times 2$)

| Bedingung | Art ($Q K^T$) | Folge (Pos) | Zugleichsein ($\sum V$) | Loss (NLL) | Top-1 Acc | Top-5 Acc | Perplexität |
|---|---|---|---|---|---|---|---|
| **T111** | 1 | 1 | 1 | **3.21** | 40.4% | 63.2% | 24.9 |
| **T011** | 0 | 1 | 1 | **4.99** | 22.1% | 43.1% | 146.4 |
| **T101** | 1 | 0 | 1 | **5.54** | 17.8% | 36.2% | 253.6 |
| **T110** | 1 | 1 | 0 | **23.73** | 0.0% | 0.0% | 20317535267.4 |
| **T100** | 1 | 0 | 0 | **23.73** | 0.0% | 0.0% | 20317535267.4 |
| **T010** | 0 | 1 | 0 | **23.73** | 0.0% | 0.0% | 20317535267.4 |
| **T001** | 0 | 0 | 1 | **6.38** | 14.3% | 30.6% | 591.6 |
| **T000** | 0 | 0 | 0 | **23.73** | 0.0% | 0.0% | 20317535267.4 |

## 2. Prüfung der Hypothesen (PRAEREG_6_tripel.md)

| Hypothese | Test / Differenz | 95%-KI | p-Wert | Bestätigt? |
|---|---|---|---|---|
| **H-Art (Inhärenz fehlt)** | T011 − T111 | +1.77 (+1.66 bis +1.91) | p = 0.0002 | ✓ JA |
| **H-Folge (Sukzession fehlt)** | T101 − T111 | +2.32 (+2.15 bis +2.54) | p = 0.0002 | ✓ JA |
| **H-Zugleichsein (Gemeinschaft fehlt)** | T110 − T111 | +20.52 (+19.21 bis +22.12) | p = 0.0002 | ✓ JA |
| **Rangordnung der Schäden** | T110 > T101 > T011 | 20.52 > 2.32 > 1.77 | - | ✓ JA |
| **H-Synthese (Superadditivität)** | Voll − Summe(Einzel) | +3.17 (+2.89 bis +3.49) | p = 0.0002 | ✓ JA |