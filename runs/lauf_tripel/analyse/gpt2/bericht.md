# Auswertung: gpt2 (Tripelstruktur-Ablation)

Datum: 2026-09-27 10:23, n = 100 Sequenzen, N = 128, Gerät: mps

## 1. Vollfaktorielle Matrix ($2 \times 2 \times 2$)

| Bedingung | Art ($Q K^T$) | Folge (Pos) | Zugleichsein ($\sum V$) | Loss (NLL) | Top-1 Acc | Top-5 Acc | Perplexität |
|---|---|---|---|---|---|---|---|
| **T111** | 1 | 1 | 1 | **3.33** | 38.4% | 60.8% | 28.0 |
| **T011** | 0 | 1 | 1 | **9.00** | 4.2% | 14.3% | 8138.7 |
| **T101** | 1 | 0 | 1 | **9.80** | 3.0% | 10.5% | 18073.4 |
| **T110** | 1 | 1 | 0 | **16.12** | 0.0% | 2.3% | 9977001.6 |
| **T100** | 1 | 0 | 0 | **13.65** | 1.6% | 6.0% | 844338.9 |
| **T010** | 0 | 1 | 0 | **16.12** | 0.0% | 2.3% | 9977001.6 |
| **T001** | 0 | 0 | 1 | **10.64** | 4.1% | 13.6% | 41671.8 |
| **T000** | 0 | 0 | 0 | **13.65** | 1.6% | 6.0% | 844338.9 |

## 2. Prüfung der Hypothesen (PRAEREG_6_tripel.md)

| Hypothese | Test / Differenz | 95%-KI | p-Wert | Bestätigt? |
|---|---|---|---|---|
| **H-Art (Inhärenz fehlt)** | T011 − T111 | +5.67 (+5.54 bis +5.80) | p = 0.0002 | ✓ JA |
| **H-Folge (Sukzession fehlt)** | T101 − T111 | +6.47 (+6.26 bis +6.70) | p = 0.0002 | ✓ JA |
| **H-Zugleichsein (Gemeinschaft fehlt)** | T110 − T111 | +12.78 (+12.63 bis +12.94) | p = 0.0002 | ✓ JA |
| **Rangordnung der Schäden** | T110 > T101 > T011 | 12.78 > 6.47 > 5.67 | - | ✓ JA |
| **H-Synthese (Superadditivität)** | Voll − Summe(Einzel) | +9.77 (+9.53 bis +10.03) | p = 0.0002 | ✓ JA |