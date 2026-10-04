# Auswertung: EleutherAI/pythia-160m (Valenz-Inkongruenz)

Datum: 2026-09-27 09:36, n = 100 Minimalpaare, Gerät: mps

## 1. Übersicht der Mittelwerte

| Metrik | K1 (Konsistent) | K2 (Paradox) | K3 (Neutral) |
|---|---|---|---|
| Loss am Target (Surprise) | 6.47 | 7.01 | 9.45 |
| MLP-Norm Mid (Schichten 4–8) | 12.68 | 12.35 | 12.23 |
| MLP-Sparsity Mid (< 1e-4) | 0.769 | 0.774 | 0.775 |
| Residual Cosine Drift (Mid) | 0.6573 | 0.6480 | 0.6396 |
| Residual Cosine Drift (Final) | 0.9925 | 0.9923 | 0.9925 |
| Sink-Masse am Target | 0.5144 | 0.5298 | 0.5279 |
| Sink-Masse nach Target (t+1) | 0.4521 | 0.4597 | 0.4595 |
| Ausgabe-Entropie (t+1) | 2.69 | 2.95 | 3.17 |

## 2. Prüfung der Hypothesen (PRAEREG_5_valenz.md)

| Hypothese | Vergleich | Differenz (95%-KI) | p-Wert | Bestätigt? |
|---|---|---|---|---|
| **V1 (MLP-Norm Mid)** | K2 − K1 | -0.33 (-0.48 bis -0.17) | p = 0.0002 | ✓ JA |
| **V2 (Residual Cos Drift)** | K2 − K1 | -0.0093 (-0.0179 bis -0.0009) | p = 0.0186 | ✓ JA |
| **V3 (Sink-Masse t+1)** | K2 > K1 | +0.0076 (+0.0042 bis +0.0111) | p = 0.0002 | ✓ JA |
| **V4 (|K2−K1| > |K3−K1|)** | |K2| − |K3| | -0.14 (-0.29 bis -0.01) | p = 0.9750 | ✗ NEIN |