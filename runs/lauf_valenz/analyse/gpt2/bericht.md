# Auswertung: gpt2 (Valenz-Inkongruenz)

Datum: 2026-09-27 09:35, n = 100 Minimalpaare, Gerät: mps

## 1. Übersicht der Mittelwerte

| Metrik | K1 (Konsistent) | K2 (Paradox) | K3 (Neutral) |
|---|---|---|---|
| Loss am Target (Surprise) | 6.74 | 7.95 | 9.93 |
| MLP-Norm Mid (Schichten 4–8) | 10.83 | 10.68 | 10.71 |
| MLP-Sparsity Mid (< 1e-4) | 0.846 | 0.849 | 0.849 |
| Residual Cosine Drift (Mid) | 0.6641 | 0.6519 | 0.6449 |
| Residual Cosine Drift (Final) | 0.9924 | 0.9925 | 0.9911 |
| Sink-Masse am Target | 0.5347 | 0.5492 | 0.5547 |
| Sink-Masse nach Target (t+1) | 0.4973 | 0.5006 | 0.5004 |
| Ausgabe-Entropie (t+1) | 2.39 | 2.69 | 3.13 |

## 2. Prüfung der Hypothesen (PRAEREG_5_valenz.md)

| Hypothese | Vergleich | Differenz (95%-KI) | p-Wert | Bestätigt? |
|---|---|---|---|---|
| **V1 (MLP-Norm Mid)** | K2 − K1 | -0.15 (-0.25 bis -0.04) | p = 0.0092 | ✓ JA |
| **V2 (Residual Cos Drift)** | K2 − K1 | -0.0122 (-0.0188 bis -0.0056) | p = 0.0006 | ✓ JA |
| **V3 (Sink-Masse t+1)** | K2 > K1 | +0.0032 (+0.0004 bis +0.0062) | p = 0.0188 | ✓ JA |
| **V4 (|K2−K1| > |K3−K1|)** | |K2| − |K3| | -0.00 (-0.08 bis +0.07) | p = 0.5173 | ✗ NEIN |