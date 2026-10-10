# Auswertung: EleutherAI/pythia-160m (Tripelstruktur-Ablation)

Datum: 2026-10-05 18:47, n = 100 Sequenzen, N = 128, Gerät: mps

## 1. Vollfaktorielle Matrix ($2 \times 2 \times 2$)

| Bedingung | Art ($Q K^T$) | Folge (Pos) | Zugleichsein ($\sum V$) | Loss (NLL) | Top-1 Acc | Top-5 Acc | Perplexität |
|---|---|---|---|---|---|---|---|
| **T111** | 1 | 1 | 1 | **3.21** | 40.4% | 63.2% | 24.9 |
| **T011** | 0 | 1 | 1 | **7.97** | 9.9% | 23.3% | 2888.5 |
| **T101** | 1 | 0 | 1 | **5.54** | 17.8% | 36.2% | 253.6 |
| **T110** | 1 | 1 | 0 | **6.88** | 11.4% | 25.9% | 969.5 |
| **T100** | 1 | 0 | 0 | **6.88** | 11.4% | 25.9% | 969.5 |
| **T010** | 0 | 1 | 0 | **6.88** | 11.4% | 25.9% | 969.5 |
| **T001** | 0 | 0 | 1 | **7.97** | 9.9% | 23.3% | 2888.5 |
| **T000** | 0 | 0 | 0 | **6.88** | 11.4% | 25.9% | 969.5 |

**Identische Bedingungen** (Design nicht vollständig faktoriell): T011 = T001; T110 = T100 = T010 = T000

## 2. Prüfung der Hypothesen (PRAEREG_6_tripel.md)

| Hypothese | Test / Differenz | 95%-KI | p-Wert | Bestätigt? |
|---|---|---|---|---|
| **H-Art (Inhärenz fehlt)** | T011 − T111 | +4.75 (+4.07 bis +5.65) | p = 0.0002 | ✓ JA |
| **H-Folge (Sukzession fehlt)** | T101 − T111 | +2.32 (+2.15 bis +2.54) | p = 0.0002 | ✓ JA |
| **H-Zugleichsein (Gemeinschaft fehlt)** | T110 − T111 | +3.66 (+3.55 bis +3.80) | p = 0.0002 | ✓ JA |
| **Rangordnung der Schäden** | T110 > T101 > T011 | 3.66 > 2.32 > 4.75 | - | ✗ NEIN |
| **H-Synthese (Superadditivität)** | Voll − Summe(Einzel) | +4.75 (+4.07 bis +5.65) | p = 0.0002 | ✓ JA |