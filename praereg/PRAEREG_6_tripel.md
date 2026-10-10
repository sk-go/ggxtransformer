# Präregistrierung 6: Die faktorielle Ablations-Triade (Synthese-Beweis der Tripelstruktur)

*Vor dem Hauptlauf ausfüllen und committen. Datum: 27. September 2026*

## Anlass

Bisherige Experimente untersuchten Störungen als Mangelphänomene (Asynchronizität, Flucht in den Attention Sink). Der eigentliche theoretische Kern der Arbeit ist jedoch die **positive Existenz der Tripelstruktur des Sinns** im Transformer:

> **Sinn ist eine Folge von Gleichzeitigkeiten von Gleichartigkeiten.**

Im mathematischen Vollzug des Transformer-Attention-Blocks entsprechen die drei Dimensionen exakt drei distinkten Komponenten:
1. **Art (kantische Inhärenz / Humes Ähnlichkeit):** Der semantische Resonanzterm $Q \cdot K^T / \sqrt{d_k}$. Er bestimmt, *welche* Erscheinungsprofile zueinander passen.
2. **Folge (kantische Sukzession / Kausalität):** Die Positionskodierung ($\text{wpe}$ bei GPT-2, RoPE bei Pythia) und die kausale Maske $M_{\text{kausal}}$. Sie stiftet die gerichtete Ordnung der Zeit.
3. **Zugleichsein (kantische Gemeinschaft):** Die Multiplikation der Aufmerksamkeitsgewichte mit den Values $V$ ($\sum_{j \le i} \alpha_{ij} V_j$). Sie integriert alle im Kontextfenster simultan anwesenden Gehalte in einen gemeinsamen neuen Vektor im Residual Stream.

Dieses Experiment testet die These, dass Sinn nicht die bloße Summe dreier isolierter Module ist, sondern eine **nicht-lineare Synthese**: Erst wenn alle drei Momente zugleich aktiv sind, kristallisiert Sinn.

---

## Das vollfaktorielle $2 \times 2 \times 2$-Design

Jede der drei Dimensionen wird im Inferenzschritt gezielt auf Aktiv ($1$) oder Ablation ($0$) geschaltet:

| Bedingung | Art ($Q \cdot K$) | Folge ($\text{Pos}$) | Zugleichsein ($\sum V$) | Intervention im Modell |
|---|---|---|---|---|
| **T111 (Voller Sinn)** | 1 | 1 | 1 | Unverändertes Modell (Basislinie) |
| **T011 (Ohne Art)** | 0 | 1 | 1 | $Q K^T = 0$; Aufmerksamkeit fließt kausal, aber inhaltsblind ($\alpha_{ij} = \frac{1}{i+1}$) |
| **T101 (Ohne Folge)** | 1 | 0 | 1 | $\text{Pos} = 0$; zeitlose Menge, reines semantisches Set-Matching |
| **T110 (Ohne Zugleichsein)** | 1 | 1 | 0 | $\alpha_{ij} = \delta_{ij}$ (Einheitsmatrix); jedes Token prozessiert nur sich selbst |
| **T100 (Nur Art)** | 1 | 0 | 0 | Pos=0, Diagonale $V$ (isoliertes, ortloses Einzel-Embedding) |
| **T010 (Nur Folge)** | 0 | 1 | 0 | $QK^T=0$, Diagonale $V$ (reine Positionsmarkierung ohne Inhalt) |
| **T001 (Nur Zugleichsein)** | 0 | 0 | 1 | $QK^T=0$, Pos=0; ungerichtete gleichmäßige Vermischung (Bag-of-Words Mittelwert) |
| **T000 (Vollständige Leere)** | 0 | 0 | 0 | $QK^T=0$, Pos=0, Diagonale $V$ |

---

## Messgrößen

Pro Sequenz und Bedingung werden erfasst:
1. **Vorhersage-Loss (Perplexität):** $\text{NLL} = -\frac{1}{L} \sum_t \log P(x_t \mid x_{<t})$.
2. **Top-1 Genauigkeit:** Anteil der exakt korrekt vorhergesagten nächsten Tokens.
3. **Ausgabe-Entropie:** Mittlere Unsicherheit der Vorhersageverteilung.
4. **Synthese-Synergie (Nicht-Linearität dritter Ordnung):**
   $$\text{Synergie} = \Delta \text{Loss}(\text{T111}) - \left[ \Delta \text{Loss}(\text{T011}) + \Delta \text{Loss}(\text{T101}) + \Delta \text{Loss}(\text{T110}) \right]$$
   misst, ob der volle Sinn mehr ist als die Summe seiner Teile.

---

## Vorhersagen

- **H-Art (Bedeutung der Inhärenz):** Das Abschalten der Art ($T011$) erhöht den Loss signifikant gegenüber $T111$ ($T011 > T111$, $p < 0{,}001$).
- **H-Folge (Bedeutung der Sukzession):** Das Abschalten der Folge ($T101$) erhöht den Loss signifikant gegenüber $T111$ ($T101 > T111$, $p < 0{,}001$).
- **H-Zugleichsein (Bedeutung der Gemeinschaft):** Das Abschalten des Zugleichseins ($T110$) erzeugt den stärksten Einzeleffekt ($T110 > T101 > T011$).
- **H-Synthese (Nicht-Lineare Superadditivität):** Sinn kollabiert nicht-linear: Der gemeinsame Informationsgewinn in $T111$ ist signifikant größer als die Summe der drei Einzelbeiträge ($T100, T010, T001$).

---

## Festgelegte Parameter

- **Modelle:** `gpt2` und `EleutherAI/pythia-160m` (step143000).
- **Daten:** $n = 100$ ungesehene Sequenzen aus dem Wikipedia-Korpus (`lauf_wiki/korpus/texte.json`), $N = 128$ Tokens, Seed 1234.
- **Arbeitsordner:** `lauf_tripel`.
