# Präregistrierung 7: Ontogenese des Sinns – Die Entstehung der Tripelstruktur durch Gewohnheit (Training)

**Datum:** 2026-09-28  
**Autor:** Sasha Kaun  
**Projekt:** ggxtransformer (Die Tripelstruktur des Sinns im Transformer)  
**Status:** Präregistriert vor Durchführung der Messung.

---

## 1. Theoretischer Hintergrund & Fragestellung

Bisherige Messungen (PRAEREG_6_tripel) haben an fertig trainierten Modellen (GPT-2, Pythia-160m @ step143000) gezeigt, dass Sinn im Transformer eine superadditive Synthese von **Art** ($Q \cdot K$), **Folge** (Position/Maske) und **Zugleichsein** (Value-Aggregation) ist. 

Im Inferenzzustand wirkt die Architektur wie ein kantisches System *a priori*. Doch die leere Architektur erzeugt vor dem ersten Trainingsschritt (`step 0`) nur Rauschen. Gemäß David Hume entsteht die Verknüpfung von Vorstellungen nicht aus dem Verstand, sondern einzig aus **Gewohnheit** (*custom / habit*). Im Transformer wird diese Gewohnheit durch die fortlaufende Ent-Täuschung (Loss) und die Wartung der Gewichte über **Backpropagation** eingeschliffen.

Dieses Experiment untersucht die **Ontogenese des Sinns**: Wie bildet sich die Tripelstruktur des Sinns und ihre Superadditivität über den Verlauf des Trainings (11 Checkpoints von Pythia-160m) heraus?

---

## 2. Modell & Daten

* **Modell:** `EleutherAI/pythia-160m`
* **Checkpoints (11 Stufen):**
  `step0`, `step512`, `step1000`, `step2000`, `step4000`, `step8000`, `step16000`, `step33000`, `step66000`, `step100000`, `step143000`.
* **Korpus:** $n = 100$ unversehrte Wikipedia-Sequenzen aus `lauf_wiki/korpus/texte.json` (Bedingung A), einheitliche Länge $N = 128$ Tokens.
* **Hardware/Präzision:** Apple Silicon (`mps`), `float32`, `eager` Attention.

---

## 3. Bedingungen der Faktoriellen Ablations-Triade ($2 \times 2 \times 2$)

Pro Checkpoint werden die 8 Bedingungen gemessen:
* $T111$: Voller Sinn (Art + Folge + Zugleichsein)
* $T011$: Ohne Art ($Q = 0 \implies$ uniforme Kausal-Attention)
* $T101$: Ohne Folge (RoPE $\cos = 1, \sin = 0$)
* $T110$: Ohne Zugleichsein (Keine Token-Mischung, $W = I$)
* $T100$: Nur Art
* $T010$: Nur Folge
* $T001$: Nur Zugleichsein
* $T000$: Vollständige Leere (keine Art, keine Folge, kein Zugleichsein)

---

## 4. Präregistrierte Hypothesen

### H-Gewohnheit 1 (Kein Sinn vor aller Gewohnheit):
Bei `step 0` (zufällige Initialisierung) ist die Superadditivität der Dimensionen vernachlässigbar klein:
$$H_{synthese}(\text{step 0}) \approx 0$$
Das Modell besitzt zwar die vollständige Architektur (Hardware/Bias), aber noch keinen Sinn.

### H-Gewohnheit 2 (Monotones Wachstum der Synthese):
Die Superadditivität $H_{synthese}$ wächst über das Training signifikant an:
$$H_{synthese}(\text{step 143000}) > H_{synthese}(\text{step 0})$$
Der Gewinn des vollen Sinns gegenüber der Leere ($\Delta_{voll} = \mathcal{L}_{T000} - \mathcal{L}_{T111}$) übersteigt mit fortschreitendem Training die Summe der isolierten Einzelbeiträge ($\Sigma_{iso}$).

### H-Gewohnheit 3 (Ontogenetische Reihenfolge der Warten):
Welche Dimension des Sinns wird zuerst durch Gewohnheit gelernt?
* **3a:** Der Verlust durch Wegfall von Zugleichsein ($T111 \to T110$) dominiert von Beginn an, da Wechselwirkung die physikalische Trägerstruktur darstellt.
* **3b:** Der Verlust durch Wegfall von Folge ($T111 \to T101$) und Art ($T111 \to T011$) bildet sich erst ab mittleren Trainingsschritten heraus, wenn syntaktische und semantische Regularitäten sedimentieren.

### H-Gewohnheit 4 (Attention Sink als erlernte Entlastung):
Bei `step 0` existiert kein Attention Sink; die Aufmerksamkeitsmasse auf Token 0 ist im Mittel $\approx 1/L$. Erst im Verlauf des Trainings kondensiert der Sink als sedimentierte Entlastungs-Warte heraus ($> 0{,}10$).

---

## 5. Statistische Auswertung & Entscheidungsregeln

* **Metrik:** Vorhersage-Loss $\mathcal{L}$ (Cross-Entropy in Nats) und Top-1 Genauigkeit.
* **Superadditivität:**
  $$H_{synthese} = (\mathcal{L}_{T000} - \mathcal{L}_{T111}) - [(\mathcal{L}_{T000} - \mathcal{L}_{T100}) + (\mathcal{L}_{T000} - \mathcal{L}_{T010}) + (\mathcal{L}_{T000} - \mathcal{L}_{T001})]$$
* **Konfidenzintervalle:** 95%-Bootstrap-Konfidenzintervalle (2000 Resamples) pro Checkpoint.
* **Signifikanz:** Permutationstest (5000 Permutationen), $\alpha = 0{,}05$.
