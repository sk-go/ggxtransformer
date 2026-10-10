# Spec: Gleichartigkeit, Zugleichsein und bevorzugte Basis (Chroma-Neubau)

*Stand: 10.10.2026. Abgestimmt im Brainstorming; Hintergrund und philosophische Fassung in `docs/chroma_ideen.md` (§§ 10, 10a, 10b). Dieses Dokument ist der Bauplan, keine Präregistrierung. Die Präregistrierungen (PRAEREG_8, PRAEREG_9) werden daraus abgeleitet und vor dem jeweiligen Lauf committet.*

---

## 1. Ziel und Leitfrage

**Leitfrage:** Ist die qualitative Verschiedenheit der Dimensionen (bevorzugte Basis) Bedingung der Möglichkeit *abgestufter* Gleichartigkeit, oder genügt das Zugleichsein vieler Dimensionen? In der Sprache der Systemskizze: Gehört die Farbe zum minimalen Gerüst oder zu den Füllungen?

**Definition, metaphysisch:** Eine bevorzugte Basis liegt vor, wenn die Warte ihre Achsen unterscheiden kann. Homogen heißt: Alle Drehungen des Raums sind für die Warte ununterscheidbar. Bevorzugt heißt: Die Symmetrie schrumpft auf Vertauschungen der Achsen.

**Definition, technisch:** Ein Modell hat *architektonisch* keine bevorzugte Basis, wenn eine orthogonale Drehung aller Gewichte, die den Residual Stream berühren, identische Ausgaben liefert. *Empirisch* zeigt sich eine bevorzugte Basis als Exzess-Kurtosis der Stream-Aktivierungen über die Dimensionen, die bei zufälliger Drehung verschwindet.

**Ziele in dieser Reihenfolge:** (c) konstruktiv (ein Transformer, dessen Basis eine Variable ist), (b) Test (Vorhersagen, die scheitern können), (a) Sichtbarmachung (später, nicht Teil dieses Specs).

## 2. Nicht-Ziele

- **Folge** (über Zeit oder Schichten): eigene Folgestudie.
- **Spiel** (Schiller, `chroma_ideen.md` § 12a) und Detektiv-Tor: eigene Brainstorming-Runde.
- **Visualisierung** (HSV-Abbildung): Stufe (a), später.
- **Cloud-Skalierung:** später, mit eigener Präregistrierung.
- Nichtnegativer Stream selbst (Variante δ): zurückgestellt.
- Weiterentwicklung des alten `chroma_transformer/`: wird ersetzt, nicht erweitert.

## 3. Vorstudie: Genealogie der Achsen (ohne Training)

**Modelle:** GPT-2 (small) und Pythia-160m mit den 11 Checkpoints aus `legacy_experiments/gewoehnung_experiment.py` (`step0` bis `step143000`). Für E3 zusätzlich ein dichteres Raster von etwa 30 Pythia-Checkpoints.

**Text:** das vorhandene, ungesehene Wikipedia-Korpus aus `runs/lauf_wiki` (n = 300 Sequenzen, N = 256).

**Präregistrierte Messungen (PRAEREG_8):**

| Kürzel | Messung | Vorhersage |
|---|---|---|
| V1 | Exzess-Kurtosis der Stream-Aktivierungen über die Dimensionen, je Schicht, im Endmodell; Kontrolle: dieselben Aktivierungen nach zufälliger orthogonaler Drehung (10 Drehungen) | Kurtosis deutlich über der Drehungskontrolle |
| V2 | dieselbe Kurtosis über die 11 Pythia-Checkpoints | `step0` liegt auf Kontrollniveau, danach Anstieg |
| V3 | Spearman-Korrelation zwischen dem Betrag der Ausreißerdimensionen eines Tokens und seiner Frequenz (Frequenz aus dem Korpus als Proxy, als Einschränkung benannt) | positiv |

Position 0 wird bei der Kurtosis **getrennt** ausgewiesen (mit und ohne), weil der Attention Sink dort massive Aktivierungen erzeugt, die das Maß dominieren könnten.

**Explorativ (nicht präregistriert entscheidend, ausdrücklich so gekennzeichnet):**

- **E1 Nachträglichkeit:** Spearman-Korrelation der paarweisen Kosinus-Matrix der 2000 Wörter zwischen aufeinanderfolgenden Checkpoints; effektiver Rang des Embeddings (Participation Ratio der Kovarianz-Eigenwerte) über die Checkpoints. Frage: Ordnet sich Gleichartigkeit nachträglich um, wenn Dimensionen hinzukommen?
- **E2 Fraktale Dimension:** Korrelationsdimension (Grassberger-Procaccia) der Embedding-Punktwolke über mehrere Radius-Skalen. Frage: nicht ganzzahlig und skalenabhängig?
- **E3 Knotenlinie:** Verlauf der Kurtosis über das dichte Checkpoint-Raster. Frage: allmählicher Anstieg oder Sprung?
- **E4 Reflex:** Participation Ratio kontextueller Repräsentationen je Token gegen log-Frequenz. Frage: Werden häufige Tokens niedrigdimensionaler?

## 4. Hauptstudie: vier Warten

### 4.1 Arme

| Arm | Normalisierung | Optimierer | Farbe | Status der Basis |
|---|---|---|---|---|
| A₀ | RMSNorm, ein skalarer Gain | SGD mit Momentum | keine | keine |
| A | RMSNorm, ein skalarer Gain | AdamW | keine | kann als Füllung entstehen |
| Bα | RMSNorm, ein skalarer Gain | AdamW | E = softplus(W): nichtnegatives Embedding | Bestandteil im Archetyp |
| Bβ | RMSNorm, ein skalarer Gain | AdamW | jeder Block liest RMSNorm(softplus(x)); der Stream bleibt vorzeichenbehaftet | Bestandteil im Vollzug |

**5 Seeds je Arm**, Seed als Einheit der Inferenz. Gleiche Seeds über die Arme (gleiche Initialisierung der gemeinsamen Teile, gleiche Datenreihenfolge).

### 4.2 Modell (`chroma/modell.py`)

- Decoder-only, 6 Schichten, d = 256, 8 Köpfe, Kontext 256, MLP-Breite 4d mit GELU, gelernte absolute Positionen. Etwa 9M Parameter.
- **Embedding und Unembedding entkoppelt.**
- RMSNorm mit einem einzigen skalaren Gain (keine Gains je Dimension, keine Mittelwertsubtraktion).
- Isotrope Gauß-Initialisierung.
- Konfiguration über `farbe ∈ {keine, alpha, beta}`.
- Bekannte Nebenwirkung von Bα: Alle Embedding-Kosinusse sind ≥ 0 (Gleichartigkeit ohne Gegensatz). Das Kernmaß ist rangbasiert und davon nicht betroffen.

### 4.3 Daten (`chroma/korpus.py`)

- **WikiText-103** (CC BY-SA), heruntergeladen, nicht committet (`*/korpus/` ist bereits in `.gitignore`).
- **Eigener BPE-Tokenizer mit 8192 Tokens**, einmal auf dem Trainingsteil trainiert, von allen Armen geteilt.
- Tokenisierte Splits als memmap (`train.bin`, `val.bin`, `test.bin`).
- Messkontexte kommen aus dem Testsplit, nie aus Training oder Validierung.

### 4.4 Training (`chroma/training.py`)

- Kreuzentropie, Batch und Sequenzlänge für alle Arme gleich.
- **Lernrate:** kleiner Sweep je Optimierer im Pilot, Auswahl **nur nach Validierungs-Loss**, nie nach einem Messwert der Studie.
- **Loss-Stufen-Checkpoints:** Der Validierungs-Loss wird in festen Abständen bestimmt. Das Modell wird gespeichert, sobald es eine Stufe erstmals erreicht. Die Stufen werden im Pilot kalibriert (der eigene Tokenizer verschiebt die Skala) und in PRAEREG_9 festgeschrieben.
- **Budget:** Adam-Arme etwa eine Epoche (Größenordnung 10⁸ Tokens). A₀ bis zum Fünffachen, mit festem Deckel.
- Ausgabe je Lauf: Checkpoints je Stufe, Lernkurve, Konfiguration, Seed, Laufzeit.

### 4.5 Messung (`chroma/messung/`)

**Wortliste:** die 2000 häufigsten Ganzwort-Tokens des Tokenizers (Leerzeichen-Präfix, alphabetisch, mindestens 3 Zeichen). Fest für alle Arme.

**Kernmaß (b), Ersetzbarkeit (`ersetzbarkeit.py`):**
- Etwa 20 000 Paare je Modell, geschichtet nach Kosinus-Dezilen *dieses* Modells.
- Je Paar (x, y): 20 Testkontexte, in denen x an Position t steht, mit mindestens 8 Folgetokens; Fenster von 64 Tokens.
- S(x, y) = −mittlere KL(p_original ‖ p_ersetzt) über die Positionen t bis t+7. Symmetrisiert als Mittel beider Richtungen.
- **G** = Spearman(Kosinus der Embedding-Vektoren, S) über die Paare im mittleren 80-%-Bereich der Kosinus-Ränge.

**Sekundär (`verschachtelung.py`, P3):** hierarchisches Clustering (average linkage) auf Kosinus-Distanzen der 2000 Wörter, kophenetische Korrelation.

**Vergleich (`normen.py`, (a)):** SimLex-999, beschränkt auf abgedeckte Wörter; die Abdeckung wird berichtet. Entscheidet nichts.

**Basis (`basis.py`):** Exzess-Kurtosis des Streams je Schicht; Participation Ratio PR(v) = (Σvᵢ²)² / Σvᵢ⁴; FastICA-Drehung für A.

**Explorativ, sekundär (`metrik.py`):**
- **G_QK:** wie G, aber mit der eigenen Metrik des Modells statt Kosinus: symmetrisierte Bilinearform W_Q·W_Kᵀ der Köpfe in Schicht 0, gemittelt über die Köpfe, auf den normierten Embeddings.
- **G_top:** Überlappung (Jaccard) der 10 nächsten Nachbarn nach Kosinus und nach Ersetzbarkeit.

**Register „Begriffsbedarf“ (`register.py`):** Cluster, die über mindestens 3 von 5 Seeds stabil sind (Jaccard ≥ 0,5 der Mitglieder), werden mit Mitgliederlisten exportiert. Der Autor benennt sie; Unbenennbares geht ins Register. Rein explorativ.

### 4.6 Vorhersagen und Entscheidungsregeln (für PRAEREG_9)

**Primäre Loss-Stufe:** die niedrigste Stufe, die alle Seeds aller Arme erreicht haben. Erreicht A₀ eine Stufe nicht, wird das als Einschränkung berichtet.

**Manipulationsprüfung (zuerst):**
- Architektur: Invarianztest (siehe § 6) bestanden für A₀/A, gescheitert für Bα/Bβ unter Drehung, bestanden unter Permutation.
- Training: mittlere Exzess-Kurtosis des Streams in A₀ nahe 0, in A deutlich darüber. Schwellen vorläufig A₀ < 2, A > 10; endgültig in PRAEREG_9 vor dem Hauptlauf.
- Scheitert A₀ hier, ist der Vergleich nichtig. Das wird berichtet, nicht umgedeutet.

**P1, Gerüstfrage (Kern):** G(A₀) < G(A) bei gleicher Loss-Stufe.
- Test auf Seed-Ebene (5 gegen 5): einseitiger Welch-Test für die Differenz; TOST mit **δ = 0,05** für Äquivalenz; α = 0,05.
- **Bestätigt:** Differenz signifikant. **Widerlegt:** Äquivalenz signifikant. **Unentschieden:** keines von beiden.

**P2, Sichtbarkeitsfrage:**
- Bα: mittlere PR der Embedding-Vektoren in der eigenen Basis < mittlere PR von A *nach* FastICA-Drehung.
- Bβ: dasselbe für die Lesungen softplus(x) in Schicht 3 über die Wortvorkommen in Testkontexten, gegen den Stream von A in Schicht 3 nach FastICA.
- **Preisprüfung:** G(B) ≥ G(A) − δ (Nicht-Unterlegenheit). Scheitert sie, ist die Lesbarkeit mit Gleichartigkeit erkauft.

**P3, Verschachtelung (sekundär, gerichtet):** kophenetische Korrelation B ≥ A ≥ A₀.

**Pilotregel:** Pilot mit 1 Seed je Arm dient der Kalibrierung (Lernraten, Loss-Stufen, Laufzeit, Instrumente). Seine Effekte werden nicht gedeutet.

## 5. Komponenten

```
chroma/
  __main__.py          CLI: selbsttest | korpus | vorstudie | train | messen | auswerten
  modell.py            Modell mit Arm-Schalter
  korpus.py            Download, Tokenizer, memmap-Splits
  training.py          Trainingsschleife, Optimierer, Loss-Stufen-Checkpoints
  vorstudie.py         V1–V3, E1–E4 auf GPT-2/Pythia
  messung/
    ersetzbarkeit.py   S und G
    verschachtelung.py kophenetische Korrelation
    normen.py          SimLex-Vergleich
    basis.py           Kurtosis, PR, ICA
    metrik.py          G_QK, G_top
    register.py        Begriffsbedarf
  auswertung.py        Statistik über Seeds, Berichte (Markdown), Grafiken
tests/
  test_invarianz.py    Drehung/Permutation je Arm
  test_messung.py      Maße auf synthetischen Fällen
```

Jede Einheit ist ohne die anderen prüfbar: `modell.py` hängt von nichts ab; `messung/` nimmt ein Modell und Token-Folgen; `auswertung.py` liest nur gespeicherte Ergebnisse (JSON/CSV).

**Abhängigkeiten zusätzlich:** `scipy`, `scikit-learn`, `pytest` (`tokenizers` kommt mit `transformers`).

**Altbestand:** `chroma_transformer/` wird nach `legacy_experiments/chroma_transformer_alt/` verschoben.

## 6. Prüfungen

- **Invarianztest (`tests/test_invarianz.py`):** Zufällige orthogonale Matrix Q; alle Gewichte, die den Stream lesen oder schreiben, entsprechend transformieren (Embedding, Positionen, Ein- und Ausgangsprojektionen von Attention und MLP, Unembedding). A₀/A: Logits identisch (Toleranz numerisch). Bα/Bβ: Logits verschieden unter Q, identisch unter Permutationsmatrix P. Das ist die architektonische Manipulationsprüfung in Code.
- **Maßtests (`tests/test_messung.py`):** PR eines Einheitsvektors = 1, eines gleichverteilten Vektors = d; Kurtosis einer isotropen Gaußwolke ≈ 0; G = 1 auf einem konstruierten Fall, in dem S eine monotone Funktion des Kosinus ist.
- **`python -m chroma selbsttest`:** Offline-Durchlauf mit winzigem Zufallsmodell durch Training, Messung und Auswertung. Nach jeder Codeänderung auszuführen (Projektregel).

## 7. Ablauf

1. Code, Tests und Selbsttest.
2. `praereg/PRAEREG_8_achsen.md` committen → Vorstudie in `runs/chroma_vorstudie/`.
3. Pilot (1 Seed je Arm) in `runs/chroma_pilot/`: Kalibrierung, keine Deutung.
4. `praereg/PRAEREG_9_chroma.md` committen (mit kalibrierten Loss-Stufen, Lernraten, Schwellen) → Hauptlauf in `runs/chroma_haupt/`.
5. Bericht: alle Arme, alle Seeds, alle Stufen, explorative Teile gekennzeichnet.

## 8. Risiken

- **A₀ lernt zu langsam** und erreicht keine brauchbare Loss-Stufe. Dann wird der Vergleich auf einer hohen Stufe geführt und das so berichtet; im schlimmsten Fall bleibt P1 lokal unentschieden und geht in die Cloud-Studie.
- **Geringe Teststärke** mit 5 Seeds. Wird in PRAEREG_9 benannt; „unentschieden“ ist ein zulässiges Ergebnis.
- **Laufzeit auf MPS** unbekannt. Der Pilot misst sie; bei Bedarf werden Modellgröße oder Budget *vor* PRAEREG_9 angepasst.
- **Messkosten** der Ersetzbarkeit (etwa 400 000 Fenster je Modell). Gebündelt in Batches; Pilot misst die Dauer.
- **Frequenz-Proxy** in V3 stammt aus dem Wikipedia-Korpus, nicht aus den Trainingsdaten von GPT-2/Pythia. Als Einschränkung benannt.

## 9. Literatur (vor Verwendung prüfen)

Geprüft vorhanden (Inhalt laut Quelle): Anthropic 2023, *Privileged Bases in the Transformer Residual Stream*; Caples & Neuhaus, *Adam Optimizer Causes Privileged Basis in Transformer LM Residual Stream* (Blogpost, nicht begutachtet); Kovaleva et al. 2021, *BERT Busters*; Puccetti et al. 2022, *Outlier Dimensions that Disrupt Transformers are Driven by Frequency*.

Noch zu prüfen: Lee & Seung 1999 (NMF); Grassberger & Procaccia 1983; Hill, Reichart & Korhonen 2015 (SimLex-999); Merity et al. 2016 (WikiText-103); Kant B XIII/B XVIII; Hegel, Knotenlinie; Sartre, Brieföffner; D&G, Immanenzebene; Schiller, 15. Brief.
