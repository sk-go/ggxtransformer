# Präregistrierung 5: Valenz-Inkongruenz und Rekognition im Begriff (Paradoxe Handlungen)

*Vor dem Lauf ausfüllen und committen. Datum: 27. September 2026*

## Anlass

1. **Die Asymmetrie der bisherigen Läufe:** In den globalen Messungen (`lauf`, `lauf_wiki`, `lauf_c0`) war der Art-Effekt ($C - A$ bzw. $C - C0$) um den Faktor 5 bis 10 schwächer als der Folge-Effekt ($B - A$).
2. **Der empirische Befund der Satzgrenzen-Analyse:** Die positionsbezogene Auswertung (27. September 2026) zeigte, dass Bedingung C die Art nur **punktuell an den Satzgrenzen** stört (wo ein massiver Loss-Schock von $+1{,}88$ und ein Peak der Sink-Masse auftritt). Innerhalb jedes Satzes ($d \ge 5$) baut sich sofort wieder eine lokale Warte auf; der globale Mittelwert verdünnt den Effekt über hunderte Tokens.
3. **Theoretisches Fundament der Hausarbeit (Abschnitt 4.1):**
   > *„Die Wertung stiftet die Gleichartigkeit mit anderen Phänomenen, die Unterstellung eines Urhebers ordnet das Ereignis einer Folge zu...“*
   Gleichartigkeit (Art) ist im menschlichen Geist keine neutrale Sachkategorie (Enzyklopädie-Thema), sondern eine **affektiv-evaluative Kategorie (Wertung, Archetyp)**. Die radikalste Störung von Gleichartigkeit ist daher nicht der Wechsel von Wikipedia-Artikeln, sondern die **Zerstörung der Handlungs- und Valenz-Kohärenz (paradoxe Handlungen: Fürsorge schlägt in Gewalt um, Freundlichkeit in Grausamkeit)** bei vollständig intakter Grammatik.
4. **Der architektonische Ort der Rekognition:** Während die Attention (situative Warte) den syntaktischen Bezug herstellt, vollziehen die **MLPs** nach Geva et al. (2021) und Meng et al. (2022) als Schlüssel-Wert-Speicher die begriffliche Rekognition (Kants „Rekognition im Begriff“). Eine Zerstörung der Art bei intakter Folge muss sich primär in den **mittleren MLPs (Schichten 4–8)** und im **Residual Stream** manifestieren.

---

## Stimulus-Design (Minimalpaare)

Verwendet werden englische Minimalpaar-Sätze mit festem syntaktischem Rahmen. Jedes Paar ist bis auf das kritische Zieltoken (Verb oder Prädikatsnomen) identisch:

- **K1 (Konsistent / Art ✓, Folge ✓):** Archetypisch und evaluativ kohärente Handlung.
  *Beispiel:* `"The loving mother embraced her crying child and comforted its trembling body."`
- **K2 (Paradox / Art ✗, Folge ✓):** Syntaktische Folge 100 % intakt, Grammatik identisch, aber die Valenz verkehrt sich paradox ins Gegenteil (Freundlichkeit $\to$ Gewalt).
  *Beispiel:* `"The loving mother embraced her crying child and fractured its trembling body."`
- **K3 (Kategoriell neutral / N400-Kontrolle):** Syntaktisch intakt, semantisch unpassend, aber ohne affektiv-moralische Paradoxie.
  *Beispiel:* `"The loving mother embraced her crying child and calculated its trembling body."`

Korpusgröße: Mindestens $n = 100$ validierte Minimalpaare pro Bedingung.

---

## Messgrößen am kritischen Zieltoken ($t$) und Folgetoken ($t+1$)

1. **MLP-Aktivierungsnorm (Rekognition im Begriff):**
   Mittlere L2-Norm der Zwischenaktivierungen nach der GeLU-Nichtlinearität in den MLPs der Schichten 4 bis 8:
   $$\|\mathbf{h}_{\text{mlp}}^{(l)}(t)\|_2$$
2. **Residual Stream Cosine Drift (Valenz-Bruch):**
   Kosinus-Ähnlichkeit der Vektoren im Residual Stream unmittelbar vor und nach dem kritischen Token:
   $$\cos(\mathbf{x}_{t-1}^{(l)}, \mathbf{x}_t^{(l)})$$
3. **Sink-Masse & Entropie (Die Warte im Vollzug):**
   - Aufmerksamkeit auf Token 0 an Position $t$ und $t+1$, gemittelt über die Schichten.
   - Ausgabe-Entropie an Position $t$ (Surprise am paradoxen Token).

---

## Vorhersagen

- **V1 (MLP-Reaktion auf Paradoxie):** In den mittleren Schichten (4–8) unterscheidet sich die mittlere MLP-Aktivierungsnorm am Zieltoken in K2 signifikant von K1; das 95%-Bootstrap-KI der Differenz $K2 - K1$ schließt 0 aus.
- **V2 (Residual Stream Drift):** Der Kosinus-Drift im Residual Stream am Zieltoken ist in K2 signifikant stärker (niedrigere Kosinus-Ähnlichkeit zu $\mathbf{x}_{t-1}$) als in K1 ($K2 < K1$).
- **V3 (Sink-Masse bei Valenz-Bruch):** An der Position unmittelbar nach dem paradoxen Token ($t+1$) ist die Sink-Masse in K2 signifikant höher als in K1 ($K2 > K1$, einseitig $p < 0{,}05$).
- **V4 (Dissoziation von K2 und K3):** K2 (affektive Paradoxie) erzeugt in den MLPs eine stärkere Abweichung von K1 als die rein neutrale semantische Unpassendheit K3 ($|K2 - K1| > |K3 - K1|$).

---

## Entscheidungsregeln

Signifikanzniveau $\alpha = 0{,}05$, 2000 Bootstrap-Ziehungen für Konfidenzintervalle, 5000 Permutationen für Hypothesentests.

### Vorab festgelegte Deutung

| V1 (MLP) | V3 (Sink) | Deutung |
|---|---|---|
| **ja** | **ja** | **Volle Konfirmation:** Die Zerstörung der Art (Valenz-Paradoxie) erschüttert sowohl die begriffliche Rekognition (MLP) als auch die situative Warte (Attention Sink). |
| **ja** | **nein** | **Dissoziation der Ebenen:** Die Art-Zerstörung wird in den begrifflichen Speichern (MLP) prozessiert, schlägt aber nicht auf den Attention Sink durch. Der Sink ist ein rein syntaktischer No-Op-Puffer. |
| **nein** | **ja** | **Unerwartet:** Der Sink reagiert, ohne dass die mittleren MLPs den Valenzbruch reflektieren. |
| **nein** | **nein** | **Theoriefälschung:** Auch affektiv-paradoxe Handlungen erschüttern weder MLP noch Attention; das Modell verarbeitet Valenz rein als Wahrscheinlichkeitsglättung. |

---

## Festgelegte Parameter

- **Modelle:** `gpt2` und `EleutherAI/pythia-160m` (step143000).
- **Batch-Größe:** 16 (oder 8 bei Speicherlimit).
- **Arbeitsordner:** `lauf_valenz`.
- **Seed:** 1234.
