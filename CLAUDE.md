# Projektkontext: Die Tripelstruktur des Sinns im Transformer

Dieses Repo gehört zu einem freien philosophischen Essay, der den Transformer mit der Begrifflichkeit einer eigenen Theorie analysiert (Warte, Tripelstruktur des Sinns, Synchronizität nach Jung). Dazu kommt ein präregistriertes Experiment an GPT-2 und Pythia. Der Autor studiert Philosophie und Mathematik und schreibt auf Deutsch. Code-Kommentare, Berichte und Antworten deshalb auf Deutsch. Wenn etwas unklar ist, nachfragen statt raten.

## Die Theorie in Kürze

Grundlage ist die Hausarbeit „Gewohnte Gleichzeitigkeiten. Humes Verknüpfungen der Vorstellungen und Jungs Synchronizität“ (FSU Jena, SS 2026).

- **Warte:** eine zeitliche Invarianz, ein Gefüge stabiler Vor-Urteile, das dem Erscheinen eines Ereignisses vorausliegt. Eingehendes wird dort nicht registriert, sondern *erblickt*: bewertet, einem Urheber zugeordnet und in seiner Dauer geschätzt. Entsteht durch Gewohnheit (Hume).
- **Tripelstruktur des Sinns:** Sinn ist eine Folge von Gleichzeitigkeiten von Gleichartigkeiten. Die drei Dimensionen sind **Art** (Gleichartigkeit, kantische Inhärenz), **Folge** (Sukzession) und **Zugleichsein** (kantische Gemeinschaft).
- **Synchronizität:** sinngemäße Koinzidenz, also Zugleichsein und Gleichartigkeit ohne Kausalverhältnis (Jung). Wiederholte Synchronizitäten erzeugen durch Gewohnheit die Vorstellung einer Kraft; diese heißt Sinn.
- **Ent-Täuschung und Wartung:** Verfehlt die Erscheinung die Erwartung, wird die Warte erschüttert und muss gewartet werden. Das ist der Vorhersagefehler.
- **Asynchronizität:** Nicht-Passung, die nach Sinn verlangt; sie ist *sinnfordernd* (Castorps Herzklopfen im Zauberberg).
- **Detektiv-Kriterium:** Sinn unterscheidet sich von Aberglauben durch seine Bindung an andere Folgen. Der Aberglaube hat Warten, die so unspezifisch sind, dass alles sie bestätigt (Barnum-Effekt).
- **Archetyp:** apriorisches psychisches Angeordnetsein, nach Jung eine *psychische Wahrscheinlichkeit*.

## Der Transformer in diesen Begriffen

Die Warte gibt es auf drei Ebenen: **sedimentiert** (trainierte Gewichte, zur Laufzeit invariant), **situativ** (Kontext und KV-Cache) und im **Vollzug** (Residual Stream eines Tokens durch die Schichten).

- **Tokenisierung (BPE):** die erste Gewohnheit, denn was sich oft berührt, wird zu einer Einheit.
- **Embedding (Input):** Vor-Urteil im Wortsinn, Art ohne Folge und ohne Zugleichsein. Mehrdeutige Wörter liegen noch in Überlagerung.
- **Positionscodierung:** GPT-2 addiert gelernte *absolute* Positionsvektoren zum Input; die Zeitstelle wird Eigenschaft der Erscheinung (Newton). Pythia nutzt RoPE, *relative* Rotation innerhalb der Attention; Zeit ist Ordnung der Relationen (Leibniz).
- **Residual Stream:** der Ort, an dem das Werden schichtweise zum Sinn kristallisiert.
- **Add (Residualverbindung):** Jede Teilschicht schreibt hinzu, nichts wird überschrieben; Erhalt der Erscheinung in der Tiefe. Der Residual Stream als formale Einheit, die alle Synthesen begleitet (Kants „Ich denke“), aber verteilt: eine Einheit pro Token, verbunden nur über die Attention. Logit Lens macht die Kristallisation schichtweise sichtbar.
- **Norm (LayerNorm, bei GPT-2 und Pythia vor jeder Teilschicht):** entfernt die Größe des Vektors, das Urteil sieht nur die Richtung. Humes Lebhaftigkeit wird für das Urteil eingeklammert, im Residual Stream aber bewahrt. Pythia rechnet Attention und MLP parallel aus derselben Eingabe (Reproduktion und Rekognition zugleich), GPT-2 nacheinander.
- **Attention:** die Query ist die Erwartung, die Keys sind Erscheinungsprofile. Q·K ergibt Art, kausale Maske und Positionscodierung ergeben Folge, die gewichtete Summe der Values ergibt Zugleichsein.
- **Unembedding (Output):** Gleichartigkeit des gegenwärtigen Sinnzustands mit jeder möglichen nächsten Erscheinung, also Erwartungshorizont; Kausalität als projizierendes Prinzip. GPT-2 nutzt dieselbe Matrix für Input und Output (weight tying): Erscheinung und Erwartung werden am selben Maß gemessen. Pythia trennt beide Räume.
- **Autoregressive Schleife:** Der gezogene Output wird zum nächsten Input. Zwischen zwei Nutzereingaben erfüllt das Modell nur seine eigene Erwartung, ohne äußere Ent-Täuschung (fensterlose Monade, Hausarbeit 3.3.2).
- **Kausale Maske:** eine *halbierte* Gemeinschaft, ohne Wechselwirkung und ohne Nachträglichkeit.
- **Softmax:** Zwang zur Synchronizität, da die Aufmerksamkeit sich zu 1 summieren muss.
- **Multi-Head:** viele Warten zugleich.
- **MLP:** positionsweise, ohne Gemeinschaft; nach Geva et al. 2021 Schlüssel-Wert-Speicher. Schlüssel = Wiedererkennen/Subsumtion unter einen Begriff, Nichtlinearität = Schwelle des Urteils, Werte = gewohnheitsmäßige Folgerung („wenn Hitze, dann Flamme“), Addition in den Residual Stream = Mitgemeintes (Husserl). Dreifache Synthesis der A-Deduktion: Embedding ≈ Apprehension, Attention/KV-Cache ≈ Reproduktion, MLP ≈ Rekognition im Begriff. Faktenwissen sitzt nach Meng et al. 2022 (ROME) vor allem in mittleren MLPs und ist dort editierbar: Wartung ohne Ent-Täuschung. Etwa zwei Drittel der Parameter pro Block liegen im MLP.
- **Ausgabeverteilung:** Jungs psychische Wahrscheinlichkeit. Das Sampling ist der Tarot-Moment.
- **Training:** Der Loss ist die Ent-Täuschung, die Backpropagation die Wartung, RLHF die Sittlichkeit bzw. Sozialisierung. Nach dem Training gibt es eine Warte ohne Wartung.

Die Bruchstellen sind das Interessante: Warte ohne Wartung, Warte ohne Dauer (nur ein Positionsindex), kein legitimer Ort der Asynchronizität, halbierte Gemeinschaft.

## Essayplan

Leitmotiv: Die Theorie macht Vorhersagen, die Architektur prüft sie, das Ergebnis wirkt auf die Theorie zurück. Rahmen ist Dilthey: Interpretierbarkeitsforschung *erklärt* die Maschine, das Essay *versteht* sie.

1. Einleitung: Verstehen statt Erklären der Maschine.
2. Methode: Tripelstruktur und Warte, Als-ob-Phänomenologie als regulative Heuristik ohne Behauptung eines Erlebens.
3. Tripelstruktur im Generierungsschritt, Archetypen als Features. Dazu „Golden Gate Claude“ als experimentell erzeugter Aberglaube.
4. Säule 1: Attention Sinks und Asynchronizität, ergänzt um die Experimentbefunde (siehe unten; die ursprüngliche These muss angepasst werden).
5. Säule 2: Titans, also Wartung durch Ent-Täuschung. Deren blinder Fleck ist das Fehlen des Detektiv-Kriteriums. Architekturskizze: ein „Detektiv-Tor“, das Überraschung mit dem Bindungsgrad gewichtet.
6. Vorhersagen und Architekturskizzen, klar als Skizzen markiert.
7. Rückwirkung auf die Theorie: Asynchronizität als Leistung; Prediction Error notwendig, aber nicht hinreichend; Typen von Warten.
8. Schluss: die Warte des Lesers. Ist das eine Theorie des Bewusstseins? Antwort: eine Theorie der Sinnkonstitution, deren Lücken angeben, wo Bewusstsein zu suchen wäre. Reasoning-Tokens als „eingeräumte Dauer der Reflexion“ (Hausarbeit 4.1).

## Dateien

- `sinn_experiment.py`: Befehle `korpus`, `messen`, `analyse`, `dynamik`, `selbsttest` (Details in `README.md`).
- `wiki_texte.py`: sammelt über die Wikipedia-API Artikel, deren erste Version ab 2024 angelegt wurde (CC BY-SA, Quellen in `quellen.csv`).
- `README.md`: Anleitung und Präregistrierung (H1, H2, H3).
- `PRAEREG_2.md`: H3b, neue Hypothese zum Einbruch von D − A in der Trainingsmitte.
- `PRAEREG_3_replikation.md`: Replikation mit Wikipedia-Korpus, Vorhersagen R1–R3.

Ordner: `pilot/` (Seed 99), `lauf/` (Hauptlauf, Crawl-Korpus), `lauf_wiki/` (Replikation, Wikipedia-Korpus ab 2024), `wiki_texte/` (die Artikel selbst mit `quellen.csv` und `LIZENZ.md`).

## Experiment: Stand

Bedingungen: A kohärent, B wortgewürfelt (Folge zerstört), B2 satzgewürfelt, C disparat (Art zerstört), D Rauschen (beides zerstört). Messgrößen: Sink-Masse pro Kopf (Aufmerksamkeit auf das feste Start-Token, Query-Positionen ab 8), Ausgabe-Entropie, Loss. Parameter: N = 256, n = 300, Seed 1234.

### Hauptlauf GPT-2 (n = 300)

Korpus: englische Web-Crawl-Texte (FineWeb-artig), **nicht** Wikipedia. Nachträglich aus dem
Korpusinhalt bestimmt, weil die Herkunft damals nicht protokolliert wurde; seitdem hält
`texte.json` sie fest. Der frühere Verdacht, hier sei der Wikipedia-Dump 2023 gelaufen, kam vom
Pilot — *der* lief mit Wikipedia.

| Bedingung | Sink-Masse | Loss |
|---|---|---|
| A kohärent | 0,4066 | 3,47 |
| B wortgewürfelt | 0,3271 | 6,85 |
| B2 satzgewürfelt | 0,4060 | 3,62 |
| C disparat | 0,4124 | 4,43 |
| D Rauschen | 0,3335 | 8,55 |

**Nach Präregistrierung:**
- **H1 widerlegt** in der vorhergesagten Form. Nur C liegt über A (+0,006, p = 0,0002, sehr klein). B und D liegen deutlich darunter (etwa −0,08).
- **H2 nicht bestätigt.** Die Schwerpunktdifferenz C − B beträgt +0,19, 95%-KI −0,04 bis +0,42. Bekannter Konstruktionsfehler: Der Schwerpunkt wertet nur positive Differenzen aus, B wirkt aber fast nur negativ. Im Essay offen benennen.
- Im Pilot (n = 30) sah H2 noch signifikant aus (+0,68). Das illustriert, warum präregistriert wurde.

**Explorativ (aus `koepfe.csv`, nicht präregistriert):**
- ΔB und ΔD korrelieren mit r = 0,97: Die zerstörte Folge dominiert, zusätzlich zerstörte Art ändert fast nichts.
- Der Folge-Effekt wächst mit der Schichttiefe, von etwa 0 in Schicht 0–2 bis etwa −0,2 in Schicht 10–11. Die stärksten Köpfe sind (Schicht, Kopf) (8,1), (7,7), (10,8), (9,11), (7,11), jeweils etwa −0,5.
- ΔB und ΔC korrelieren mit r = −0,32: Art und Folge wirken gegenläufig. Das trennt die Dimensionen über die Richtung der Reaktion, nicht über den Ort.
- Eine kleine Gegenpopulation steigt unter B und D: (7,9), (8,5), (9,3), (7,4).
- B2 hat keinen Effekt; die Manipulation ist für zusammenhängende Absätze zu schwach.
- **Nachträgliche Deutung, noch zu prüfen:** Der Sink ist der *Ruheort der erfüllten Erwartung*. Wird die Folge zerstört, verlässt die Aufmerksamkeit ihn und beginnt zu suchen (Asynchronizität als sinnfordernd, Hausarbeit 5.4). Fehlt die Art, zieht sie sich leicht in ihn zurück. Nur als neue Hypothese verwenden, nicht als Befund.

### Pythia-160m, selber Korpus (n = 300)

Endstand bei step143000: A 0,2069 | B 0,1853 | B2 0,2057 | C 0,2103 | D 0,2073.

- Grundmuster wie bei GPT-2, abgeschwächt: B deutlich unter A (−0,022), C leicht darüber (+0,003, p = 0,019), B2 ohne Effekt. **H1 auch hier widerlegt.**
- Anders als bei GPT-2 liegt D praktisch auf A (+0,0004, KI enthält null) statt deutlich darunter.
- **H2 kehrt das Vorzeichen um:** C − B = −0,34 (95%-KI −0,54 bis −0,15), eindeutig negativ statt wie vorhergesagt positiv.

### Genese über elf Checkpoints (H3)

- Die Sink-Masse steigt wie vorhergesagt, von 0,0136 (step 0) auf 0,207 (step 143000).
- D − A ist bei step 1000–2000 kurz signifikant positiv, bei step 4000–100000 durchgehend signifikant **negativ** (−0,020 bis −0,030) und erst bei step 143000 wieder bei ~0.
- **H3 widerlegt:** Vorhergesagt war, D − A werde nach dem Anstieg positiv. Stattdessen bricht es mitten im Training ein. Neue Hypothese H3b dazu in `PRAEREG_2.md`, ungeprüft.

### Replikation mit Wikipedia-Korpus (`lauf_wiki`, n = 300)

Korpus: 600 Artikel, erste Version zwischen 2024-01-04 und 2026-09-10, 516 verschiedene Autoren. Von GPT-2 und Pythia nachweislich nicht gesehen. Herkunft protokolliert.

GPT-2: A 0,4280 | B 0,3457 | B2 0,4273 | C 0,4344 | D 0,3420.

- **R1 vollständig repliziert:** B − A = −0,082 und D − A = −0,086 (beide KI unter null), C − A = +0,0064 (KI über null), B2 − A ununterscheidbar von null.
- **R2 repliziert:** r(ΔB, ΔD) = +0,963 (gefordert > 0,8).
- *Explorativ:* Die kopfweisen Differenzkarten korrelieren **zwischen den beiden Korpora** mit r = 0,99 (B), 0,95 (C), 0,99 (D). Welcher Kopf wie reagiert, ist über zwei ganz verschiedene Textsorten hinweg fast identisch — spricht für eine Eigenschaft der Architektur, nicht des Korpus.
- **Die vorab notierte C-Erwartung ist gescheitert:** C − A sollte im einheitlichen Register kleiner ausfallen, ist aber minimal größer (+0,0064 statt +0,0058). Die Registererklärung trägt nicht.
- **Vorsicht bei H2:** Hier ist C − B = +0,56 (95%-KI +0,37 bis +0,76), nach der ursprünglichen Entscheidungsregel also bestätigt — im Hauptlauf war sie es nicht, bei Pythia mit umgekehrtem Vorzeichen. Das Maß ist dasselbe kaputte Maß; es wird nicht dadurch gültig, dass es diesmal passt. Erst reparieren, dann deuten.
- Der Loss in A ist mit 3,14 niedriger als im Crawl-Korpus (3,47), obwohl die Texte garantiert ungesehen sind. Niedriger Loss in A belegt also keine Memorierung, sondern Regelmäßigkeit des Registers. Das entkräftet die Kontrollannahme im Bericht (bei Pythia genauso: 3,05 statt 3,43).

Pythia bei step143000: A 0,2152 | B 0,1849 | B2 0,2134 | C 0,2186 | D 0,2079. Dasselbe Muster, H1 erneut widerlegt.

- **R3 repliziert:** D − A ist an sechs aufeinanderfolgenden Checkpoints (4000 bis 100000) signifikant negativ, KI vollständig unter null. Der Einbruch in der Trainingsmitte ist also kein Artefakt des Crawl-Korpus.
- Ein Unterschied zum Hauptlauf: Am Ende (step143000) bleibt D − A hier signifikant negativ (−0,0073), statt wie im Crawl-Korpus auf ~0 zurückzukehren.
- *Explorativ:* Auch bei Pythia korrelieren die Karten zwischen den Korpora hoch (r = 0,94 / 0,89 / 0,95 für B / C / D).
- **R2 nur für GPT-2:** r(ΔB, ΔD) = 0,963 bei GPT-2, aber 0,69 bei Pythia — in beiden Korpora praktisch gleich, also stabil, aber unter der geforderten 0,8. In `PRAEREG_3_replikation.md` ist R2 ohne Modellangabe formuliert; das ist ein Formulierungsfehler der Präregistrierung und wird so berichtet, nicht nachträglich auf GPT-2 eingeschränkt.

**Der H2-Befund über alle vier Läufe — das eigentliche Ergebnis:**

| Lauf | C − B | 95%-KI | nach Entscheidungsregel |
|---|---|---|---|
| `lauf` GPT-2 | +0,19 | −0,04 bis +0,42 | nicht bestätigt |
| `lauf` Pythia | −0,34 | −0,54 bis −0,15 | signifikant *gegen* die Vorhersage |
| `lauf_wiki` GPT-2 | +0,56 | +0,37 bis +0,76 | bestätigt |
| `lauf_wiki` Pythia | −0,04 | −0,19 bis +0,11 | nicht bestätigt |

Vier Läufe, vier verschiedene Antworten, darunter beide Vorzeichen signifikant. Das Maß misst nicht den Effekt, sondern das Rauschen seiner eigenen Konstruktion. H2 ist damit nicht offen, sondern **als Test wertlos, solange das Maß nicht ersetzt ist** — und der eine Lauf, der „bestätigt" sagt, ist das beste Argument dafür und nicht dagegen.

### Ersatzmaß |Δ| — explorativ, aber stabil

`centroid_abs` gewichtet die Schichttiefe mit |Δ| statt nur mit den positiven Anteilen. Dasselbe Bild in allen vier Läufen:

| Lauf | Schwerpunkt \|ΔB\| | Schwerpunkt \|ΔC\| | C − B | 95%-KI |
|---|---|---|---|---|
| `lauf` GPT-2 | 8,25 | 7,02 | −1,23 | −1,32 bis −1,11 |
| `lauf_wiki` GPT-2 | 8,21 | 6,89 | −1,32 | −1,41 bis −1,21 |
| `lauf` Pythia | 7,55 | 7,10 | −0,45 | −0,57 bis −0,34 |
| `lauf_wiki` Pythia | 7,53 | 7,04 | −0,49 | −0,58 bis −0,39 |

- **Die Dissoziation existiert, aber mit umgekehrter Polarität.** H2 behauptete zweierlei: dass die Schwerpunkte sich unterscheiden, und dass C *tiefer* liegt als B („Folge früh, Art spät"). Der erste Teil hält robust, der zweite ist umgekehrt: **Folge wirkt tief, Art wirkt flach.**
- Über zwei Korpora hinweg stimmen die Werte je Modell auf 0,1 Schichten überein; der Modellunterschied (GPT-2 ≈ −1,3, Pythia ≈ −0,5) ist größer als der Korpusunterschied.
- Passt zum älteren explorativen Befund, dass der Folge-Effekt mit der Schichttiefe wächst.
- **Status: explorativ.** Das Maß wurde nach Kenntnis der Daten definiert. Die Übereinstimmung über vier Läufe ist stark, ersetzt aber keinen konfirmatorischen Test — der muss aus einem neuen Lauf kommen (größere Modelle) und vorher präregistriert werden.
- Die Neuberechnung ließ alle präregistrierten Zahlen unverändert (36 Einfügungen, null Löschungen in den vier Berichten); die neuen Bootstrap-Ziehungen stehen im Code hinter allen alten.

### Schichtprofil (explorativ, Wikipedia-Korpus)

Mittleres |Δ| je Schicht, Schicht 0 bis 11:

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-2 \|ΔB\| | ,000 | ,000 | ,000 | ,014 | ,008 | ,069 | ,089 | ,143 | ,101 | ,140 | **,223** | ,200 |
| GPT-2 \|ΔC\| | ,000 | ,000 | ,003 | ,018 | ,009 | ,022 | ,010 | ,009 | ,004 | ,003 | ,017 | ,003 |
| Pythia \|ΔB\| | ,000 | ,004 | ,022 | ,000 | ,022 | ,054 | ,066 | ,037 | **,089** | ,063 | ,053 | ,005 |
| Pythia \|ΔC\| | ,001 | ,005 | ,005 | ,003 | ,010 | ,001 | ,011 | ,011 | ,011 | ,008 | ,035 | ,011 |

- **Die Tiefe ist nicht der Engpass.** Der Folge-Effekt steigt nicht bis zum Rand, sondern erreicht ein Maximum (GPT-2 Schicht 10, Pythia Schicht 8) und fällt danach ab, bei Pythia auf fast null. Die Folge wird innerhalb der verfügbaren Tiefe fertig konstituiert; „es fehlen die Durchläufe“ ist damit ausgeschlossen.
- **Der Größenunterschied ist der eigentliche Befund:** |ΔB| erreicht 0,223, |ΔC| nur 0,022 — Faktor zehn. Zerstörte Art stört die Sink-Masse fast nicht. „Art wirkt flach“ heißt vor allem: Art wirkt kaum, und das Wenige liegt früh.
- Das schränkt das Ersatzmaß-Ergebnis ein: Der Schwerpunkt von |ΔC| liegt stabil (7,02 / 6,89 / 7,10 / 7,04 über vier Läufe), aber über einem sehr kleinen Effekt.
- Schichten 0 bis 2 zeigen unter beiden Bedingungen praktisch keine Veränderung. Ob das Invarianz der frühen Repräsentation anzeigt oder nur, dass frühe Köpfe den Sink ohnehin inhaltsunabhängig bedienen, ist mit diesem Maß nicht zu entscheiden.
- **Offene theoretische Frage:** Wenn Sinn eine Folge von Gleichzeitigkeiten von *Gleichartigkeiten* ist, müsste zerstörte Gleichartigkeit die Warte erschüttern. Sie tut es fast nicht. Diese Asymmetrie muss die Theorie erklären — oder C misst nicht, was es messen soll (siehe C0).

## Offene Punkte

1. **Instrumente reparieren**, bevor weitere Modelle gemessen werden — sonst nur mehr nicht deutbare Zahlen:
   - H2 mit vorzeichenunabhängigem Maß neu fassen (z. B. Schwerpunkt von |Δ|). Dringend, weil das alte Maß je nach Lauf jedes Vorzeichen liefert.
   - B2 wirksam machen; Satzwürfeln tut messbar nichts.
   - C: Register ist als Erklärung raus (siehe Replikation), also offen, was C eigentlich variiert.
2. **H3b prüfen** (`PRAEREG_2.md`): B − A und C − A je Checkpoint mit Bootstrap-KI. Kein neuer Messlauf nötig, Rohdaten liegen in `lauf/ergebnisse/`.
3. **Größere Modelle** gegen die schwache externe Validität (GPT-2 medium/large, Pythia-410m/1.4b). Speicher ist der Engpass, weil `output_attentions` die vollen Matrizen erzwingt. Vorhersagen vorher festlegen, **alle** gelaufenen Modelle berichten.
4. **Architekturskizze** Detektiv-Tor (Essayplan 5/6), hängt an keinem Messergebnis.
5. **Weitere Erweiterungen**, jeweils mit eigener, vorher committeter Präregistrierung:
   - Dosis-Wirkung: Würfeln innerhalb von Fenstern mit 2, 4, 8, 16 Wörtern. Vorhersage: Der Sink sinkt stetig mit der Fenstergröße.
   - Wohin die Aufmerksamkeit geht: Attention-Entropie und mittlere Attention-Distanz pro Kopf messen. Vorhersage: In B und D verteilt sie sich auf nahe Tokens.
   - Gegenpopulation der steigenden Köpfe gesondert untersuchen.

## Arbeitsregeln

- Präregistrierte Hypothesen und Entscheidungsregeln nie nachträglich ändern. Neue Hypothesen kommen in eine eigene Datei (z. B. `PRAEREG_2.md`) und werden *vor* dem Lauf committet.
- Explorative Auswertungen im Bericht ausdrücklich als explorativ kennzeichnen.
- Pilot und Hauptlauf in getrennten Ordnern halten (`--ordner`).
- Nur frei lizenzierte Texte ins Repo committen (Wikipedia mit `quellen.csv` und `LIZENZ.md`).
- Nach Codeänderungen `python sinn_experiment.py selbsttest` ausführen.
- Literaturangaben vor der Verwendung prüfen. Ein früheres Mapping-Dokument enthielt falsche Titel und Zuordnungen (z. B. heißt Buckners Buch *From Deep Learning to Rational Machines*; Butlin et al. 2023 ist der Bewusstseinsbericht und hat nichts mit Kant zu tun).

## Literatur

Geprüft:
- Behrouz, Zhong, Mirrokni (2025): *Titans: Learning to Memorize at Test Time*, arXiv:2501.00663.
- Qiu et al. (2025): *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*, NeurIPS 2025 (Best Paper), arXiv:2505.06708.
- Bondarenko, Nagel, Blankevoort (2023): *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing*, NeurIPS 2023.
- Barbero et al. (2025): *Why do LLMs attend to the first token?*, COLM 2025.

Noch zu prüfen: Geva et al. 2021 (FFN als Key-Value-Speicher); Meng et al. 2022 (ROME, Faktenwissen in mittleren MLPs — wird in „Der Transformer in diesen Begriffen“ bereits verwendet, Titel und Autorenschaft noch nicht geprüft); Tenney et al. 2019 (BERT-Pipeline); Olsson et al. 2022 (Induction Heads); Anthropic 2024 (Scaling Monosemanticity, Golden Gate Claude); Bender & Koller 2020; Butlin, Long et al. 2023 (*Consciousness in Artificial Intelligence*); Millière & Buckner 2024; Gadamer, *Wahrheit und Methode*; Carhart-Harris & Friston 2019 (REBUS: Psychedelika *lockern* Priors, im Widerspruch zu Hausarbeit 6.2 prüfen).
