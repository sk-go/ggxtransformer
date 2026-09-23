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
- **Embedding:** die erste Wertung, rohe Gleichartigkeit.
- **Residual Stream:** der Ort, an dem das Werden schichtweise zum Sinn kristallisiert.
- **Attention:** die Query ist die Erwartung, die Keys sind Erscheinungsprofile. Q·K ergibt Art, kausale Maske und RoPE ergeben Folge (Folge wird zu Geometrie), die gewichtete Summe der Values ergibt Zugleichsein.
- **Kausale Maske:** eine *halbierte* Gemeinschaft, ohne Wechselwirkung und ohne Nachträglichkeit.
- **Softmax:** Zwang zur Synchronizität, da die Aufmerksamkeit sich zu 1 summieren muss.
- **Multi-Head:** viele Warten zugleich.
- **MLP:** Gewohnheit ohne Gemeinschaft, positionsweise; nach Geva et al. 2021 Schlüssel-Wert-Speicher.
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
- `PRAEREG_3_replikation.md`: Replikation des Hauptlaufs mit Wikipedia-Korpus (R1–R3).

## Experiment: Stand

Bedingungen: A kohärent, B wortgewürfelt (Folge zerstört), B2 satzgewürfelt, C disparat (Art zerstört), D Rauschen (beides zerstört). Messgrößen: Sink-Masse pro Kopf (Aufmerksamkeit auf das feste Start-Token, Query-Positionen ab 8), Ausgabe-Entropie, Loss. Parameter: N = 256, n = 300, Seed 1234.

### Hauptlauf GPT-2 (n = 300)

Korpus: englische Web-Crawl-Texte (FineWeb-artig), *nicht* Wikipedia. Nachträglich aus dem
Korpusinhalt bestimmt, weil die Herkunft damals nicht protokolliert wurde; seitdem hält
`texte.json` sie fest. Für die Kontaminationsfrage ist das günstig, weil ein Crawl von 2024
hinter den Trainingsdaten beider Modelle liegt.

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
- B2 hat keinen Effekt; die Manipulation ist für diese Absätze zu schwach.
- **Nachträgliche Deutung, noch zu prüfen:** Der Sink ist der *Ruheort der erfüllten Erwartung*. Wird die Folge zerstört, verlässt die Aufmerksamkeit ihn und beginnt zu suchen (Asynchronizität als sinnfordernd, Hausarbeit 5.4). Fehlt die Art, zieht sie sich leicht in ihn zurück. Nur als neue Hypothese verwenden, nicht als Befund.

### Pythia-160m, selber Korpus (n = 300)

Endstand bei step143000:

| Bedingung | Sink-Masse | Loss |
|---|---|---|
| A kohärent | 0,2069 | 3,43 |
| B wortgewürfelt | 0,1853 | 6,76 |
| B2 satzgewürfelt | 0,2057 | 3,59 |
| C disparat | 0,2103 | 4,27 |
| D Rauschen | 0,2073 | 9,08 |

- Das Grundmuster von GPT-2 wiederholt sich abgeschwächt: B deutlich unter A (−0,022), C leicht darüber (+0,003, p = 0,019), B2 ohne Effekt. **H1 auch hier widerlegt.**
- Anders als bei GPT-2 liegt D praktisch auf A (+0,0004, KI enthält null), statt deutlich darunter.
- **H2 kehrt das Vorzeichen um:** Schwerpunktdifferenz C − B = −0,34 (95%-KI −0,54 bis −0,15), also eindeutig negativ statt wie vorhergesagt positiv. Bei GPT-2 war sie +0,19 mit KI über null hinweg. Die Schwerpunktkonstruktion trägt nicht.

### Genese über elf Checkpoints (H3)

- Die Sink-Masse steigt wie vorhergesagt, von 0,0136 (step 0) auf 0,207 (step 143000).
- D − A ist bei step 1000–2000 kurz signifikant positiv, bei step 4000–100000 durchgehend signifikant **negativ** (−0,020 bis −0,030) und erst bei step 143000 wieder bei ~0 (KI enthält null).
- **H3 widerlegt:** Vorhergesagt war, dass D − A nach dem Anstieg positiv wird. Stattdessen bricht es mitten im Training ein. Neue Hypothese H3b dazu in `PRAEREG_2.md`: Der Einbruch könnte daher rühren, dass sich der Folge-Effekt früher ausbildet als der Art-Effekt. Ungeprüft.

## Offene Punkte

1. **Replikation mit Wikipedia-Korpus** (`PRAEREG_3_replikation.md`, Ordner `lauf_wiki`): Artikel ab 2024 über `wiki_texte.py`, damit die Texte garantiert hinter den Trainingsdaten beider Modelle liegen und die Lizenz für das Repo geklärt ist. Erwartet wird ein schwächerer C-Effekt, weil Wikipedia stilistisch einheitlich ist.
2. **H3b prüfen** (`PRAEREG_2.md`): B − A und C − A je Checkpoint mit Bootstrap-KI. Braucht keinen neuen Messlauf, die Rohdaten liegen in `lauf/ergebnisse/`.
3. **Erweiterungen**, jeweils mit eigener, vorher committeter Präregistrierung:
   - Dosis-Wirkung: Würfeln innerhalb von Fenstern mit 2, 4, 8, 16 Wörtern. Vorhersage: Der Sink sinkt stetig mit der Fenstergröße.
   - Wohin die Aufmerksamkeit geht: Attention-Entropie und mittlere Attention-Distanz pro Kopf messen. Vorhersage: In B und D verteilt sie sich auf nahe Tokens.
   - Gegenpopulation der steigenden Köpfe gesondert untersuchen.
   - Pythia-Dynamik: Entstehen der Folge-Effekt und der Art-Effekt zu verschiedenen Zeitpunkten?
   - H2 mit einem vorzeichenunabhängigen Maß neu formulieren (z. B. Schwerpunkt von |Δ|).

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

Noch zu prüfen: Geva et al. 2021 (FFN als Key-Value-Speicher); Tenney et al. 2019 (BERT-Pipeline); Olsson et al. 2022 (Induction Heads); Anthropic 2024 (Scaling Monosemanticity, Golden Gate Claude); Bender & Koller 2020; Butlin, Long et al. 2023 (*Consciousness in Artificial Intelligence*); Millière & Buckner 2024; Gadamer, *Wahrheit und Methode*; Carhart-Harris & Friston 2019 (REBUS: Psychedelika *lockern* Priors, im Widerspruch zu Hausarbeit 6.2 prüfen).
