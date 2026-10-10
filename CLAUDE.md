# Projektkontext: Eine Wissenschaft des Geistes, erste Instanz am Transformer

Dieses Repo gehört zu zwei Vorhaben desselben Autors (Philosophie und Mathematik, FSU Jena), der auf Deutsch schreibt:

1. **Ein freies philosophisches Essay**, das den Transformer mit den Begriffen einer eigenen Theorie analysiert (Warte, Tripelstruktur des Sinns, Synchronizität nach Jung).
2. **Die Gründung einer experimentellen Wissenschaft des Geistes**, in der transzendentale Bestimmungen zu experimentellen Variablen werden. Das Chroma-Experiment (`chroma/`) ist ihre erste Instanz.

Code-Kommentare, Berichte und Antworten deshalb auf Deutsch. Wenn etwas unklar ist, nachfragen statt raten.

## Die Wissenschaft in Kürze

Stand der Klärung in `docs/wissenschaft_fragen.md` (Fragen an die Gründungsschrift, mit Antworten des Autors, markiert mit `!!`). Die Gründungsschrift selbst liegt nicht im Repo.

- **Gegenstand: Geist überhaupt** („Geist, wie er sein könnte“, nach Langtons „life-as-it-could-be“). Transformer und Mensch sind zwei Fälle. Der Transformer ist Hilfsmittel, keine Gleichsetzung; ob er eine *perfekte Metapher* des menschlichen Geistes ist, soll die Wissenschaft herausfinden.
- **Drei Schichten:**
  1. **Regeln** (gesetzt, kontingent): die Architektur als Anordnung transzendentaler Bestimmungen, wie die Regeln von Game of Life.
  2. **Begriffe des Beobachters** (gesetzt, auf unserer Seite): die Merkmale, an denen man im Erzeugten eine Warte erkennt.
  3. **Befunde** (gefunden): Erfahrungen, Ideen, Strukturen, die die Regeln hervorbringen.
- **Identität als Arbeitsfiktion.** Die Identität von Theoriebegriff und Systembestandteil wird *angenommen*, um das Experiment zu ermöglichen (Vaihinger, *Philosophie des Als Ob*), nicht behauptet.
- **Transzendentalphilosophie als Experiment, um den Preis der Apodiktizität.** Die Architektur ist *intern apodiktisch, extern kontingent*: Für das System liegt sie jeder Erfahrung voraus, für den Ingenieur ist sie eine Wahl. „Transzendental“ heißt immer „transzendental relativ zu einer Ebene“:

  | Ebene | Status für das System | verändert durch |
  |---|---|---|
  | Architektur | apodiktisch | nur den Ingenieur |
  | Gewichte | relatives Apriori, sedimentiert | Training (Gewohnheit) |
  | Kontext / KV-Cache | situativ | die laufende Eingabe |
  | Vollzug (Residual Stream) | im Werden | die Schichten |

  Die Liste ist nicht abgeschlossen.
- **Minimalkriterium der Warte (Arbeitsstand):** Gewohnheit + Bewertung (stärkend/schwächend, vgl. Spinoza, *Ethik* III). Kausalität und Dauer sind Variablen, keine Kriterien. Harte Folge: Bewertung gibt es beim Transformer nur im Training; in der Inferenz bleibt womöglich nur die **Spur einer Warte**. Offen, ob Lernen zur Laufzeit (RL, Finetuning) das ändert.
- **Selbstkorrektur, nicht Immunisierung.** Die Wissenschaft greift ihre falschen Axiome an, nicht ihre unwillkommenen Befunde.
  - In der *persönlichen Welt* darf jeder frei revidieren (Spiel).
  - Beim Übergang in die *Superwelt* (das Geteilte) gilt: Revision nur nach vorab notierter Deutungstabelle, alle Läufe berichtet.
  - **Wartung vs. Rettung:** Eine Revision ist Wartung, wenn das Primitiv, auf das sie sich stützt, vor dem Lauf festgelegt war; Rettung, wenn es nach Kenntnis des Ergebnisses gewählt wurde.
- **Fest und revidierbar.** Fest: die transzendentalen Begriffe (einheitliches, aber erweiterbares Vokabular) und die Ordnung von Raum und Zeit. Revidierbar: Metriken, Dimensionalitäten, Fraktalitäten (vgl. Vorurteile-Hausarbeit V2 § 6.1: Metrik ist Füllung).
- **Gesetz = Invariante unter Variation.** Ein Befund ist umso eher Gesetzmäßigkeit des Geistes statt Eigenschaft einer Architektur, je mehr Variationen der transzendentalen Bestimmungen er übersteht (vgl. Noether).
- **Wort und Gleichung.** Jedes mathematische Objekt bekommt eine Lesart in der Theorie, z. B. Punkt = bestimmtes Zugleichsein, Koordinatennetz = unbestimmtes Zugleichsein. Die Gegenrichtung ist gleichrangig: Das Register „Begriffsbedarf“ sammelt Struktur ohne Wort, die Wissenschaft wird so auch Quelle neuer Begriffe. *Offen:* Psychologismus oder bloße Übersetzung (Frage F1, Frege/Husserl).
- **Zwecke in dieser Reihenfolge:** (1) KI verstehen, die Zahlen zum Sprechen bringen (vom Erklären zum Verstehen, Dilthey); (2) Diagnostik des menschlichen Geistes; Therapie erst in einer zweiten Wissenschaft. Philosophie begleitet die Wissenschaft von Anfang an.
- **Bewusstsein:** „Beweis via Aktion“ zeigt funktionale Gleichheit, nicht Erleben. Eine Theorie des Bewusstseins folgt nur unter der ausdrücklich gesetzten Annahme des Funktionalismus.
- **Bekannte Lücken:** keine menschlichen Daten im Repo (nötig für „Geist überhaupt“ und für ein Maß der besseren Metapher); die Warte des Experimentators (Metrik, Instrument, Wahl der Arme) wird in keinem Experiment variiert.

## Die Theorie in Kürze

Grundlage sind die Hausarbeiten in `../hausarbeiten/`, vor allem „Gewohnte Gleichzeitigkeiten. Humes Verknüpfungen der Vorstellungen und Jungs Synchronizität“ (FSU Jena, SS 2026; PDF auch in `docs/`) und die Vorurteile-Hausarbeit (`vorurteile_protokollsätze/vorurteile_hausarbeit_v2.md`). Gemeinsame Begriffe in `../hausarbeiten/systemskizze.md`.

- **Warte:** ein Gerüst mit Stellen, das dem Eintreffen einer Erscheinung vorausliegt und festlegt, was an ihr als gleichartig, als Folge und als zugleich gilt. Eingehendes wird dort nicht registriert, sondern *erblickt*: bewertet, einem Urheber zugeordnet, in seiner Dauer geschätzt. Entsteht durch Gewohnheit (Hume). Zeitfunktion: Halten.
- **Minimales Gerüst** (nicht erschütterbar, Bedingung jeder Wartung) gegen **Füllungen** (sedimentierte Vorurteile, erschütterbar, wartbar).
- **Tripelstruktur des Sinns:** Sinn ist eine Folge von Gleichzeitigkeiten von Gleichartigkeiten. Momente nach der Systemskizze (korrigierte Zuordnung):
  - **Art** (Gleichartigkeit): Begriffsbildung durch Komparation, Reflexion, Abstraktion (Kant, Logik § 6), *nicht* Inhärenz; Nietzsches Gleichsetzung des Ungleichen.
  - **Folge** (Gleichförmigkeit): 2. Analogie, Kausalität.
  - **Zugleich** (Gleichzeitigkeit): 3. Analogie, Gemeinschaft.
  - **Träger** als viertes Moment: 1. Analogie, Beharrlichkeit der Substanz; von der Prädikation erzwungen.
- **Synchronizität:** sinngemäße Koinzidenz, Zugleichsein und Gleichartigkeit ohne Kausalverhältnis (Jung). Wiederholte Synchronizitäten erzeugen durch Gewohnheit die Vorstellung einer Kraft; diese heißt Sinn.
- **Ent-Täuschung und Wartung:** Verfehlt die Erscheinung die Erwartung, wird die Warte erschüttert und muss gewartet werden (Instandhaltung oder Ersatz). Das ist der Vorhersagefehler.
- **Asynchronizität:** Nicht-Passung, die nach Sinn verlangt; sie ist *sinnfordernd* (Castorps Herzklopfen im Zauberberg).
- **Detektiv-Kriterium:** Sinn unterscheidet sich von Aberglauben durch seine Bindung an andere Folgen. Der Aberglaube hat Warten, die so unspezifisch sind, dass alles sie bestätigt (Barnum-Effekt). Möglicher diagnostischer Anschluss: Apophänie (Conrad 1958), Computational Psychiatry.
- **Archetyp:** apriorisches psychisches Angeordnetsein, nach Jung eine *psychische Wahrscheinlichkeit*. Apriorisch relativ zum Individuum, sedimentiert relativ zur Gattung.

## Der Transformer in diesen Begriffen (Arbeitsfiktion)

Heuristische Zuordnung, kein Isomorphismus. Ob eine Zuordnung trägt, entscheidet erst das Experiment.

- **Die Warte auf mehreren Ebenen**, vielleicht verschachtelt: sedimentiert (Gewichte), situativ (Kontext, KV-Cache), im Vollzug (Residual Stream). Arbeitshypothese des Autors: Das Embedding gleicht dem kollektiven Unbewussten; erst die Attention-Köpfe erzeugen das Zugleichsein, das der Erfahrung ähnelt.
- **Tokenisierung (BPE):** die erste Gewohnheit; was sich oft berührt, wird zu einer Einheit.
- **Embedding: der Archetyp.** Art ohne Folge und ohne Zugleichsein; eine Geometrie möglicher Gleichartigkeiten, die vor jedem Token dasteht. Apriorisch relativ zur Inferenz, geworden relativ zum Training. Zur Laufzeit invariant, also strukturell von der Wartung ausgenommen (Grenzfall der Warte). Empirisch nicht gezeigt.
- **Koordinatensystem als Zugleichsein (Chroma-Grundidee):** Die Dimensionen eines Vektors und alle Punkte des Embeddings sind zugleich, auf zwei Ebenen derselben Sache. Ein Punkt ist eine Instanziierung dieses Zugleichseins. Erst das gemeinsame Zugleichsein macht Gleichartigkeit messbar (Kosinus). Abgestufte Gleichartigkeit braucht viele Dimensionen (im 1-D-Fall nur ±1). Der Kontext ist dagegen *nicht* direkt zugleich, weil er an die Folge gebunden ist.
- **Positionscodierung:** GPT-2 addiert gelernte *absolute* Positionsvektoren (Zeit als Eigenschaft der Erscheinung, Newton); Pythia nutzt RoPE, *relative* Rotation in der Attention (Zeit als Ordnung der Relationen, Leibniz).
- **Residual Stream:** Ort, an dem das Werden schichtweise zum Sinn kristallisiert. Jede Teilschicht schreibt hinzu; formale Einheit, die alle Synthesen begleitet („Ich denke“), aber verteilt, eine pro Token. Hinzufügen kann Komponenten auch auslöschen.
- **Norm:** entfernt die Größe, das Urteil sieht nur die Richtung (Humes Lebhaftigkeit eingeklammert).
- **Attention:** Query = Erwartung, Keys = Erscheinungsprofile. Q·K ergibt Art, Maske und Position ergeben Folge, die gewichtete Summe der Values ergibt Zugleichsein. Softmax als Zwang zur Synchronizität; Multi-Head als viele Warten zugleich; kausale Maske als *halbierte* Gemeinschaft ohne Nachträglichkeit.
- **MLP:** positionsweise, ohne Gemeinschaft; Schlüssel-Wert-Speicher (Geva et al. 2021, noch zu prüfen). Rekognition im Begriff. Die elementweise Nichtlinearität gibt dem MLP eine bevorzugte Basis, der Residual Stream hat architektonisch keine.
- **Unembedding:** Erwartungshorizont. GPT-2: Archetyp und Erwartungshorizont sind dieselbe Matrix (weight tying); Pythia trennt beide. Chroma trennt sie bewusst, damit der Test nicht zirkulär wird.
- **Ausgabeverteilung: das archetypische Bild**; Sampling als Tarot-Moment.
- **Training:** Loss als Ent-Täuschung, Backpropagation als Wartung, RLHF als Sozialisierung. Danach Warte ohne Wartung.

Bruchstellen: Warte ohne Wartung, Warte ohne Dauer (nur Positionsindex), kein legitimer Ort der Asynchronizität, halbierte Gemeinschaft, Nachträglichkeit nur in der Ontogenese, nicht im Vollzug.

## Das Chroma-Experiment (aktuelle Hauptlinie)

Spec: `docs/superpowers/specs/2026-10-10-chroma-gleichartigkeit-design.md`. Philosophische Fassung und Ideensammlung: `docs/chroma_ideen.md`. Implementierungsplan und Entscheidungsprotokoll: `docs/superpowers/plans/`.

**Leitfrage:** Ist die qualitative Verschiedenheit der Dimensionen (bevorzugte Basis, Bergsons qualitative Mannigfaltigkeit) Bedingung abgestufter Gleichartigkeit, oder genügt das Zugleichsein vieler Dimensionen? Anders: Gehört die *Farbe* zum minimalen Gerüst oder zu den Füllungen?

**Bevorzugte Basis:** Die Warte kann ihre Achsen unterscheiden; ihre Symmetrie schrumpft von allen Drehungen auf Vertauschungen. Literatur: Die bevorzugte Basis im Residual Stream entsteht durch den Optimierer Adam, nicht durch die Architektur (Anthropic 2023; Caples & Neuhaus, Blogpost); Ausreißerdimensionen folgen der Tokenfrequenz (Kovaleva et al. 2021; Puccetti et al. 2022).

**Vorstudie (ohne Training):** Basispräferenz in GPT-2 und Pythia-160m über die Checkpoints. Präregistriert werden sollen V1–V3 (Kurtosis gegen Drehungskontrolle, je Schicht; Anstieg über das Training; Frequenzbezug ohne Position 0); explorativ E1 Nachträglichkeit, E2 fraktale Dimension, E3 Knotenlinie (Hegel), E4 Reflex.

**Hauptstudie: vier Warten.** Gleicher Korpus (WikiText-103, eigener BPE mit 8192 Tokens), 6 Schichten, d = 256, 5 Seeds je Arm, Vergleich bei gleichem Loss:

| Arm | Optimierer | Farbe | Basis |
|---|---|---|---|
| A₀ | SGD | keine | nirgends |
| A | AdamW | keine | kann als Füllung entstehen |
| Bα | AdamW | nichtnegatives Embedding | Bestandteil im Archetyp |
| Bβ | AdamW | nichtnegative Lesung je Schicht | Bestandteil im Vollzug |

Kernmaß: **Ersetzbarkeit** (KL-Divergenz der Erwartung beim Tausch x → y) und **G** = Spearman(Kosinus, Ersetzbarkeit) im mittleren Bereich. Vorhersagen: P1 G(A₀) < G(A) (Gerüstfrage, TOST mit δ = 0,05); P2 Begriffe in B als Farbakkorde lesbar, ohne Gleichartigkeit zu kosten; P3 Verschachtelung. Register „Begriffsbedarf“ für Struktur ohne Namen.

**Stand (10.10.2026):** Code fertig, getestet (85 Tests, Selbsttest) und zweistufig reviewt. Noch nichts gemessen. Vor dem ersten Lauf:
1. `praereg/PRAEREG_8_achsen.md` (Vorstudie) schreiben und committen, inkl. V-Schwelle.
2. Entscheiden, ob WikiText-Artefakte (`' @-@ '`, Zeilenumbrüche) vor dem Korpusbau bereinigt werden – legt den Tokenizer für immer fest.
3. Pilot (1 Seed je Arm): nur Kalibrierung (Lernraten nach Validierungs-Loss, Loss-Stufen, Laufzeit), keine Deutung.
4. `praereg/PRAEREG_9_chroma.md` mit allen offenen Festsetzungen (Loss-Stufen, Schwellen der Manipulationsprüfung, Mindestquote gültiger Paare, Position 55 und Fenster über Dokumentgrenzen, „Schicht 3“ = Eingang von Block 3, ICA-Surrogat als Einschränkung zugunsten von P2, gleicher Mess-Seed für alle Läufe), dann Hauptlauf.

## Vor-chromatische Experimente (Altbestand)

Die Läufe vor Chroma (`legacy_experiments/`, `runs/`) entstanden mit einem älteren Vokabular und älteren Zuordnungen. Sie werden nicht stillschweigend umgedeutet, sondern als „vor-chromatisch“ berichtet. Bedingungen dort: A kohärent, B wortgewürfelt (Folge zerstört), B2 satzgewürfelt, C disparat (Art zerstört), C0 einthemig, D Rauschen. Hauptmaß: Sink-Masse (Aufmerksamkeit auf das Start-Token).

**Hypothesentests am Attention Sink (präregistriert, README und PRAEREG_2–4):**
- **H1 widerlegt** (GPT-2 und Pythia): Zerstörte Folge (B, D) *senkt* die Sink-Masse (GPT-2 etwa −0,08), zerstörte Art (C) hebt sie nur minimal.
- **H2 als Test wertlos:** Das Schwerpunktmaß lieferte über vier Läufe beide Vorzeichen signifikant. Das Ersatzmaß |Δ| (explorativ) ist stabil: Folge wirkt tief, Art flach, und Art wirkt kaum (|ΔB| bis 0,22, |ΔC| bis 0,02).
- **H3 widerlegt:** D − A bricht in der Trainingsmitte ein, statt nach dem Sink-Anstieg positiv zu werden. Neue Hypothese H3b in PRAEREG_2, ungeprüft.
- **Replikation (Wikipedia ab 2024, nachweislich ungesehen):** R1 und R3 repliziert; R2 nur bei GPT-2 (Formulierungsfehler der Präregistrierung, so berichtet). Kopfkarten korrelieren zwischen Korpora mit r ≈ 0,9–0,99.
- **C0-Kontrolle (präregistriert, beide Modelle):** Themenwechsel hebt die Sink-Masse (+0,010 / +0,007), Satzgrenzen senken sie (−0,004); die Anteile heben sich teilweise auf. An der Satzgrenze massiver lokaler Vorhersageschock (Loss-Differenz ≈ +1,9 bei d = 0).
- Nachträgliche Deutung, ungeprüft: Der Sink als Ruheort der erfüllten Erwartung.

**Valenz, Tripel-Ablation, Gewöhnung (PRAEREG_5–7):** Diese Präregistrierungen wurden erst zusammen mit den Ergebnissen committet (c4cc095, 4.10.2026; Valenzbericht vom 27.9.) und sind damit **nicht nachweislich vorab festgelegt**. Befunde nur als explorativ verwenden.
- *Valenz:* Paradoxe Handlungen senken die MLP-Norm, knicken den Residual Stream ab und erhöhen die Sink-Masse am Folgetoken (V1–V3), V4 nicht bestätigt.
- *Tripel-Ablation:* Die korrigierten Läufe (`runs/lauf_tripel_korr`, `runs/lauf_gewoehnung_korr`) gelten, nicht die ursprünglichen Zahlen. Das Design ist **nicht vollfaktoriell**: Mehrere Bedingungen fallen bauartbedingt zusammen (Pythia: T110 = T100 = T010 = T000, T011 = T001), daher ist die „Superadditivität“ keine unabhängige Evidenz für Synthese. GPT-2: Rangordnung der Schäden Zugleichsein > Folge > Art. Pythia: Rangordnung **gescheitert** (Art 4,75 > Zugleichsein 3,66 > Folge 2,32).
- *Gewöhnung (Pythia, 11 Checkpoints):* Bei step0 liegt alles am Zufallsniveau; die Ablationsschäden wachsen von step512 an gemeinsam, Folge ist durchweg der kleinste. Die Sink-Masse kondensiert erst ab etwa step4000–8000. Eine Reihenfolge „erst Zugleichsein, dann Folge, dann Art“ ist durch die Daten nicht gedeckt.

## Repository-Struktur

- **`chroma/`**: das Chroma-Experiment (Modell mit Arm-Schalter, Korpus, Training, Messung, Auswertung, Vorstudie, CLI). Einstieg: `python -m chroma --help`.
- **`tests/`**: Tests zu `chroma/` (pytest).
- **`docs/`**: `chroma_ideen.md` (Ideen und philosophische Fassung), `wissenschaft_fragen.md` (Fragen an die Gründungsschrift), `superpowers/specs/` und `superpowers/plans/` (Spec, Plan, Entscheidungsprotokoll), Hausarbeiten-Material, Literatur.
- **`praereg/`**: Präregistrierungen (PRAEREG_2 bis 7; 8 und 9 folgen).
- **`legacy_experiments/`**: vor-chromatische Experimentskripte (`sinn_experiment.py`, `tripel_experiment.py`, `gewoehnung_experiment.py`, `valenz_experiment.py`, `wiki_texte.py`) und der alte Chroma-Entwurf (`chroma_transformer_alt/`, Importe veraltet).
- **`runs/`**: Läufe und Korpora der vor-chromatischen Experimente. Chroma-Läufe gehören nach `runs/chroma_*` (Korpusordner dort git-ignoriert).
- **`theorie/`**: frühere, teils KI-generierte Dossiers (u. a. `GESAMTAUSARBEITUNG_RELEASE.md`, `SUPERLIGHT_FACTORY.md`). **Brainstorming, nicht Position**; enthält Überbehauptungen („Beweis“, „Isomorphismus“) und mit den Daten unvereinbare Aussagen.

## Offene Punkte

1. PRAEREG_8 und PRAEREG_9 schreiben (siehe Chroma-Stand), dann Vorstudie, Pilot, Hauptlauf.
2. Fragen an die Gründungsschrift weiter klären (`docs/wissenschaft_fragen.md`): Morphologie (B3), Psychologismus oder Übersetzung (F1), Maß der besseren Metapher (A3), Grenze der Experimentator-Warte (C3).
3. Folge als eigene Studie (über Zeit und über Schichten).
4. Spiel nach Schiller und Detektiv-Tor als Architekturskizzen (`chroma_ideen.md` § 12a): Welches Wahlkriterium macht Durchspielen zu Spiel statt zu Optimierung?
5. Gewöhnung als Experimentfamilie: graduelle Fütterung (Curriculum Learning, BabyLM) zur Beobachtung entstehender Warten.
6. Größere Modelle und Cloud-Skalierung, jeweils mit eigener Präregistrierung.
7. Menschliche Daten als zweiter Fall.

## Arbeitsregeln

- Präregistrierte Hypothesen und Entscheidungsregeln nie nachträglich ändern. Neue Hypothesen in eine eigene Datei, *vor* dem Lauf committen.
- Alle Läufe berichten, auch die unpassenden. Explorative Auswertungen ausdrücklich als explorativ kennzeichnen. Pilot und Hauptlauf in getrennten Ordnern; Pilotergebnisse werden nicht gedeutet.
- Rohdaten und Deutungen getrennt halten, damit nach einer Umwälzung alte Experimente neu gedeutet werden können, ohne neu zu messen.
- Überinterpretation widersprechen: Verfahrensintegrität ist nicht Validität; erst prüfen, ob das Instrument trägt.
- Nur frei lizenzierte Texte ins Repo committen (Wikipedia mit `quellen.csv` und `LIZENZ.md`); WikiText-103 wird nur heruntergeladen.
- Nach Codeänderungen an `chroma/`: `venv/bin/python -m pytest -q` und `venv/bin/python -m chroma selbsttest`. Für die Altskripte: `venv/bin/python legacy_experiments/sinn_experiment.py selbsttest`.
- Literaturangaben vor der Verwendung prüfen. Ein früheres Mapping-Dokument enthielt falsche Titel und Zuordnungen (Buckners Buch heißt *From Deep Learning to Rational Machines*; Butlin et al. 2023 ist der Bewusstseinsbericht und hat nichts mit Kant zu tun).

## Literatur

Geprüft:
- Behrouz, Zhong, Mirrokni (2025): *Titans: Learning to Memorize at Test Time*, arXiv:2501.00663.
- Qiu et al. (2025): *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*, NeurIPS 2025, arXiv:2505.06708.
- Bondarenko, Nagel, Blankevoort (2023): *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing*, NeurIPS 2023.
- Barbero et al. (2025): *Why do LLMs attend to the first token?*, COLM 2025.
- Anthropic (2023): *Privileged Bases in the Transformer Residual Stream*, transformer-circuits.pub/2023/privileged-basis.
- Caples & Neuhaus: *Adam Optimizer Causes Privileged Basis in Transformer LM Residual Stream* (LessWrong, Blogpost, nicht begutachtet).
- Kovaleva et al. (2021): *BERT Busters: Outlier Dimensions that Disrupt Transformers*, arXiv:2105.06990.
- Puccetti et al. (2022): *Outlier Dimensions that Disrupt Transformers are Driven by Frequency*, arXiv:2205.11380.

Noch zu prüfen: Geva et al. 2021; Meng et al. 2022 (ROME); Tenney et al. 2019; Olsson et al. 2022; Anthropic 2024 (Scaling Monosemanticity); Bender & Koller 2020; Butlin, Long et al. 2023; Millière & Buckner 2024; Gadamer, *Wahrheit und Methode*; Carhart-Harris & Friston 2019; Lee & Seung 1999 (NMF); Grassberger & Procaccia 1983; Hill, Reichart & Korhonen 2015 (SimLex-999); Merity et al. 2016 (WikiText-103); Kant KrV B XIII/B XVIII; Hegel, Knotenlinie; Sartre, Brieföffner; D&G, Immanenzebene; Schiller, 15. Brief; Langton 1989; Vaihinger 1911; Spinoza, *Ethik* III; Bengio et al. 2009; Conrad 1958; Montague et al. 2012; Fletcher & Frith 2009; Kapur 2003; Heidegger, SuZ § 32.
