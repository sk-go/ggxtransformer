# Chroma-Transformer: Ideensammlung

*Stand: 10.10.2026. Festgehalten aus dem Brainstorming zum Neubau des Chroma-Modells. Arbeitsmaterial, keine Präregistrierung und kein Befund. Der Altbestand (`theorie/`, alter `chroma_transformer/`, `GESAMTAUSARBEITUNG_RELEASE.md`) zählt ebenfalls nur als Brainstorming.*

---

## 1. Grundidee

Ausgangspunkt ist GG 4.1: „Jede Vorliebe bildet eine Dimension des Erwartungsraumes. So ergibt sich ein persönliches, multidimensionales Koordinatensystem, das das Werden überspannt.“

- **Ein Koordinatensystem ist ein Zugleichsein.** Alle Achsen werden auf einmal gehalten; das ist das Netz der Warte.
- **Ein Punkt ist eine Instanziierung dieses Zugleichseins:** eine erblickte Erscheinung, deren Koordinaten alle auf einmal bestimmt sind.
- **Gleichartigkeit ist die Relation zwischen Punkten.** Weil sie im selben Zugleichsein liegen, sind sie vergleichbar. Darum lässt sich die Kosinus-Ähnlichkeit als Gleichartigkeit lesen. !!wir vergleichen aber auch menschen, die in folge aufeinander uns treffen, sodass hier schon gewohnheit eine rolle spielt, an einem späteren zeitpunkt kann eine andere dimension zugleich sein, die vorher nicht da war und somit im nachhinein die ähnlichkeit beeinflusst

  > **Reaktion:** Gleichartigkeit ist nicht statisch: Eine später zugleich werdende Dimension ändert die Ähnlichkeit nachträglich. Prüfbar über die Pythia-Checkpoints: Ordnet sich die Ähnlichkeitsrangfolge fester Wortpaare im Training um, und wächst die effektive Zahl der Dimensionen? In der Inferenz ist das ausgeschlossen (kausale Maske, festes Embedding). **Nachträglichkeit gibt es nur in der Ontogenese, nicht im Vollzug** – eine Bruchstelle fürs Essay. → explorativ in die Vorstudie (E1).

- These: Ähnlichkeitsmessung wird von der Theorie nicht bloß benutzt, die Theorie sagt, was sie möglich macht. **Ohne gemeinsames Zugleichsein keine Gleichartigkeit.**

## 2. Zugleichsein auf zwei Ebenen

- **Im Vektor:** Die Dimensionen eines Punktes sind zugleich.
- **Im Embedding:** Alle Punkte teilen dieses eine Zugleichsein.
- Beides sind zwei Ebenen derselben Sache.
- Der Kontext (das Fenster) ist **nicht** direkt zugleich, weil er an die syntaktische Folge gebunden ist. Damit wird die alte Zuordnung „Zugleichsein = Σ αV“ fraglich.

## 3. Verschachtelte Warten (Arbeitshypothese, offen)

- Erste Intuition: Das ganze Modell ist eine Warte.
- Genauer: Erst die Attention-Köpfe erzeugen das Zugleichsein, das der Erfahrung ähnelt.
- Das Embedding gleicht eher dem kollektiven Unbewussten: sedimentiert, der Gattung zugehörig, nicht dem Individuum.
- Möglich: verschachtelte Warten (Embedding, Residual Stream je Schicht, QK-Raum je Kopf, Unembedding).

## 4. Dimensionen und die „perfekte Metapher“

- Es gibt einzelne Dimensionen, aber sie sind nicht notwendig in menschlichen Bedeutungen formulierbar.
- Vielleicht erreichbar durch Regression der Begriffe auf ihre Hauptkomponenten.
- Weil der Transformer aus Text lernt, ist er womöglich keine **perfekte Metapher** menschlicher Erfahrung; seine Grunddimensionen können abweichen.
!! hier kann es auch vorkommen, dass diese dimensionen eine fraktale natur haben, ähnlich wie deleuzes immanenzebene

  > **Reaktion:** Messbar als intrinsische bzw. Korrelationsdimension der Punktwolke, *über mehrere Skalen*. „Fraktal“ hieße: nicht ganzzahlig und skalenabhängig. **Prüfen:** ob D&G die Immanenzebene in *Was ist Philosophie?* wörtlich fraktal nennen. → explorativ in die Vorstudie (E2).


## 5. Zwei Thesen, die die Mathematik trennt

1. **Abgestufte Gleichartigkeit braucht viele Dimensionen.** Im Eindimensionalen kennt der Kosinus nur +1 und −1: gleich oder entgegengesetzt, keine Grade. Grade entstehen erst, wenn mehrere Dimensionen zugleich gehalten werden. Mathematisch sicher. !! vergleich mit reflexen, impuls gegeben, handlung ausgeführt, vollständig binär, wobei reflexe auch in unterschiedlichen maßen hervorgerufen werden können, auch sind sie durch gewöhnung (körper als gewöhnung?) zu einer binarität geworden, sie wurden entkompliziert

  > **Reaktion:** Unterscheidung: Ein Reflex mit verschiedener *Intensität* ist eindimensional, aber Intensität ist Größe, nicht Richtung – Humes Lebhaftigkeit ohne Art (im Modell: die Größe, die RMSNorm entfernt). Reflexe haben also Grade der Intensität, keine Grade der Gleichartigkeit; These 1 hält. Interessant ist die Gegenrichtung: **Gewohnheit kann Dimensionen auch abbauen** (GG 4.1: das Tier, viel Reflex, wenig Reflexion). Prüfbar: Sind Repräsentationen sehr häufiger Tokens niedrigdimensionaler? → explorativ in die Vorstudie (E4).

2. **Die Dimensionen müssen qualitativ verschieden sein** (Bergsons qualitative Mannigfaltigkeit). Der Kosinus braucht das *nicht*: Er ist rotationsinvariant, die Achsen sind gegeneinander austauschbar. !! metrik ist nach dem neusten entwurf von protokollsätze in ../hausarbeiten auch ein vorurteil, das andere vorurteile ermöglicht

  > **Reaktion:** V2 § 6.1: „Die Ordnung von Raum und Zeit gehört zu den Bestandteilen, ihre Metrik zu den revidierbaren Füllungen.“ Folge fürs Design: **Der Kosinus ist unser Vorurteil, nicht das des Modells** – die externalisierte Warte des Experimentators, das Messinstrument. G misst also, wie weit *unsere* Metrik mit der *eigenen* Behandlung durch das Modell (Ersetzbarkeit) übereinstimmt. Zwei Ergänzungen: (1) die **eigene Metrik des Modells**, die gelernte Bilinearform W_Q·W_Kᵀ der Attention, eine durch Gewohnheit sedimentierte Füllung; (2) ein **topologisches Maß** (Überlappung der nächsten Nachbarn) als Gegenstück zur Ordnung als Bestandteil, passend zu GG 5 (Geisteswissenschaft topologisch, Naturwissenschaft metrisch). → beide als sekundäre explorative Maße in die Hauptstudie.


Im Transformer:
- Der Residual Stream und das Embedding liegen nahe am austauschbaren Fall: keine bevorzugte Basis, eine *quantitative*, homogene Mannigfaltigkeit. !!parallele zu hegel? dass quantität in qualität umschlagen kann?

  > **Reaktion:** Hegels Begriff für den sprunghaften Umschlag ist die **„Knotenlinie von Maßverhältnissen“** (WdL, Lehre vom Sein; **Stelle prüfen**) – darin steckt der *Knoten*. Prüfbar: Steigt die Kurtosis über die Checkpoints allmählich oder springt sie? Einschränkung: Bei A treibt nicht bloße Quantität den Umschlag, sondern Adams Regel. → explorativ in die Vorstudie (E3).

- Die MLP-Neuronen haben eine bevorzugte Basis, weil ihre Nichtlinearität auf jede Dimension einzeln wirkt.
- Die Architektur enthält also beide Arten von Mannigfaltigkeit an verschiedenen Stellen.

**Entscheidung:** Beide Thesen werden im Vergleich geprüft (rotationsinvariantes Modell gegen Modell mit bevorzugter Basis). Vorhersagen:
- **(i) stark, Emergenz:** Gleichartigkeit braucht qualitative Verschiedenheit so sehr, dass ein Modell ohne eingebaute Basis sich eine bildet. Kann scheitern.
- **(ii) schwach, Lesbarkeit:** Mit eingebauter Basis werden Begriffe als Farbakkorde aus wenigen Dimensionen lesbar, bei etwa gleicher Leistung.

## 6. Literaturbefund: Die Basis kommt durch den Optimierer

- Anthropic, *Privileged Bases in the Transformer Residual Stream* (Transformer Circuits, 2023): https://transformer-circuits.pub/2023/privileged-basis/ — Hypothese: Adam erzeugt die bevorzugte Basis.
- Caples & Neuhaus, *Adam Optimizer Causes Privileged Basis in Transformer LM Residual Stream* (LessWrong, Blogpost, nicht begutachtet): https://www.lesswrong.com/posts/yrhu6MeFddnGRSLtQ/adam-optimizer-causes-privileged-basis-in-transformer-lm — mit Adam steigt die Exzess-Kurtosis über 100, mit SGD bleibt sie nahe 0.
- Kovaleva et al. 2021, *BERT Busters: Outlier Dimensions that Disrupt Transformers*: https://arxiv.org/abs/2105.06990 — Ausreißerdimensionen entstehen früh im Training, immer an denselben Positionen.
- Puccetti et al. 2022, *Outlier Dimensions that Disrupt Transformers are Driven by Frequency*: https://arxiv.org/abs/2205.11380 — sie folgen der Tokenfrequenz.

Deutung:
- **Problem:** Die Basis entsteht nicht aus dem Zugleichsein, sondern weil die *Regel der Gewohnheit* (Adam) jede Koordinate einzeln behandelt. Sie wird über das Training eingeschmuggelt. !!ähnlichkeit zu sartres existence before essence

  > **Reaktion:** Siehe die Reaktion zum Sartre-Bezug in § 10b.

- **Chance:** Der saubere Test wird möglich. Bildet ein mit SGD trainiertes Modell (nirgends bevorzugte Achsen) trotzdem abgestufte Gleichartigkeit, ist die starke These 2 widerlegt.
- Puccetti fast humesch gelesen: Was oft vorkommt, wird zur eigenen Dimension.

## 7. Die Farbe

- Die Farbe zeigt die qualitative Verschiedenheit der Dimensionen, die für Gleichartigkeiten vorhanden sein muss (Bergson), und macht Begriffe sichtbar.
- **Nichtnegativität als Bauprinzip:** Additive Farbmischung kennt nur nichtnegative Intensitäten. Ist der Raum auf x ≥ 0 beschränkt, funktionieren Drehungen nicht mehr; übrig bleiben nur Vertauschungen der Achsen. Das ist mathematisch genau eine bevorzugte Basis: unterscheidbare Kanäle, nur durch Umbenennen austauschbar.
- Bekannter Präzedenzfall: Nichtnegativität liefert teilbasierte, lesbare Zerlegungen (Lee & Seung 1999, NMF; Angabe noch prüfen).
- Der alte `W_RGB` war eine *zufällige* Projektion von 192 auf 3 Dimensionen und hat 189 Dimensionen willkürlich verworfen. Er wird nicht weiterverwendet.
!!vielleicht auch mit dem 4dimensionalen system arbeiten cmyk oder auch der 3d h,s,v

  > **Reaktion:** Betrifft nur Stufe (a), die Sichtbarmachung; das Experiment braucht nur Nichtnegativität. CMYK ist subtraktiv (Pigment) und redundant (K aus CMY ableitbar); der Residual Stream *addiert*, das ist näher am Licht, also RGB-artig. **HSV ist vielversprechend:** Value = Größe (Lebhaftigkeit), Saturation = Konzentration auf wenige Dimensionen (genau das Lesbarkeitsmaß aus P2), Hue = welche Dimensionsgruppe dominiert. Haken: 256 Dimensionen passen nicht auf einen Farbkreis; ihre Anordnung ist eine Wahl, ein Vorurteil des Visualisierers, das benannt werden muss.


## 8. Wie Gleichartigkeit gemessen wird

Bedingung: Das Maß darf von keiner Basiswahl abhängen, sonst ist der Vergleich zugunsten von B entschieden.

- **(b) Kernmaß, Ersetzbarkeit:** Zwei Tokens sind in dem Grad gleichartig, in dem die Warte des Modells sie gleich behandelt. Ersetzt man x im Kontext durch y: Wie stark ändert sich die Verteilung der nächsten Tokens (KL-Divergenz)? „Abgestuft“ heißt: Der Kosinus sagt diese Änderung *stetig* voraus, nicht bloß gleich oder verschieden.
- **(c) Zweitmaß, Verschachtelung:** Abgestufte Gleichartigkeit sollte eine Hierarchie ergeben (Hund ⊂ Säugetier ⊂ Tier). Gemessen wird, wie gut sich die Kosinus-Struktur als Baum rekonstruieren lässt.
- **(a) Nur Vergleich:** menschliche Ähnlichkeitsnormen (z. B. SimLex-999). Entscheiden nichts, weil sie voraussetzen, dass die Dimensionen des Modells menschliche sind.

## 9. Register „Begriffsbedarf“

- Wo (b) oder (c) stabile Struktur finden, die kein menschliches Label abdeckt, zählt das weder als Treffer noch als Fehlschlag.
- Es kommt explorativ in ein eigenes Register: ein Phänomen, das einen Begriff verlangt.
- Das ist die Asynchronizität auf der Ebene der Methode: sinnfordernd, kein Fehler.
- (a) wird hier nützlich, weil es genau die Stellen markiert, an denen Modellstruktur und menschliche Normen auseinandergehen.

## 10. Studiendesign (Abschnitt 1, abgestimmt)

**Vorstudie (ohne Training).** Basispräferenz in GPT-2 und Pythia-160m messen, auch über die 11 Pythia-Checkpoints:
- Kurtosis je Dimension;
- Vergleich mit zufälligen Drehungen desselben Raums;
- Korrelation mit der Tokenfrequenz (Puccetti).

Frage: Entsteht die bevorzugte Basis durch Gewohnheit, und wann?

!! bevorzugte basis muss noch in metaphysischen begriffen definiert werden

  > **Reaktion:** Vorschlag: **Eine bevorzugte Basis liegt vor, wenn die Warte ihre Achsen unterscheiden kann.** Im homogenen Fall sind alle Drehungen des Raums für die Warte *ununterscheidbar*, nach Leibniz' Satz von der Identität des Ununterscheidbaren also dieselbe Warte; die Achsen sind bloß Koordinatenwahl des Betrachters. Mit bevorzugter Basis schrumpft die Symmetrie auf Vertauschungen: Die Achsen werden *unterscheidbar*, bleiben aber *gleichrangig* – Bergsons qualitative Mannigfaltigkeit, symmetrietheoretisch gefasst. Was eine Warte nicht unterscheiden kann, ist ihre Symmetrie (vgl. die Noether-Notizen in `vorurteile_protokollsätze/notes.md`). Mit V2s drei Ebenen: A₀ hat keine Basis; in A kann sie als *Füllung* entstehen (durch Gewohnheit erworben); in Bα/Bβ ist sie *Bestandteil* (eingebaut).


**Hauptstudie (Training, lokal, klein).** Vier Arme mit sonst identischer Architektur, identischem Korpus (WikiText-103, eigener 8k-Tokenizer; das vorhandene Wikipedia-Korpus ist mit ~1M Tokens zu klein und bleibt Messmaterial für GPT-2/Pythia) und identischen Seeds:

| Arm | Normalisierung | Optimierer | Farbe |
|---|---|---|---|
| A₀ | RMSNorm, skalarer Gain | SGD | keine |
| A | RMSNorm, skalarer Gain | Adam | keine |
| Bα | RMSNorm, skalarer Gain | Adam | nichtnegatives Embedding (Archetyp) |
| Bβ | RMSNorm, skalarer Gain | Adam | nichtnegative Lesung des Streams in jeder Schicht (Vollzug) |

- **Technische Bedingung:** Standard-LayerNorm hat einen gelernten Gain je Dimension und bevorzugt durch die Mittelwertsubtraktion die Einsen-Richtung. Beides bricht die Symmetrie. Deshalb RMSNorm mit einem einzigen skalaren Gain, sonst ist A₀ nicht der saubere Nullfall.
- **Vergleich bei gleichem Loss**, nicht bei gleicher Schrittzahl, weil SGD Transformer viel langsamer lernt. Checkpoints werden an festen Loss-Stufen gesichert.
- **Mindestens 3 Seeds je Arm.**
- Frage hinter Bα gegen Bβ: Wo wird qualitative Verschiedenheit notwendig, im Archetyp oder im Vollzug?
- Später nichtnegativer Stream selbst (Additionen können nur hinzufügen; „nichts wird abgezogen“) als mögliche Variante (δ). Lernt vermutlich schlecht, daher zurückgestellt.

**Reihenfolge und Regeln:**
- Lokal klein entwickeln und vortesten, später größer in der Cloud.
- Jede Studie bekommt ihre eigene Präregistrierung, committet *vor* dem Lauf.

**Bewusst ausgeklammert:** Folge, sowohl über die Zeit als auch über die Schichten (offene These, eigene Folgestudie).

Ziele in dieser Reihenfolge: (c) konstruktiv, (b) Test, (a) Sichtbarmachung.

## 10a. Vorhersagen und Entscheidungsregeln (Abschnitt 2, abgestimmt)

**Kernmaß (b), operationalisiert:**
- Etwa 2000 häufige Ein-Token-Wörter.
- Je Paar (x, y): Ersetzung in N Kontexten, in denen x vorkommt.
- Ersetzbarkeit S(x, y) = −(mittlere KL-Divergenz der Verteilungen des nächsten Tokens über die folgenden Positionen).
- **Gleichartigkeitsindex G** = Spearman-Korrelation zwischen dem Kosinus der Archetyp-Vektoren und S, nur über Paare im *mittleren 80-%-Bereich* der Kosinuswerte. So ist „abgestuft“ definiert: Der Kosinus muss auch im Mittelfeld unterscheiden, nicht nur Gleiches von sehr Verschiedenem trennen.
- **Embedding und Unembedding entkoppelt** (nicht wie bei GPT-2), sonst misst der Kosinus teilweise die Ausgabeseite selbst und der Test wird zirkulär. **Entschieden.**

**Manipulationsprüfung (bevor eine Vorhersage zählt):**
- Kurtosis A₀ ≈ 0, A deutlich > 0.
- Bα und Bβ baulich nichtnegativ.
- Bildet A₀ trotzdem eine Basis aus, ist der ganze Vergleich nichtig. Das wird berichtet, nicht umgedeutet.

**P1, starke These (Kern):** Bei gleichem Loss gilt G(A₀) < G(A).
- **Widerlegt**, wenn G(A₀) zu G(A) *äquivalent* ist, also innerhalb ±δ laut Äquivalenztest (TOST). **δ = 0,05 in ρ, entschieden.**
- **Unentschieden**, wenn weder Differenz noch Äquivalenz signifikant ist.

**P2, Lesbarkeit:** In Bα und Bβ konzentriert sich ein Archetyp-Vektor auf weniger Dimensionen als in A (Participation Ratio in der eigenen Basis des Modells).
- **Fairness:** A bekommt zusätzlich seine beste Drehung (ICA). B muss *ohne* nachträgliche Drehung gewinnen, A darf *mit*.
- **Preisprüfung:** G(B) darf nicht unter G(A) − δ fallen, sonst ist die Lesbarkeit mit Gleichartigkeit erkauft.

**P3, Verschachtelung (sekundär):** Güte der Baumrekonstruktion (kophenetische Korrelation). Nur gerichtet: B ≥ A ≥ A₀.

**Loss-Stufen:**
- Vorab festgelegt, etwa 5,0 / 4,5 / 4,0 Nats.
- Erreicht A₀ eine Stufe lokal nicht, zählt die höchste gemeinsame Stufe. Das wird als Einschränkung berichtet, nicht als Ergebnis.

**Pilotregel (aus der H2-Erfahrung):** Ein Pilot mit 1 Seed dient nur der Prüfung der Instrumente. Seine Effekte werden nicht gedeutet.

## 10b. Das Design in der Sprache der Theorie

*Status: Die Begriffe der Theorie benennen hier Funktionen des Systems. Sie sind regulative Heuristik im Als-ob und behaupten kein Erleben.*

### Die Leitfrage

Ist die **qualitative Verschiedenheit der Dimensionen** Bedingung der Möglichkeit abgestufter Gleichartigkeit? Oder genügt das **Zugleichsein** vieler Dimensionen allein?

In der Sprache der Systemskizze heißt das: Gehört die Farbe zum **minimalen Gerüst** oder zu den **Füllungen**?

- Ist sie Gerüst, kann ohne sie keine Gleichartigkeit erblickt werden.
- Ist sie Füllung, ist sie Sache der Sichtbarkeit: Sie macht Begriffe lesbar, aber nicht möglich.

Beim Menschen ist diese Frage nicht prüfbar, weil das Gerüst immer schon in Betrieb ist. Im gebauten System lässt sich die Bedingung abschalten. Das Experiment ist **Transzendentalphilosophie als Experiment**: Die Bedingungen werden zu Variablen, um den Preis der Apodiktizität (relativiertes Apriori).

### Vorstudie: Genealogie der Achsen

Gefragt wird, ob im sedimentierten Archetyp bereits ausgezeichnete Achsen liegen und ob sie durch Gewohnheit entstanden sind.

- **Kurtosis je Dimension:** misst, wie stark einzelne Achsen hervortreten. Das ist der Unterschied zwischen *homogener* (quantitativer) und *qualitativer* Mannigfaltigkeit im Sinne Bergsons. !! vielleicht auch für fraktale dimension testen?

  > **Reaktion:** Ja, siehe Reaktion in § 4 (E2).

- **Vergleich mit Drehungen:** Probe, ob die Achsen dem Gegenstand eigen sind oder bloß Koordinatenwahl des Betrachters.
- **Korrelation mit der Tokenfrequenz:** Humes Häufigkeit. Was oft zusammen auftritt, wird zur eigenen Dimension.
- **Checkpoints:** die Ontogenese der Achsen. Wann wird aus der homogenen eine qualitative Mannigfaltigkeit?

### Hauptstudie: vier Warten unter verschiedenen Bedingungen

| Arm | Theoretischer Zustand |
|---|---|
| **A₀** | Eine Warte **ohne jede Vorliebe**, weder in der Anlage noch in der Gewohnheit. Die Architektur zeichnet keine Richtung aus, und die Gewöhnung (SGD) behandelt alle Richtungen gleich. Reine homogene Mannigfaltigkeit: der Nullfall. |
| **A** | Homogene Anlage, aber eine **Gewohnheit mit Vorliebe**: Adam misst jede Achse einzeln. Ausgezeichnete Richtungen können entstehen, aber nur durch Gewöhnung, nicht durch Anlage. Die Farbe wäre hier erworben. |
| **Bα** | Die Farbe ist **im Archetyp angelegt.** Das apriorische Angeordnetsein selbst ist farbig, nichtnegative Intensitäten wie in der additiven Farbmischung. Qualitative Verschiedenheit liegt im Vorrat der Gattung (kollektives Unbewusstes). |
| **Bβ** | Die Farbe liegt **im Vollzug.** Der Strom selbst bleibt homogen, aber jede Schicht erblickt ihn nur durch eine farbige Linse. Qualitative Verschiedenheit liegt im Akt des Erblickens, nicht im Vorrat. |

Bα gegen Bβ fragt: Ist die qualitative Mannigfaltigkeit eine Eigenschaft dessen, *was* vorliegt, oder der Art, *wie* es erblickt wird?

!! bezug zu startes existence preceedes essence

  > **Reaktion:** Strukturell passt die Reihenfolge genau. In *L'existentialisme est un humanisme* ist der Brieföffner (das Papiermesser) der Fall, in dem die Essenz der Existenz vorausgeht, weil der Handwerker ihn vorher entworfen hat: **Bα ist der Brieföffner.** A ist „Existenz vor Essenz“: Es erwirbt seine Achsen durch seine Geschichte. Zwei Grenzen: (1) Bei Sartre geht es um Freiheit, Selbstwahl durch einen Entwurf; A wählt nichts, seine Achsen kommen aus Daten und Adams Regel – das ist eher Faktizität als Freiheit. (2) Auch in A gibt es einen verborgenen Handwerker: Die Wahl von Adam legt die Essenz indirekt hinein. Pointe fürs Essay: **Es gibt keine Warte ohne Handwerker, nur solche mit mehr oder weniger verborgenem.** (Zur Freiheit siehe § 12a, Spiel.)


**Bedingungen der Vergleichbarkeit:**
- **Ein Maß ohne ausgezeichnete Richtung (RMSNorm mit skalarem Gain):** Das Maß, an dem die Warte normiert, darf selbst keine Richtung bevorzugen. Sonst wäre der Nullfall schon gefüllt, bevor der Versuch beginnt.
- **Vergleich bei gleichem Loss:** Der Loss ist die mittlere Ent-Täuschung. Verglichen werden Warten *gleicher Erfahrungssättigung*, nicht gleicher Lebensdauer. Eine langsam lernende Warte (A₀) soll nicht dafür bestraft werden, dass sie langsamer gewöhnt wird.
- **Seeds:** verschiedene Biographien desselben Gattungswesens. Eine Aussage gilt erst, wenn sie über Biographien hinweg trägt.

### Messung: Gleichartigkeit in Anlage und Vollzug

- **Ersetzbarkeit (b), Kernmaß:** Gleichartig ist, was die Warte als gleich behandelt. Das ist Nietzsches *Gleichsetzung des Ungleichen*, operationalisiert: Man tauscht x gegen y und misst, wie stark die Erwartung des Folgenden ent-täuscht wird (KL-Divergenz).
  - Der **Kosinus** misst die Gleichartigkeit in der *Anlage* (im Archetyp).
  - Die **Ersetzbarkeit** misst die Gleichartigkeit im *Vollzug* (in der Erwartung).
  - **G** misst, wie weit Anlage und Vollzug übereinstimmen.
  - **Abgestuft** heißt: Die Übereinstimmung muss auch im Mittelfeld gelten, in den Graden zwischen Gleichheit und Verschiedenheit. Im Eindimensionalen gäbe es nur gleich oder entgegengesetzt. Grade setzen das Zugleichsein vieler Dimensionen voraus.
- **Entkoppelte Gewichte:** Archetyp und Erwartungshorizont werden getrennt, wie bei Pythia auseinandergetreten. Sonst prüfte die Erwartung sich am eigenen Maß, und der Test wäre zirkulär. Der Preis: Die Einheit von Erscheinung und Erwartung am selben Maß, die GPT-2 auszeichnet, wird für dieses Experiment aufgegeben.
- **Verschachtelung (c):** Begriffsbildung durch Komparation, Reflexion und Abstraktion (Logik § 6) ergibt Gattungen und Arten, einen *porphyrischen Baum*. Gemessen wird, ob sich aus der Gleichartigkeit der Anlage eine solche Ordnung von Gattung und Art rekonstruieren lässt.
- **Menschliche Normen (a):** ein Vergleich der Warte der Maschine mit der kollektiven Warte der Menschen (dem Paradigma). Diese Normen sind keine entscheidende Instanz, weil sie voraussetzen würden, dass die Maschine eine perfekte Metapher menschlicher Erfahrung ist.
- **Register „Begriffsbedarf“:** Stabile Gleichartigkeit, für die es keinen Namen gibt, ist weder Treffer noch Fehler. Sie ist **Asynchronizität auf der Ebene der Methode**: ein Knoten, der einen Begriff verlangt, also sinnfordernd.

### Entscheidungsregeln

- **Manipulationsprüfung:** Erst wird geprüft, ob die Bedingungen tatsächlich gesetzt sind, ob also A₀ wirklich ohne Vorliebe blieb. Bildet A₀ trotzdem ausgezeichnete Achsen, ist nicht die These widerlegt, sondern der Versuch misslungen: Die Bedingung wurde nicht abgeschaltet.
- **P1, die Gerüstfrage:**
  - *Bestätigt* (G(A₀) < G(A)): Ohne Vorliebe keine abgestufte Gleichartigkeit. Die qualitative Verschiedenheit ist Bedingung, aber eine durch Gewohnheit *erworbene*: ein sedimentiertes, relativiertes Apriori.
  - *Widerlegt* (A₀ äquivalent zu A): Das Zugleichsein genügt. Die Farbe gehört zu den Füllungen, nicht zum Gerüst.
  - *Unentschieden*: Der Versuch schweigt, und das wird so berichtet.
- **δ = 0,05 als Festsetzung:** Die Schwelle, unterhalb derer ein Unterschied unerheblich heißt, folgt nicht aus der Erfahrung. Sie wird vorab gesetzt, eine Festsetzung im Sinne Neuraths und Poppers. Das System benennt damit seinen eigenen konventionellen Anteil, statt ihn zu verbergen.
- **P2, die Sichtbarkeitsfrage:** Werden Begriffe als Farbakkorde sichtbar, ohne dass der Betrachter nachträglich seine eigene Achsenwahl hineinlegt? (ICA wäre die Basis des Betrachters.) Nach Kant B XVIII ist zu erwarten, dass B lesbar wird, denn wir haben die Basis selbst hineingelegt. Die eigentliche Frage ist der **Preis**: Sichtbarkeit darf nicht mit Gleichartigkeit bezahlt werden.
- **P3:** Die Ordnung von Gattung und Art ist sekundär und nur gerichtet vorhergesagt.
- **Pilot:** Ein einzelner Lauf prüft das Instrument, nicht den Gegenstand. Er wird nicht gedeutet.

## 11. Kant: Transzendentalphilosophie als Experiment

- KrV B XVIII: „daß wir nämlich von den Dingen nur das a priori erkennen, was wir selbst in sie legen“. Davor B XIII: Die Vernunft sieht nur ein, was sie selbst nach ihrem Entwurfe hervorbringt; sie nötigt die Natur, auf ihre Fragen zu antworten (Galilei, Torricelli). **Wortlaut und Stellen vor der Verwendung prüfen.**
- B zu bauen heißt: Wir *legen* die Basis hinein und sehen, was daraus folgt.
- Verschiebung: Kant wollte die Bedingungen möglicher Erfahrung *apodiktisch*. Hier sind sie eher Reichenbachs relativiertes Apriori: **Bedingungen werden zu abschaltbaren Architekturvariablen.**
- Gewinn für die Systemskizze: Das minimale Gerüst ist beim Menschen nicht prüfbar, weil es immer schon in Betrieb ist. Im gebauten System lässt es sich entfernen. *Was für die Erfahrung transzendentale Bedingung ist, wird für die Maschine experimentelle Variable.*
- Grenze: Man lernt die Bedingungen der Gleichartigkeit *für diese Art System*. Der Rückweg zur Erfahrung läuft über die „perfekte Metapher“; diese Brücke ist Arbeit des Essays, nicht des Experiments.
- Formel fürs Essay: nicht „Metaphysik als Wissenschaft“, sondern **Transzendentalphilosophie als Experiment, um den Preis der Apodiktizität.**

## 12a. Spiel (Schiller) – eigene Folgestudie, nicht Teil dieses Specs

Idee (10.10.): „Der Mensch spielt nur, wo er in voller Bedeutung des Worts Mensch ist, und er ist nur da ganz Mensch, wo er spielt“ (Schiller, *Über die ästhetische Erziehung des Menschen*, 15. Brief; **Wortlaut prüfen**). Der Transformer bräuchte einen Spielraum: Er könnte verschiedene Linien durchspielen und wählen, welche passt. Hier läge auch die Stelle, an der das Modell *wählen* könnte – die Freiheit, die Sartre einfordert und die A fehlt.

Präzisierungen, bevor daraus eine Architektur wird:
- **Spiel ist nicht Auswahl.** Bei Schiller vermittelt der Spieltrieb zwischen Stofftrieb und Formtrieb, ohne dass einer zwingt; bei Kant ist es das *freie* Spiel von Einbildungskraft und Verstand, Zweckmäßigkeit ohne Zweck (KdU). „Linien durchspielen und die passende wählen“ hat zwei Momente: das Spiel (zwanglose Variation, im Modell etwa Sampling mehrerer Fortsetzungen) und die Wahl (Auswahl nach einem Kriterium). Das zweite ist schon wieder Zweck.
- **Wonach wird gewählt?** Wählt das Modell die Linie, die seiner eigenen Erwartung am besten entspricht, bestätigt es sich selbst: fensterlose Monade, im schlimmsten Fall Aberglaube (eine Warte, die alles bestätigt). Das Kriterium müsste das **Detektiv-Kriterium** sein: Bindung an andere Folgen, nicht Übereinstimmung mit sich selbst.
- **Bezug zur Dauer:** Durchspielen braucht Zeit, die nicht sofort in Ausgabe übergeht – die „eingeräumte Dauer der Reflexion“ (GG 4.1; Reasoning-Tokens im Essayplan). Spiel wäre dann die Verwendung dieser Dauer.
- **Bestehende Technik prüfen:** Mehrfaches Sampling mit Auswahl (best-of-n, Baumsuche über Fortsetzungen, Reasoning-Modelle) existiert. Die Frage ist nicht, ob man das bauen kann, sondern welches Wahlkriterium es zu Spiel statt zu Optimierung macht.
- Gehört zu Essayplan 5/6 (Architekturskizzen, Detektiv-Tor). Eigene Brainstorming-Runde.

## 12. Noch offen

- Abschnitt 3 des Designs abgestimmt (WikiText-103, eigener 8k-Tokenizer, 5 Seeds mit Seed als Einheit, alter Chroma-Code nach `legacy_experiments/`). Spec: `docs/superpowers/specs/2026-10-10-chroma-gleichartigkeit-design.md`.
- Spiel (Schiller) als eigene Folgestudie, § 12a.
- Welche Warte ist welche (Frage 2, verschachtelte Warten).
- Folge über Schichten (zurückgestellt).
- Nichtnegativer Stream (δ).
- Literaturangaben prüfen: Lee & Seung 1999; Kant-Stellen; Caples & Neuhaus ist ein Blogpost; D&G zur fraktalen Immanenzebene; Hegel, Knotenlinie; Sartre, Brieföffner; Schiller, 15. Brief.
