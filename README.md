# ggxtransformer: Sinn, Warte und Gleichartigkeit im Transformer

Philosophisch-experimentelles Projekt. Eine eigene Theorie der Sinnkonstitution (Warte, Tripelstruktur des Sinns aus Art, Folge und Zugleichsein, Synchronizität nach Jung) wird auf Sprachmodelle angewandt, und ihre Begriffe werden zu experimentellen Variablen. Ziel ist eine Wissenschaft des Geistes, die transzendentale Bestimmungen im gebauten System variiert, statt sie nur zu behaupten: *Transzendentalphilosophie als Experiment, um den Preis der Apodiktizität*.

Begriffe und Stand der Theorie: `CLAUDE.md`. Gründungsfragen: `docs/wissenschaft_fragen.md`.

## Aktuelle Hauptlinie: Chroma

**Frage:** Braucht abgestufte Gleichartigkeit qualitativ verschiedene Dimensionen (eine bevorzugte Basis), oder genügt das Zugleichsein vieler Dimensionen?

Dazu werden vier kleine Transformer mit sonst identischem Aufbau trainiert und bei gleichem Loss verglichen:

| Arm | Optimierer | Farbe | bevorzugte Basis |
|---|---|---|---|
| A₀ | SGD | keine | nirgends |
| A | AdamW | keine | kann durch das Training entstehen |
| Bα | AdamW | nichtnegatives Embedding | im Archetyp eingebaut |
| Bβ | AdamW | nichtnegative Lesung des Streams | im Vollzug eingebaut |

Kernmaß ist die **Ersetzbarkeit**: Wie stark ändert sich die Erwartung des Modells, wenn man im Kontext ein Wort x durch y ersetzt? Der Gleichartigkeitsindex G misst, wie gut der Kosinus der Embeddings diese Ersetzbarkeit abgestuft vorhersagt. Eine Vorstudie misst die Basispräferenz in GPT-2 und Pythia-160m über deren Trainings-Checkpoints.

- Spec: `docs/superpowers/specs/2026-10-10-chroma-gleichartigkeit-design.md`
- Ideen und philosophische Fassung: `docs/chroma_ideen.md`
- Implementierungsplan und Entscheidungsprotokoll: `docs/superpowers/plans/`

**Stand:** Code fertig und getestet, noch keine Messung. Vor jedem Lauf wird die zugehörige Präregistrierung (`praereg/PRAEREG_8_achsen.md`, `praereg/PRAEREG_9_chroma.md`) committet.

### Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m pytest -q             # Tests
python -m chroma selbsttest     # Offline-Durchlauf mit winzigen Modellen, prüft nur den Code
```

### Befehle

```bash
python -m chroma korpus --ordner runs/chroma_korpus                      # WikiText-103 laden, BPE (8192) trainieren
python -m chroma train --korpus runs/chroma_korpus --haupt runs/chroma_pilot \
    --arm A --seed 0 --lr 3e-4 --stufen 5.0,4.5,4.0                     # ein Arm, ein Seed
python -m chroma messen --ckpt runs/chroma_pilot/A/seed0/stufe_4.00.pt \
    --korpus runs/chroma_korpus                                           # alle Maße eines Checkpoints
python -m chroma auswerten --haupt runs/chroma_haupt                      # Entscheidungsregeln, bericht.md
python -m chroma register --haupt runs/chroma_haupt --korpus runs/chroma_korpus --arm A --stufe 4.0
python -m chroma vorstudie --modell EleutherAI/pythia-160m --revisionen pythia \
    --seqs runs/lauf_wiki/korpus/seq__EleutherAI__pythia-160m__N256.npz
python -m chroma vorstudie-bericht --v-schwelle <Wert aus PRAEREG_8>
```

Der Mess-Seed (`messen --seed`) muss für alle Läufe einer Studie gleich sein, sonst gilt die Auswertung als inkonsistent. Lernraten, Loss-Stufen und Budget (A₀ bis zum Fünffachen) werden im Pilot kalibriert und in PRAEREG_9 festgeschrieben.

## Repository-Struktur

- **`chroma/`**: das Chroma-Experiment (Modell, Korpus, Training, Messung, Auswertung, Vorstudie, CLI)
- **`tests/`**: Tests zu `chroma/`
- **`docs/`**: Ideen, Gründungsfragen, Spec, Plan, Hausarbeiten-Material, Literatur
- **`praereg/`**: Präregistrierungen
- **`legacy_experiments/`**: vor-chromatische Experimente (Attention Sink, Valenz, Tripel-Ablation, Gewöhnung) und der alte Chroma-Entwurf
- **`runs/`**: Läufe und Korpora
- **`theorie/`**: frühere Dossiers, Brainstorming und keine Position

## Vor-chromatische Experimente: Attention Sink

Das erste Experiment prüfte die These, dass der Attention Sink eine verdrängte Asynchronizität ist: der Ort, an dem ein Kopf seine Aufmerksamkeit ablädt, wenn im Kontext nichts resoniert. Dazu wurden Art und Folge in fünf Bedingungen gezielt zerstört.

| Bedingung | Art | Folge | Konstruktion |
|---|---|---|---|
| A kohärent | ✓ | ✓ | zusammenhängender Text |
| B wortgewürfelt | ✓ | ✗ | Wörter von A permutiert |
| B2 satzgewürfelt | ✓ | teilweise | Sätze von A permutiert, Grammatik bleibt |
| C disparat | ✗ | ✓ | Einzelsätze aus fremden Dokumenten |
| D Rauschen | ✗ | ✗ | Zufallstokens nach Häufigkeit |

**Ergebnis:** H1 und H3 sind widerlegt, und H2 ist als Test unbrauchbar, weil das Maß je nach Lauf beide Vorzeichen liefert. Zerstörte Folge *senkt* die Sink-Masse deutlich, zerstörte Art hebt sie nur minimal. Einzelheiten und Replikation: `CLAUDE.md`, Berichte in `runs/*/analyse/`.

Ausführen (Skripte jetzt in `legacy_experiments/`):

```bash
python legacy_experiments/sinn_experiment.py selbsttest
python legacy_experiments/sinn_experiment.py --ordner runs/pilot --seed 99 korpus --n 30 --pool 50 --min-zeichen 2000
python legacy_experiments/sinn_experiment.py --ordner runs/lauf messen --modell gpt2
python legacy_experiments/sinn_experiment.py --ordner runs/lauf analyse --modell gpt2
python legacy_experiments/sinn_experiment.py --ordner runs/lauf messen --modell EleutherAI/pythia-160m --revisionen pythia
python legacy_experiments/sinn_experiment.py --ordner runs/lauf dynamik --modell EleutherAI/pythia-160m
python legacy_experiments/wiki_texte.py --kontakt deine@mail.de --ziel 500 --ordner meine_texte
```

Messgrößen: Sink-Masse (Anteil der Aufmerksamkeit auf das Start-Token, Query-Positionen ab 8), Ausgabe-Entropie, Loss. Statistik: einseitige Permutationstests, Perzentil-Bootstrap, Benjamini-Hochberg. Neue Wikipedia-Artikel (erste Version ab 2024) unter CC BY-SA 4.0 mit `quellen.csv` und `LIZENZ.md`.

### Präregistrierung des ersten Hauptlaufs (unverändert)

*Vor dem Hauptlauf ausfüllen und committen. Datum: 23. September 2026*

**H1 (Asynchronizität):** Die mittlere Sink-Masse ist in B, B2, C und D höher als in A und in D höher als in B und C.

**H2 (Dissoziation von Art und Folge):** Der schichtgewichtete Schwerpunkt der Differenzkarte ΔC = C − A liegt höher als der von ΔB = B − A (Folge wirkt früh, Art spät), und die beiden Karten korrelieren schwach.

**H3 (Genese):** Die Sink-Masse steigt im Trainingsverlauf deutlich an, und die Differenz D − A wird erst nach diesem Anstieg positiv.

**Gegenhypothese:** Die Sink-Masse ist über die Bedingungen konstant; der Sink wäre dann eine inhaltsunabhängige Gewohnheit.

**Entscheidungsregeln:** Signifikanzniveau 0,05, einseitig; für H2 gilt die Vorhersage als bestätigt, wenn das 95%-Konfidenzintervall der Schwerpunktdifferenz C − B über null liegt.

**Festgelegte Parameter:** N = 256, skip = 8, n = 300, Seed 1234, Modelle gpt2 und EleutherAI/pythia-160m.
