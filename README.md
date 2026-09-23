# Die Tripelstruktur des Sinns im Transformer

Experiment zur These, dass der Attention Sink eines Sprachmodells eine verdrängte Asynchronizität ist: der Ort, an dem ein Attention-Kopf seine Aufmerksamkeit ablädt, wenn im Kontext nichts resoniert. Dazu werden die drei Dimensionen des Sinns (Art, Folge, Zugleichsein) in fünf Bedingungen gezielt zerstört.

| Bedingung | Art | Folge | Konstruktion |
|---|---|---|---|
| A kohärent | ✓ | ✓ | zusammenhängender Text |
| B wortgewürfelt | ✓ | ✗ | Wörter von A permutiert |
| B2 satzgewürfelt | ✓ | teilweise | Sätze von A permutiert, Grammatik bleibt |
| C disparat | ✗ | ✓ | Einzelsätze aus fremden Dokumenten |
| D Rauschen | ✗ | ✗ | Zufallstokens nach Häufigkeit |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python sinn_experiment.py selbsttest   # prüft den Code offline mit einem winzigen Zufallsmodell
```

## Ablauf

**1. Pilot** (separater Ordner und Seed, fließt nicht in die Hypothesentests ein):

```bash
python sinn_experiment.py --ordner pilot --seed 99 korpus --n 30 --pool 50 --min-zeichen 2000
python sinn_experiment.py --ordner pilot --seed 99 messen --modell gpt2
python sinn_experiment.py --ordner pilot --seed 99 analyse --modell gpt2 --permutationen 1000
```

Danach `pilot/korpus/beispiele__gpt2.txt` lesen: Sehen die Bedingungen so aus wie gedacht?

**2. Präregistrierung:** Den Abschnitt unten ausfüllen und committen, *bevor* der Hauptlauf startet.

**3. Hauptlauf** (H1, H2):

```bash
python sinn_experiment.py korpus --n 300 --pool 200 --min-zeichen 2000 --textordner meine_texte
python sinn_experiment.py messen --modell gpt2
python sinn_experiment.py analyse --modell gpt2
```

**4. Genese der Warte** (H3, Pythia-Checkpoints):

```bash
python sinn_experiment.py messen --modell EleutherAI/pythia-160m --revisionen pythia
python sinn_experiment.py analyse --modell EleutherAI/pythia-160m --revision step143000
python sinn_experiment.py dynamik --modell EleutherAI/pythia-160m
```

`--revisionen pythia` misst elf Checkpoints von Schritt 0 bis 143000. Jeder Checkpoint wird einzeln heruntergeladen (rund 300 MB bei Pythia-160M). Bereits gemessene Checkpoints werden übersprungen, ein abgebrochener Lauf kann also einfach neu gestartet werden.

## Welche Texte?

GPT-2 (2019) und Pythia (2020) haben Wikipedia und viele Webtexte im Training gesehen. Kohärente Texte, die sie auswendig kennen, würden Bedingung A verzerren. Für den Hauptlauf deshalb neue Wikipedia-Artikel verwenden, deren erste Version nach Ende 2023 angelegt wurde:

```bash
python wiki_texte.py --kontakt deine@mail.de --ziel 500 --ordner meine_texte
python sinn_experiment.py korpus --n 300 --pool 200 --min-zeichen 2000 --textordner meine_texte
```

`wiki_texte.py` nutzt die offizielle Wikipedia-API mit einer Anfrage pro Sekunde und Kontaktangabe im User-Agent, wie Wikimedia es verlangt. Rechne mit etwa einer halben bis einer Stunde. Ein abgebrochener Lauf setzt beim erneuten Start fort. Für Themenvielfalt werden Artikel an zufälligen Zeitpunkten im Anlage-Log gezogen, mit höchstens drei Artikeln pro Zeitpunkt und pro anlegender Person. `quellen.csv` enthält Titel, Permalink und Anlagedatum; die Texte stehen unter CC BY-SA 4.0 und dürfen mit dieser Quellenangabe (siehe `LIZENZ.md`) im Repo veröffentlicht werden.

Eine Restunschärfe bleibt: Ein neuer Artikel kann Absätze aus älteren Artikeln übernehmen, etwa bei Abspaltungen. Der Loss in Bedingung A dient als Kontrolle.

Für den Pilot reicht ein schneller Web-Datensatz ohne eigenen Abruf, zum Beispiel FineWeb (den genauen Dump-Namen auf der Datensatzseite prüfen):

```bash
python sinn_experiment.py --ordner pilot --seed 99 korpus --n 30 --pool 50 --min-zeichen 2000 \
  --hf-datensatz HuggingFaceFW/fineweb --hf-config CC-MAIN-2024-51
```

Englisch, weil beide Modelle fast nur auf Englisch trainiert sind.

## Ausgaben

- `korpus/texte.json` – die verwendeten Texte (gleich für alle Modelle)
- `korpus/seq__<modell>__N256.npz` – Token-Sequenzen pro Tokenizer
- `korpus/beispiele__<modell>.txt` – lesbare Beispiele jeder Bedingung
- `ergebnisse/<modell>__<revision>.npz` – Sink-Masse [n, Schichten, Köpfe], Entropie und Loss pro Sequenz
- `analyse/<modell>__<revision>/bericht.md` – Tabellen und Tests zu H1 und H2, Heatmaps, `koepfe.csv`
- `analyse/<modell>__dynamik/` – Verlauf über das Training (H3)

## Messgrößen

- **Sink-Masse:** Anteil der Aufmerksamkeit eines Kopfes, der auf das feste Start-Token fällt, gemittelt über die Query-Positionen ab `--skip` (Standard 8).
- **Ausgabe-Entropie:** Unsicherheit der Vorhersageverteilung, Asynchronizität auf der Ausgabeseite.
- **Loss:** Kontrolle, ob die Bedingungen wirken. A sollte am niedrigsten, D am höchsten liegen.

Statistik: einseitige Permutationstests, Perzentil-Bootstrap für Konfidenzintervalle, Benjamini-Hochberg-Korrektur für die kopfweisen Tests.

## Rechenaufwand

GPT-2 small und Pythia-160M laufen auf einer normalen CPU. Pro Bedingung und Checkpoint sind je nach Rechner einige Minuten zu erwarten. Mit Grafikkarte (oder Apple Silicon) geht es deutlich schneller. Bei Speicherproblemen `--batch 1` setzen.

## Präregistrierung

*Vor dem Hauptlauf ausfüllen und committen. Datum: 23. September 2026*

**H1 (Asynchronizität):** Die mittlere Sink-Masse ist in B, B2, C und D höher als in A und in D höher als in B und C.

**H2 (Dissoziation von Art und Folge):** Der schichtgewichtete Schwerpunkt der Differenzkarte ΔC = C − A liegt höher als der von ΔB = B − A (Folge wirkt früh, Art spät), und die beiden Karten korrelieren schwach.

**H3 (Genese):** Die Sink-Masse steigt im Trainingsverlauf deutlich an, und die Differenz D − A wird erst nach diesem Anstieg positiv.

**Gegenhypothese:** Die Sink-Masse ist über die Bedingungen konstant; der Sink wäre dann eine inhaltsunabhängige Gewohnheit.

**Entscheidungsregeln:** Signifikanzniveau 0,05, einseitig; für H2 gilt die Vorhersage als bestätigt, wenn das 95%-Konfidenzintervall der Schwerpunktdifferenz C − B über null liegt.

**Festgelegte Parameter:** N = 256, skip = 8, n = 300, Seed 1234, Modelle gpt2 und EleutherAI/pythia-160m.
