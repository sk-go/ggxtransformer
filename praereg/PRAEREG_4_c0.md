# Präregistrierung 4: Kontrollbedingung C0

*Vor dem Lauf ausfüllen und committen. Datum: 23. September 2026*

## Anlass

Bedingung C soll die **Art** zerstören (Gleichartigkeit), indem Einzelsätze aus vielen fremden
Dokumenten aneinandergereiht werden. Sie verändert gegenüber A aber zweierlei zugleich:

1. das Thema wechselt an jeder Satzgrenze,
2. die Sätze waren nie benachbart — keine Konnektoren, keine Anaphern über die Grenzen hinweg.

Der gemessene C-Effekt ist klein (|ΔC| erreicht 0,022 gegenüber 0,223 bei |ΔB|, Faktor zehn) und
liegt in frühen Schichten. Ob dieser kleine Effekt der Art-Manipulation zuzuschreiben ist oder
bloß der Bauweise, ist mit C allein nicht zu entscheiden. Die Register-Erklärung ist bereits
widerlegt (`PRAEREG_3_replikation.md`: C − A war im einheitlichen Wikipedia-Register nicht
kleiner, sondern minimal größer).

## Die Kontrollbedingung

**C0 einthemig:** Einzelsätze aus **einem** fremden Dokument, zufällig ausgewählt und zufällig
angeordnet, sonst exakt wie C gebaut. Damit ist Punkt 2 (Satzgrenzen, nie benachbart) identisch zu
C, während Punkt 1 (Themenwechsel) entfällt.

- **C − C0** isoliert den Themenwechsel.
- **C0 − A** isoliert das Satzgrenzen-Artefakt.

## Vorhersagen

**V1 (Themenwechsel wirkt):** Die mittlere Sink-Masse in C liegt über der in C0; das 95%-KI der
Differenz C − C0 liegt vollständig über null.

**V2 (Satzgrenzen wirken):** C0 unterscheidet sich von A; das 95%-KI der Differenz C0 − A schließt
null nicht ein.

**Entscheidungsregel:** Signifikanzniveau 0,05, Intervalle aus 2000 Bootstraps, Permutationstests
mit 5000 Permutationen, wie bisher. Eine Vorhersage gilt als zutreffend, wenn die genannte
Intervallbedingung erfüllt ist.

## Vorab festgelegte Deutung

Damit das Ergebnis nicht hinterher passend erzählt werden kann, steht die Zuordnung hier vorher
fest:

| V1 | V2 | Deutung |
|---|---|---|
| ja | nein | C misst den Themenwechsel. Die Art-Manipulation funktioniert; der Effekt ist eben klein. |
| nein | ja | Der C-Effekt ist ein Satzgrenzen-Artefakt. **C misst nicht die Art**, und alle bisherigen Aussagen über den Art-Effekt (H1, H2, Ersatzmaß) verlieren ihre Grundlage. |
| ja | ja | Beides wirkt. Die Anteile werden über die Größe der beiden Differenzen verglichen und berichtet, ohne eine davon zur Hauptursache zu erklären. |
| nein | nein | Weder Thema noch Satzgrenze erklären den C-Effekt. Dann ist offen, was C überhaupt variiert, und das ist im Essay als offen zu benennen. |

Der zweite Fall wäre das für die Theorie unangenehmste Ergebnis und wird genauso berichtet wie
jedes andere.

## Festgelegte Parameter

Korpus identisch zu `lauf_wiki` (dieselbe `texte.json`, also dieselben 600 Wikipedia-Artikel),
N = 256, skip = 8, n = 300, pool = 200, Seed 1234, Modell `gpt2`. Arbeitsordner `lauf_c0`.

Der Sequenzbau für A, B, B2, C und D ist von der Ergänzung unberührt (C0 wird nach C erzeugt, D
zieht aus einem eigenen Zufallsstrom). Die Sequenzen dieser fünf Bedingungen müssen daher mit
denen aus `lauf_wiki` identisch sein; das wird vor der Auswertung geprüft und berichtet. Weicht
etwas ab, ist der Lauf ungültig.

Pythia folgt nur, wenn der GPT-2-Lauf auswertbar ist.
