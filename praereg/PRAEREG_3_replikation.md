# Präregistrierung 3: Replikation des Hauptlaufs mit Wikipedia-Korpus

*Vor dem Lauf ausfüllen und committen. Datum: 23. September 2026*

## Anlass

Der Hauptlauf in `lauf/` verwendete ein Web-Crawl-Korpus. Das war weder geplant noch
dokumentiert: `texte.json` hielt die Herkunft nicht fest, weshalb in CLAUDE.md fälschlich ein
Wikipedia-Dump vermutet wurde. Seit dem Commit „Herkunft des Korpus protokollieren" wird die
Quelle mitgeschrieben.

Diese Replikation wiederholt den Hauptlauf mit einem Korpus aus englischen Wikipedia-Artikeln,
deren erste Version ab dem 1. Januar 2024 angelegt wurde (gesammelt mit `wiki_texte.py`, Nachweis
in `quellen.csv`, Lizenz CC BY-SA 4.0). Weder GPT-2 (2019) noch Pythia (Trainingsdaten bis 2023)
können diese Texte gesehen haben.

**Die Ergebnisse in `lauf/` bleiben bestehen und werden nicht ersetzt.** Die Replikation läuft in
`lauf_wiki/` und wird im Essay als eigener Befund berichtet.

## Unveränderte Hypothesen

H1, H2 und H3 gelten wörtlich wie in README.md präregistriert; sie werden hier nicht neu gefasst.
Ihr Ausgang im Crawl-Korpus (H1 widerlegt, H2 nicht bestätigt, H3 widerlegt, siehe PRAEREG_2.md)
ändert daran nichts.

## Neue Replikationsvorhersagen

Neu und deshalb hier präregistriert ist die Frage, ob das im Crawl-Korpus *beobachtete* Muster
sich am unabhängigen Korpus wiederholt. Ohne diese Vorhersagen wäre die Replikation beliebig
deutbar.

**R1 (Richtungsmuster, GPT-2):** B − A und D − A liegen deutlich unter null (95%-KI vollständig
unter null), C − A liegt über null (95%-KI vollständig über null), B2 − A ist von null nicht zu
unterscheiden (95%-KI enthält null).

**R2 (Folge dominiert):** Über alle 144 Köpfe korrelieren die Differenzkarten ΔB und ΔD positiv
und stark, r > 0,8. Im Crawl-Korpus war r = 0,97.

**R3 (Einbruch in der Pythia-Dynamik):** Der Verlauf von D − A über die elf Checkpoints zeigt
erneut eine mittlere Trainingsphase mit signifikant negativem D − A, und zwar an mindestens zwei
aufeinanderfolgenden Checkpoints mit 95%-KI vollständig unter null.

**Entscheidungsregel:** Eine Vorhersage gilt als repliziert, wenn die genannte Bedingung zutrifft,
sonst als nicht repliziert. Es werden keine Vorhersagen nachträglich umformuliert. Signifikanzniveau
0,05, Intervalle wie bisher aus 2000 Bootstraps.

## Vorab benannte Erwartung zum C-Effekt

Bedingung C (Art zerstört) setzt Sätze aus fremden Dokumenten zusammen. Im Crawl-Korpus prallen
dabei ganze Register aufeinander (Rechtstext neben Rezept); ein reines Wikipedia-Korpus ist
stilistisch einheitlich, es wechselt nur das Thema. **Erwartet wird daher ein kleinerer C-Effekt als
die +0,0058 des Crawl-Korpus.**

Das ist eine beschreibende Erwartung, kein formaler Test: Die beiden Korpora unterscheiden sich in
mehr als einer Hinsicht, ein Vergleich zwischen ihnen ist nicht kontrolliert. Sollte C − A hier
kleiner ausfallen, ist das also ein Hinweis auf die Registerabhängigkeit der Art-Manipulation und
kein Beleg. Fällt R1 für C aus, wäre zuerst diese Schwäche der Manipulation zu prüfen, bevor
inhaltlich über den Art-Effekt geschlossen wird.

## Festgelegte Parameter

Identisch zum Hauptlauf: N = 256, skip = 8, n = 300, pool = 200, min-zeichen = 2000, Seed 1234,
Modelle `gpt2` und `EleutherAI/pythia-160m` mit den elf Checkpoints aus `PYTHIA_STEPS`.
Arbeitsordner `lauf_wiki`, Textordner `wiki_texte`.
