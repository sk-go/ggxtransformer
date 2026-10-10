# Präregistrierung 2: Abweichung von H3

*Vor der vertiefenden Analyse ausfüllen und committen. Datum: 23. September 2026*

## Ausgangsbefund (Beobachtung, nicht präregistriert)

H3 (README.md) sagte voraus: Die Sink-Masse steigt im Trainingsverlauf deutlich an, und die Differenz D − A wird erst nach diesem Anstieg positiv.

Beobachtet in `lauf/analyse/EleutherAI__pythia-160m__dynamik/dynamik.csv` (11 Checkpoints, step 0–143000):

- Die Sink-Masse steigt wie vorhergesagt: 0,0136 (step 0) auf 0,207 (step 143000).
- D − A ist bei step 1000–2000 kurz signifikant positiv (95%-KI schließt 0 aus).
- Bei step 4000–100000 ist D − A durchgehend signifikant **negativ** (−0,020 bis −0,030, KIs schließen 0 aus).
- Erst beim letzten Checkpoint (step 143000) liegt D − A wieder bei ~0 (95%-KI −0,003 bis +0,004, schließt 0 ein).

Das widerspricht der vorhergesagten monotonen Entwicklung zu positiv: Statt eines Übergangs nach dem Anstieg zeigt sich ein Einbruch in der mittleren Trainingsphase. **H3 gilt in der vorhergesagten Form als widerlegt.** Diese Präregistrierung ändert nichts an H3 in README.md — sie bleibt unverändert und widerlegt stehen. Die folgende Hypothese ist neu und noch ungeprüft.

## Neue Hypothese H3b (Ursache des Einbruchs)

Vorüberlegung aus dem explorativen Teil des GPT-2-Hauptlaufs: ΔB (Folge zerstört) und ΔD (Folge und Art zerstört) korrelieren mit r = 0,97 — der Folge-Effekt dominiert D. Wenn sich der Folge-Effekt (B − A) im Training früher ausbildet als der Art-Effekt (C − A), könnte der Einbruch von D − A in der mittleren Trainingsphase daher rühren, dass der (negative) Folge-Effekt den noch kaum ausgebildeten Art-Effekt zeitweise dominiert, und ein Ausgleich erst spät im Training eintritt.

**H3b:** |B − A| (Betrag des Folge-Effekts) erreicht Signifikanz (95%-KI ohne 0) an früheren Checkpoints als |C − A| (Betrag des Art-Effekts). Die Checkpoints mit dem stärksten D − A-Einbruch (step 4000–100000) fallen in eine Phase, in der B − A bereits deutlich ausgeprägt, C − A aber noch klein oder nicht signifikant ist.

**Gegenhypothese:** Folge-Effekt und Art-Effekt entstehen zeitlich parallel bzw. in vergleichbarem Tempo; der D − A-Einbruch hat eine andere Ursache als eine zeitliche Lücke zwischen beiden Effekten (z. B. allgemeine Instabilität des Sinks in mittleren Checkpoints).

**Messgröße:** B − A und C − A mit Bootstrap-95%-KI (2000 Bootstraps, wie bereits für D − A verwendet) für jeden der 11 Checkpoints. Berechnet aus bereits vorliegenden Rohdaten (`sink_B`, `sink_C`, `sink_A` in dynamik.csv bzw. den zugrunde liegenden `.npz`-Dateien in `lauf/ergebnisse/`) — es ist keine neue Messung nötig, nur eine erweiterte Analyse der `dynamik`-Funktion.

**Entscheidungsregel:** H3b gilt als gestützt, wenn B − A an mindestens zwei Checkpoints vor step 4000 bereits ein signifikantes KI (ungleich 0) erreicht, während C − A das an denselben Checkpoints nicht tut, und wenn die Checkpoints mit dem stärksten D − A-Einbruch (step 4000–100000) in diese Lücke fallen. Sonst gilt H3b als nicht gestützt.

**Festgelegte Parameter:** dieselben 11 Checkpoints, n = 300, N = 256, skip = 8, Seed 1234, Modell EleutherAI/pythia-160m, 2000 Bootstraps.

## Verhältnis zu den in CLAUDE.md gelisteten Erweiterungen

Diese Präregistrierung deckt sich mit dem dort offenen Punkt „Pythia-Dynamik: Entstehen der Folge-Effekt und der Art-Effekt zu verschiedenen Zeitpunkten?" — sie macht ihn konkret und legt Entscheidungsregeln fest, bevor die Analyse läuft.
