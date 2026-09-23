# Auswertung: EleutherAI/pythia-160m @ step143000

n = 300 Sequenzen pro Bedingung, N = 256, skip = 8, 12 Schichten × 12 Köpfe, 5000 Permutationen, 2000 Bootstraps.

Korpus: nicht protokolliert (Korpus vor Einführung der Herkunftsangabe gebaut)

## Übersicht

| Bedingung | Sink-Masse (95%-KI) | Ausgabe-Entropie | Loss |
|---|---|---|---|
| A kohärent | 0.2069 (0.2045–0.2095) | 3.579 | 3.434 |
| B wortgewürfelt | 0.1853 (0.1828–0.1877) | 6.427 | 6.759 |
| B2 satzgewürfelt | 0.2057 (0.2034–0.2082) | 3.663 | 3.587 |
| C disparat | 0.2103 (0.2085–0.2121) | 4.150 | 4.269 |
| D Rauschen | 0.2073 (0.2053–0.2095) | 7.190 | 9.081 |

Kontrolle: Der Loss sollte in A am niedrigsten und in D am höchsten sein. Ein auffällig niedriger Loss in A deutet auf auswendig gelernte Texte hin.

## H1: Asynchronizität erhöht die Sink-Masse

| Vergleich | Differenz | 95%-KI | p (einseitig) |
|---|---|---|---|
| B > A | -0.0216 | -0.0255 bis -0.0181 | 1.0000 |
| B2 > A | -0.0012 | -0.0049 bis +0.0023 | 0.7578 |
| C > A | +0.0034 | +0.0003 bis +0.0066 | 0.0194 |
| D > A | +0.0004 | -0.0031 bis +0.0039 | 0.4079 |
| D > B | +0.0220 | +0.0187 bis +0.0255 | 0.0002 |
| D > C | -0.0030 | -0.0057 bis -0.0002 | 0.9808 |

## Kopfweise Differenzen zu A (Benjamini-Hochberg, α = 0,05)

- B wortgewürfelt: 44 von 144 Köpfen signifikant erhöht
- B2 satzgewürfelt: 0 von 144 Köpfen signifikant erhöht
- C disparat: 38 von 144 Köpfen signifikant erhöht
- D Rauschen: 59 von 144 Köpfen signifikant erhöht

## H2: Folge und Art treiben verschiedene Köpfe in den Sink

- Schichtschwerpunkt ΔB (Folge zerstört): 7.56
- Schichtschwerpunkt ΔC (Art zerstört): 7.21
- Differenz C − B: -0.34 (95%-KI -0.54 bis -0.15); Vorhersage: positiv
- Korrelation der Karten ΔB und ΔC: r = -0.14 (95%-KI -0.21 bis -0.07); Vorhersage: schwach

Schichten sind ab 0 gezählt. Das Bootstrap resampelt A und B unabhängig, obwohl B aus A abgeleitet ist; die Intervalle sind daher eher konservativ.

### Ersatzmaß |Δ| (explorativ, nicht präregistriert)

- Schwerpunkt |ΔB| (Folge zerstört): 7.55
- Schwerpunkt |ΔC| (Art zerstört): 7.10
- Differenz C − B: -0.45 (95%-KI -0.57 bis -0.34)

Das präregistrierte Maß oben wertet nur positive Differenzen und liefert deshalb je nach Lauf ein anderes Vorzeichen, sobald eine Bedingung überwiegend negativ wirkt. Dieses Maß nutzt die volle Wirkungstiefe unabhängig vom Vorzeichen. Es wurde nach Kenntnis der Daten definiert und ist damit explorativ; ein konfirmatorischer Test verlangt einen neuen Lauf.

## Grafiken

![Sink-Karten](sink_karten.png)

![Differenzkarten](differenzkarten.png)

Punkte markieren Köpfe, die nach BH-Korrektur signifikant erhöht sind.