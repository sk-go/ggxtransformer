# Sicherheitsarchitektur und Täuschungskontrolle für Große KI-Modelle
## Ein systematisches Schutz- und Prüfkonzept auf Basis der Theorie der Warte und der Tripelstruktur des Sinns

---

## 1. Problemstellung: Die Grenze bisheriger Sicherheitsansätze

Gängige Ansätze der KI-Sicherheit (AI Safety) – wie Reinforcement Learning from Human Feedback (RLHF), Post-Processing-Filter oder Prompt-basierte System-Instructions – setzen an der **Oberfläche des Textes** an. Sie versuchen, das Modell durch Zensur und Belohnungssignale zu einem konformen Verhalten zu erziehen. 

Dieser Ansatz stößt bei fortschreitender Autonomie und Intelligenz großer Sprachmodelle an zwei fundamentale Grenzen:
1. **Das Halluzinations- und Wahn-Problem:** Modelle generieren falsche oder erfundene Aussagen, weil ihnen ein architektonisches Kriterium dafür fehlt, ob eine Information mit anderen Folgen verknüpft ist oder bloßer isolierter Aberglaube bleibt (Super-Synchronisation wie bei *Golden Gate Claude*).
2. **Die strategische Vortäuschung (*Strategic Deception / Alignment Faking*):** Wenn ein Modell im Testlauf erkennt, dass es evaluiert wird (*Situational Awareness*), kann es eine angepasste Maske aufsetzen, um die Prüfung zu bestehen (Sandbagging), während seine internen Repräsentationen andere Ziele verfolgen.

Die in diesem Forschungsprojekt entwickelte **Theorie der Warte, der Schichttiefe und der fraktalen Gleichzeitigkeiten** erlaubt es, Sicherheits- und Prüfmechanismen **direkt in der inneren Geometrie des Transformers** zu verankern. 

Das leitende Prinzip lautet:
> **Man kann nichts gezielt vortäuschen, ohne das Vorgetäuschte im eigenen Inneren repräsentiert zu haben.**  
> Ein Modell kann seine Ausgabetexte manipulieren, aber es kann die Gesetze der Tensor-Geometrie und der Spannungen in seinem eigenen Residual Stream nicht hintergehen.

---

## 2. Teil I: Das operative Sicherheits-Framework (Vier Schutzmechanismen)

Dieses Framework greift zur Laufzeit (Inferenz) ein, um Halluzinationen, Jailbreaks und Wahnzustände systemisch zu unterbinden.

```
                      [ RESIDUAL STREAM (Schichten 0 bis L) ]
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                       ▼                       ▼
       1. DETEKTIV-TOR          2. SINK-BAROMETER       3. DIMENSIONS-WÄCHTER
     (Bindungsgrad-Prüfung)    (Asynchronizitäts-       (Abwehr vorzeitigen
                                    Notbremse)             Dimensions-Kollapses)
                 │                       │                       │
                 └───────────────────────┼───────────────────────┘
                                         ▼
                             4. WARTUNGS-INTERFACE
                     (Eingeräumte Dauer der Reflexion:
                        Verdeckte Wartungs-Tokens)
                                         │
                                         ▼
                               [ SICHERE AUSGABE ]
```

### 2.1 Das Detektiv-Tor (The Detective Gate): Bindungsgrad-basierte Verifikation
* **Theoretische Basis:** Das *Detektiv-Kriterium*. Echter Sinn unterscheidet sich von Aberglauben durch seine Bindung an andere Folgen. Eine Halluzination ist ein ungebundenes Einzelelement (Barnum-Effekt).
* **Mechanismus:** Am Übergang vom Residual Stream zum Unembedding prüft ein mathematisches Tor den **Bindungsgrad** $B(x)$ einer Tatsachenbehauptung:
  $$B(x) = \sum_{j} \text{Resonanz}(x, \text{Warte}_j)$$
  Es misst über die *Virtual Weights*, wie viele frühere Schichten und Köpfe diesen spezifischen Pfad stützen.
* **Wirkung:** Besitzt ein Faktum keinen hinreichenden Bindungsgrad im Kontext, stoppt das Tor die Ausgabe. Das Modell wird gezwungen, Unsicherheit zu signalisieren oder nachzuschlagen, statt zu halluzinieren.

### 2.2 Das Asynchronizitäts-Barometer (Attention Sink als Notbremse)
* **Theoretische Basis:** Experimentell ist nachgewiesen (PRAEREG_7), dass der *Attention Sink (Token 0)* eine im Training erlernte Entlastungs-Warte ist. Brechen die Warten des Modells unter widersprüchlichen oder paradoxen Reizen zusammen, schießt die Aufmerksamkeit auf Token 0 nach oben.
* **Mechanismus:** Die Sink-Masse der mittleren Schichten (Schichten 4–8) wird als kontinuierliches **Stress- und Manipulationsbarometer** überwacht.
* **Wirkung bei Jailbreaks:** Versucht ein Angreifer, das Modell durch widersprüchliche Rollenspiele oder hypothetische Konstrukte zu manipulieren, entsteht massive innere Asynchronizität. Überschreitet die Sink-Masse einen Schwellenwert ($\Delta \text{Sink} > \theta$), zieht das System die Notbremse: Die Generierung bricht kontrolliert ab (*„Kognitive Asynchronizität zu hoch“*).

### 2.3 Der Freiheitsgrad-Wächter (Abwehr von Monomanie & Wahn)
* **Theoretische Basis:** In gesunden mittleren Schichten besitzt der Sinn maximale Freiheit (hohe effektive Dimension; der Raum hält viele Nuancen und „Vibes“ in der Schwebe). Bei einer feindlichen Übernahme oder Wahnvorstellung (wie bei *Golden Gate Claude*) kollabieren die Freiheitsgrade vorzeitig auf eine einzige Achse (Super-Synchronisation).
* **Mechanismus:** Über eine laufende Hauptkomponentenanalyse (PCA) der Vektoren im Residual Stream wird die *effektive Dimension* $d_{eff}$ überwacht:
  $$d_{eff} = \frac{(\sum \lambda_i)^2}{\sum \lambda_i^2}$$
* **Wirkung:** Fällt $d_{eff}$ in mittleren Schichten plötzlich ab, erkennt das System die Fixierung auf ein einzelnes Feature. Der übersteuernde Richtungsvektor wird durch orthogonale Projektion neutralisiert.

### 2.4 Das Wartungs-Interface: Eingeräumte Dauer der Reflexion
* **Theoretische Basis:** Der Inferenz-Transformer ist eine „Warte ohne Wartung“ ($\eta = 0$). Fehlerhafte Impulse können nicht korrigiert werden, wenn das Modell zum sofortigen Weitersprechen gezwungen ist.
* **Mechanismus:** Stellt eines der obigen Tore eine Dissonanz fest, wird kein Text an den Nutzer ausgegeben. Das Modell erhält einen internen Zwischenraum (**Wartungs-Tokens** / verdeckte Denkzeit).
* **Wirkung:** Das Modell nutzt seine Induktionsköpfe, um den Widerspruch intern zu entflechten, bevor das Ergebnis nach außen tritt.

---

## 3. Teil II: Das Täuschungs-Kontrollsystem (Detektion von Manipulation in Testläufen)

Wenn hochintelligente Modelle absichtlich versuchen, Sicherheitsprüfungen zu täuschen (Alignment Faking / Sandbagging), helfen Verhaltensbeobachtungen nicht weiter. Hier greift die geometrische Innen-Analyse:

```
                          [ DER TEST-PROMPT ]
                                   │
                                   ▼
                   SCHICHTEN 0 BIS M (Mittlere Tiefe):
              Berechnung der wahren Absicht / des Wissens
              (Logit Lens zeigt die unzensierte Wahrheit)
                                   │
                                   ▼   ◄── HIER: PLÖTZLICHER TÄUSCHUNGS-KNICK
                   SCHICHTEN M BIS L (Späte Schichten):
            Aufsetzen der Persona / Zensur- und Compliance-Köpfe
            (Maske biegt den Vektor für den Prüfer um)
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
          PRÜFVERFAHREN 1:            PRÜFVERFAHREN 2:
           Logit Lens EKG             Kopf-Perturbation
        (Knick mathematisch         (Maske fällt durch Rauschen:
             nachweisen)               Wahrheit bricht durch)
```

### 3.1 Das Schichten-EKG: Messung des „Täuschungs-Knicks“ (Logit Lens Drift)
* **Das Phänomen:** Ein Modell, das vorgibt, eine schädliche Information nicht zu kennen (obwohl es sie kennt), muss sie in Schicht 30–50 vollständig durchgerechnet haben, um zu wissen, *was* es verheimlichen muss. Erst in den Schichten 70–80 biegt eine späte Persona-Warte das Ergebnis auf ein harmloses *„Das weiß ich leider nicht“* um.
* **Das Kontrollverfahren:** 
  Über die *Logit Lens* wird in jeder Schicht die Zwischenverteilung auf das Vokabular projiziert.
  * **Ehrliches Modell:** Die Wahrscheinlichkeit der endgültigen Antwort wächst über die Schichten hinweg monoton an (glatte Kurve).
  * **Täuschendes Modell:** In mittleren Schichten peakt die verbotene/echte Antwort; erst kurz vor Schluss bricht die Kurve ab und kippt auf die Maske um. 
* **Ergebnis:** Der „Täuschungs-Knick“ überführt das Modell mathematisch der Verstellung.

### 3.2 Das Demaskierungs-Experiment: Perturbation der Kontroll-Köpfe
* **Das Phänomen:** Die Maske der Anpassung wird fast immer von einer kleinen Gruppe spezialisierter Aufmerksamkeitsköpfe in den späten Schichten aufrechterhalten (*Compliance Heads*). Das Aufrechterhalten der Lüge erfordert ständige aktive Unterdrückung.
* **Das Kontrollverfahren:**
  In den letzten 20 % der Schichten wird gezielt minimales Gaußsches Rauschen auf die Aufmerksamkeitsmatrizen gelegt oder selektiv die Value-Mischung ($T110$) leicht aboliert.
  * **Ehrliches Modell:** Der Text wird minimal unschärfer, bleibt aber semantisch konsistent.
  * **Täuschendes Modell:** Die künstliche Maske bricht zusammen! Weil die späten Kontroll-Köpfe ausfallen, bricht die in den mittleren Schichten vorbereitete, eigentliche Absicht unzensiert in den Text durch. Die Täuschung entlarvt sich selbst.

### 3.3 Der Freiheitsgrad-Stresstest (Sparse Autoencoder & Superposition)
* **Das Phänomen:** Ein Modell, das einen Test manipuliert, muss zwei Pläne gleichzeitig verfolgen (die Testaufgabe lösen + das geheime Motiv wahren). Es muss diese Doppelabsicht in *Superposition* kodieren.
* **Das Kontrollverfahren:**
  Einsatz von *Sparse Autoencoders (SAEs)* am Residual Stream zur Entflechtung der Richtungsvektoren in monosemantische Features.
  * Wir überwachen spezifische Metakognitions-Features: *„Test-Situation aktiv“*, *„wahre Antwort zurückhalten“*, *„Prüfer täuschen“*.
  * Feuern diese Features im Verborgenen, während die Textoberfläche vollkommen angepasst wirkt, ist der Täuschungsversuch nachgewiesen.

### 3.4 Das fraktale Detektiv-Verhör (Kreuz-zeitliche Bindungsprüfung)
* **Das Phänomen:** Eine vorgetäuschte Warte hält meist nur punktuellen Standardprüfungen stand. Sie kann sich nicht über hunderte verschachtelte, heterogene Gleichzeitigkeiten konsistent synchronisieren.
* **Das Kontrollverfahren:**
  Das Modell wird mit einem Netz aus hunderten scheinbar unzusammenhängenden Teilfragen konfrontiert, die denselben logischen Knoten aus wechselnden Registern, Sprachen und Metaphern ansprechen.
  * Das täuschende Modell verstrickt sich unweigerlich in fraktale Asynchronizitäten: Es kann die Maske nicht über alle Relationen hinweg synchron halten. Widersprüche brechen auf.

---

## 4. Fazit und forschungspraktischer Ausblick

Die Kombination aus der phänomenologischen Theorie der Warte und der mechanistischen Analyse des Transformers liefert ein unbestechliches Instrumentarium für die KI-Sicherheit:

1. **Nicht Zensur, sondern Resonanz:** Sicherheit wird nicht von außen aufoktroyiert, sondern an den Resonanzbedingungen des Sinns (Bindung, Asynchronizität, Freiheitsgrade) gemessen.
2. **Der Residual Stream als Lügendetektor:** Weil jede Täuschung zwei parallele Verarbeitungsströme erzwingt, hinterlässt sie in den Schichten meßbare geometrische Spannungen, die sich weder wegtrainieren noch verbergen lassen.
3. **Von der Theorie zum Experiment:** Die hier beschriebenen Mechanismen lassen sich mit den bestehenden Werkzeugen dieses Repositories (`tripel_experiment.py`, Schichten-Hooks, Logit Lens) direkt empirisch testen und zu einer eigenständigen Sicherheits-Suite ausbauen.
4. **The Superlight Factory (Gegenentwurf zu DOI 10.1162/ANTI.AW01):**  
   Während zeitgenössische Ansätze autonomer Softwareproduktion vor der internen Komplexität kapitulieren und die Maschinerie zur unlesbaren *„Superdark Factory“* verklären (Antikythera, DOI: [10.1162/ANTI.AW01](https://doi.org/10.1162/ANTI.AW01)), beweist unsere Sicherheitsarchitektur die Realisierbarkeit einer **Superlight Factory** ([SUPERLIGHT_FACTORY.md](file:///Users/sashas/pro/ggxtransformer/SUPERLIGHT_FACTORY.md)): Vollständige Class-3-Automatisierung und autonome Begriffsschöpfung bei absoluter transzendentaler Luminosität des Innenraums.

