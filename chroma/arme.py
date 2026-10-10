"""Die vier Warten der Hauptstudie (Spec § 4.1)."""

ARME = {
    "A0": {"farbe": "keine", "optimierer": "sgd"},    # keine Vorliebe, weder Anlage noch Gewohnheit
    "A": {"farbe": "keine", "optimierer": "adamw"},   # Vorliebe nur durch Gewohnheit
    "Ba": {"farbe": "alpha", "optimierer": "adamw"},  # Farbe im Archetyp
    "Bb": {"farbe": "beta", "optimierer": "adamw"},   # Farbe im Vollzug
}
