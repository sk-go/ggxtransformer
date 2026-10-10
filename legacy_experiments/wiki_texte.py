#!/usr/bin/env python3
"""
wiki_texte.py
Neue englische Wikipedia-Artikel für den Hauptlauf von sinn_experiment.py sammeln.

Gesammelt werden nur Artikel, deren erste Version im gewählten Zeitraum angelegt wurde
(Standard: ab 1. Januar 2024). GPT-2 und Pythia können diese Texte also nicht gesehen haben.
Der Abruf läuft über die offizielle MediaWiki-API, langsam und mit Kontaktangabe, wie es die
Wikimedia-Richtlinien verlangen.

Themenvielfalt: Die Kandidaten stammen aus dem Anlage-Log an zufälligen Zeitpunkten im Zeitraum.
Pro Zeitpunkt und pro anlegender Person werden nur wenige Artikel übernommen, damit keine
Serien gleichartiger Artikel das Korpus dominieren.

Lizenz: Die Texte stehen unter CC BY-SA 4.0. quellen.csv enthält Titel, Permalink und
Anlagedatum jedes Artikels für die Quellenangabe.

Beispiel:
  python wiki_texte.py --kontakt deine@mail.de --ziel 500 --ordner meine_texte
  python sinn_experiment.py korpus --n 300 --pool 200 --min-zeichen 2000 --textordner meine_texte
"""

import argparse
import csv
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
STOP_SECTIONS = re.compile(
    r"^=+\s*(See also|References|External links|Notes|Further reading|Bibliography|Sources|Citations)\s*=+\s*$",
    re.I | re.M)
CSV_FIELDS = ["datei", "titel", "permalink", "angelegt", "revision", "zeichen", "autor"]


def parse_ts(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def parse_day(s):
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


class Wiki:
    """Höflicher API-Client: eine Anfrage nach der anderen, Pause, maxlag, Backoff."""

    def __init__(self, kontakt, pause):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            f"SinnExperiment/1.0 (nichtkommerzielles Forschungsprojekt; Kontakt: {kontakt}) "
            f"python-requests/{requests.__version__}")
        self.pause = pause
        self.last = 0.0
        self.requests = 0

    def get(self, **params):
        params.update(action="query", format="json", formatversion=2, maxlag=5)
        for attempt in range(8):
            wait = self.pause - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            self.requests += 1
            try:
                r = self.session.get(API, params=params, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(_retry_after(r, 2 ** attempt))
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                if data["error"].get("code") == "maxlag":
                    time.sleep(_retry_after(r, 5))
                    continue
                raise RuntimeError(f"API-Fehler: {data['error']}")
            return data
        raise RuntimeError("Wikipedia-API wiederholt nicht erreichbar, später erneut versuchen.")


def _retry_after(r, default):
    try:
        return max(1.0, float(r.headers.get("Retry-After", default)))
    except ValueError:
        return default


# ----------------------------------------------------------------------------
# API-Abfragen
# ----------------------------------------------------------------------------

def creations(wiki, start, limit=100):
    """Artikelanlagen im Hauptnamensraum ab einem Zeitpunkt (chronologisch aufsteigend)."""
    d = wiki.get(list="logevents", letype="create", lenamespace=0, ledir="newer", lelimit=limit,
                 lestart=start.strftime("%Y-%m-%dT%H:%M:%SZ"), leprop="title|timestamp|user")
    return d.get("query", {}).get("logevents", [])


def page_infos(wiki, titles):
    """Länge, Weiterleitung, Begriffsklärung und aktuelle Revision für bis zu 50 Titel."""
    d = wiki.get(prop="info|pageprops", ppprop="disambiguation", titles="|".join(titles))
    return d.get("query", {}).get("pages", [])


def fetch_page(wiki, title):
    """Klartext des Artikels und Zeitpunkt seiner ersten Version."""
    d = wiki.get(prop="extracts|revisions", titles=title, explaintext=1, exsectionformat="wiki",
                 rvdir="newer", rvlimit=1, rvprop="timestamp")
    pages = d.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None, None
    p = pages[0]
    revs = p.get("revisions") or []
    first = parse_ts(revs[0]["timestamp"]) if revs else None
    return p.get("extract", ""), first


# ----------------------------------------------------------------------------
# Aufbereitung
# ----------------------------------------------------------------------------

def clean_extract(text):
    """Anhangsabschnitte abschneiden, Überschriften und Kurzzeilen (Listen, Bildunterschriften) entfernen."""
    m = STOP_SECTIONS.search(text)
    if m:
        text = text[:m.start()]
    lines = [l.strip() for l in text.splitlines()]
    return "\n".join(l for l in lines if not l.startswith("=") and len(l.split()) >= 8)


def safe_name(title):
    return re.sub(r"[^\w\-]+", "_", title, flags=re.UNICODE).strip("_")[:80]


# ----------------------------------------------------------------------------
# Hauptschleife
# ----------------------------------------------------------------------------

def collect(wiki, args):
    out = Path(args.ordner)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "quellen.csv"

    done_titles, per_user = set(), Counter()
    if csv_path.exists():
        with open(csv_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                done_titles.add(row["titel"])
                per_user[row["autor"]] += 1
    saved = len(done_titles)
    if saved >= args.ziel:
        print(f"Bereits {saved} Artikel vorhanden, Ziel erreicht.")
        return
    new_file = not csv_path.exists()
    csv_file = open(csv_path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()

    rng = random.Random(args.seed + saved)  # fortgesetzte Läufe ziehen neue Zeitpunkte
    von, bis = parse_day(args.von), parse_day(args.bis)
    seen = set(done_titles)
    stats = Counter()

    try:
        for sample in range(args.max_stichproben):
            if saved >= args.ziel:
                break
            t = von + (bis - von) * rng.random()
            events = [e for e in creations(wiki, t)
                      if e.get("title") and e["title"] not in seen and parse_ts(e["timestamp"]) < bis]
            rng.shuffle(events)
            users = {e["title"]: e.get("user", "(verborgen)") for e in events}
            seen.update(users)

            # Vorfilter über Seiteninfos: existiert, keine Weiterleitung/Begriffsklärung, lang genug
            candidates = []
            titles = list(users)
            for i in range(0, len(titles), 50):
                for p in page_infos(wiki, titles[i:i + 50]):
                    if p.get("missing") or p.get("redirect"):
                        stats["fehlt/weiterleitung"] += 1
                    elif "disambiguation" in (p.get("pageprops") or {}):
                        stats["begriffsklärung"] += 1
                    elif p.get("length", 0) < args.min_bytes:
                        stats["zu kurz (Quelltext)"] += 1
                    else:
                        candidates.append(p)
            rng.shuffle(candidates)

            taken = 0
            for p in candidates:
                if taken >= args.pro_stichprobe or saved >= args.ziel:
                    break
                title = p["title"]
                user = users.get(title, "(verborgen)")
                if per_user[user] >= args.pro_autor:
                    stats["Autorenlimit"] += 1
                    continue
                extract, first = fetch_page(wiki, title)
                if extract is None or first is None:
                    stats["fehlt/weiterleitung"] += 1
                    continue
                if first < von:
                    stats["ältere Erstversion"] += 1
                    continue
                text = clean_extract(extract)
                if len(text) < args.min_zeichen:
                    stats["zu kurz (Klartext)"] += 1
                    continue
                fname = f"{saved:04d}_{safe_name(title)}.txt"
                (out / fname).write_text(text, encoding="utf-8")
                writer.writerow({
                    "datei": fname, "titel": title,
                    "permalink": f"https://en.wikipedia.org/w/index.php?oldid={p.get('lastrevid', '')}",
                    "angelegt": first.strftime("%Y-%m-%d"), "revision": p.get("lastrevid", ""),
                    "zeichen": len(text), "autor": user})
                csv_file.flush()
                saved += 1
                taken += 1
                per_user[user] += 1
            print(f"Stichprobe {sample + 1:4d} ({t:%Y-%m-%d}): {saved}/{args.ziel} Artikel, "
                  f"{wiki.requests} Anfragen", flush=True)
    finally:
        csv_file.close()

    print("\nVerworfen:", ", ".join(f"{k} {v}" for k, v in stats.most_common()) or "nichts")
    if saved < args.ziel:
        print(f"Nur {saved} von {args.ziel} Artikeln gesammelt. Erneut starten setzt fort; "
              "sonst --max-stichproben erhöhen oder --min-bytes senken.")
    write_license(out)
    print(f"Fertig: {saved} Texte in {out}/, Quellen in {csv_path}")


def write_license(out):
    (out / "LIZENZ.md").write_text(
        "# Quellen und Lizenz\n\n"
        "Die Texte in diesem Ordner stammen aus der englischsprachigen Wikipedia und stehen unter der "
        "Lizenz Creative Commons Attribution-ShareAlike 4.0 "
        "(https://creativecommons.org/licenses/by-sa/4.0/).\n\n"
        "Titel, Permalink auf die verwendete Version und Anlagedatum jedes Artikels stehen in "
        "`quellen.csv`. Die Autorinnen und Autoren sind in der Versionsgeschichte der jeweiligen "
        "Artikel verzeichnet.\n\n"
        "Bearbeitung: Anhangsabschnitte (See also, References usw.), Überschriften und kurze Zeilen "
        "wie Listen und Bildunterschriften wurden entfernt.\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kontakt", required=True,
                    help="E-Mail oder URL für den User-Agent (von Wikimedia verlangt)")
    ap.add_argument("--ziel", type=int, default=500, help="Anzahl zu sammelnder Artikel")
    ap.add_argument("--ordner", default="meine_texte")
    ap.add_argument("--von", default="2024-01-01", help="frühestes Anlagedatum (JJJJ-MM-TT)")
    ap.add_argument("--bis", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--min-zeichen", type=int, default=2000, help="Mindestlänge des bereinigten Textes")
    ap.add_argument("--min-bytes", type=int, default=5000, help="Mindestlänge des Quelltexts (Vorfilter)")
    ap.add_argument("--pro-stichprobe", type=int, default=3, help="max. Artikel pro zufälligem Zeitpunkt")
    ap.add_argument("--pro-autor", type=int, default=3, help="max. Artikel pro anlegender Person")
    ap.add_argument("--max-stichproben", type=int, default=3000)
    ap.add_argument("--pause", type=float, default=1.0, help="Sekunden zwischen Anfragen")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    if parse_day(args.von) >= parse_day(args.bis):
        sys.exit("--von muss vor --bis liegen.")
    collect(Wiki(args.kontakt, args.pause), args)


if __name__ == "__main__":
    main()
