#!/usr/bin/env python3
"""`masques.v1` pour UN fichier vidéo — le jumeau de `beats_for_file.py`.

⚠⚠ CE SCRIPT EST LE CHAÎNON QUI MANQUAIT À TOUT LE CHANTIER DU MASQUAGE.

Mesuré le 2026-08-31 : `kadans-jobs` sait masquer région par région,
`kadans-api` sait porter la règle du dévoilement (RG-6, RG-6 bis, RG-26)… et
**aucun pixel n'est masqué**, parce que `scripts/build_masques.py` lit la pose
**déjà extraite** du SQLite du corpus — il ne sait donc rien produire pour la
vidéo d'un utilisateur.

Ce script prend **un fichier**, comme `beats_for_file.py`, et suit la même
convention d'appel. Le worker pourra donc l'invoquer par un
`WORKER_MASQUES_CMD` calqué sur `WORKER_BEATS_CMD` — sans embarquer aucun
modèle chez lui, ce que l'ADR-0033 interdit.

    .venv/bin/python scripts/masques_for_file.py {in} --out {outdir}

⚠ CE QU'IL COÛTE — MESURÉ, PAS ESTIMÉ (2026-08-31, MacBook, CPU, une vidéo de
55 s d'un vrai cours, ~7 personnes par image) :

    variante                       temps    ratio    poses
    6 Hz + tracker  (le défaut)    84,6 s   1,54x    2438
    3 Hz + tracker                 62,0 s   1,13x    1226
    6 Hz SANS tracker              42,5 s   0,77x     880
    2 Hz + tracker                 52,9 s   0,96x     822

⚠⚠ DEUX ENSEIGNEMENTS QUI CHANGENT LA FAÇON DE RÉGLER :

1. **La cadence est un MAUVAIS levier.** Diviser les frames par deux ne fait
   gagner que 27 %, pas 50 % : un coût FIXE (chargement des modèles, décodage)
   domine, et la décimation ne le touche pas. Descendre à 2 Hz coûte presque
   autant qu'à 3 Hz — on dégrade la couverture sans rien gagner.
2. ⚠⚠ **Le tracker coûte LA MOITIÉ du calcul, et c'est le seul vrai levier.**
   C'est aussi ce qui rend RG-26 possible (« cette silhouette est la mienne »).
   Le couper divise le temps par deux ET casse l'attribution. **Ce n'est donc
   pas un réglage de performance : c'est un choix de produit.**

⚠ Et la détection se fait UNE FOIS par vidéo, pas à chaque montage : le coût est
par vidéo PUBLIÉE, jamais par lecture.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from pipeline.masques_depuis_pose import masques_pour_un_fichier  # noqa: E402

VERSION = "masques_for_file/1"


def drapeaux(tracker: str, secondes: float, res) -> list[str]:
    """Les `quality_flags` — vocabulaire OUVERT du contrat (ADR-0033).

    ⚠ `sans_suivi` n'est pas un détail de performance : sans tracker, le
    `person_id` change d'une frame à l'autre et l'ATTRIBUTION devient fausse.
    Un masque produit ainsi masque toujours au bon endroit — mais on ne peut
    plus dire « cette silhouette est la mienne » (RG-26). Le drapeau le dit à
    qui lira le masque, au lieu de le laisser deviner.

    ⚠ Le coût est arrondi à la seconde : la précision n'apporte rien, et un
    nombre à six décimales inviterait à comparer deux exécutions que la charge
    machine sépare déjà.
    """
    f: list[str] = []
    if tracker == "none":
        f.append("sans_suivi")
    f.append(f"detection_{round(secondes)}s")
    duree = float(res.n_frames) / float(res.fps) if getattr(res, "fps", 0) else 0.0
    if duree > 0:
        # ⚠ LE RATIO, PAS SEULEMENT LE TEMPS : « 85 s » ne dit rien sans savoir
        # que la vidéo en faisait 55. C'est le ratio qui se compare d'une
        # machine à l'autre.
        f.append(f"detection_x{secondes / duree:.2f}")
    return f


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="masques.v1 pour un fichier vidéo")
    p.add_argument("video", type=Path, help="fichier vidéo (mp4/mov…)")
    p.add_argument("--out", type=Path, required=True, help="dossier de sortie")
    p.add_argument("--frame-decimation", type=int, default=5,
                   help="1 frame sur N (défaut 5 ≈ 6 Hz à 30 fps)")
    p.add_argument("--device", default="cpu")
    # ⚠ LE TRACKER EST ACTIF PAR DÉFAUT, ET CE N'EST PAS UN GOÛT : sans lui, le
    # `person_id` change d'une frame à l'autre, et « cette silhouette est la
    # mienne » (RG-26) devient impossible à tenir. On peut le couper pour
    # mesurer son coût — jamais pour produire un masque qu'on publiera.
    p.add_argument("--tracker", default="botsort", choices=["botsort", "bytetrack", "none"])
    a = p.parse_args(argv)

    if not a.video.is_file():
        print(f"vidéo introuvable : {a.video}", file=sys.stderr)
        return 2

    # ⚠ IMPORT TARDIF, ET C'EST DÉLIBÉRÉ : `pipeline.pose` tire torch, cv2 et
    # les modèles. Les importer en tête rendrait ce fichier — et ses tests —
    # inutilisables sur une machine qui ne les a pas.
    from pipeline.pose import extract_pose

    # ⚠⚠ LE TEMPS SE MESURE ET SE PUBLIE, il ne s'estime pas.
    #
    # Le coût de la détection est LA question qui décide si on branche cette
    # chaîne. Un banc sur une machine ne répond que pour cette machine, et pour
    # cette vidéo-là. En le posant dans le quality_flag, chaque exécution
    # RÉELLE alimente la réponse — et l'arbitrage se fait sur des vidéos
    # d'utilisateurs, pas sur mon MacBook.
    debut = time.perf_counter()
    res = extract_pose(
        a.video,
        frame_decimation=a.frame_decimation,
        device=a.device,
        tracker=a.tracker,
    )
    secondes = time.perf_counter() - debut
    # ⚠⚠ LA ROTATION SE TRANSMET, ELLE NE SE SUPPOSE PAS.
    #
    # `extract_pose` redresse les clips portrait (stockés en paysage avec une
    # métadonnée) de 90° ANTI-HORAIRE, et normalise les boîtes sur le repère
    # REDRESSÉ. Jusqu'au 2026-09-01, ce script laissait `rotation_deg` à son
    # défaut 0 : le masque annonçait « aucune rotation » avec des coordonnées
    # tournées. Mesuré sur une vraie vidéo de cours — les boîtes tombaient sur
    # le parquet et sur les murs, et AUCUN des six visages n'était couvert.
    #
    # ⚠ 90° anti-horaire s'écrit 270 dans la convention HORAIRE du contrat, et
    # `masques_pour_un_fichier` le DÉDUIT du résultat de pose — ce script n'a
    # rien à penser, ce qui est le but : un appelant qui doit y penser oublie.
    payload = masques_pour_un_fichier(
        a.video,
        resultat_pose=res,
        pipeline_version=VERSION,
        quality_flags=drapeaux(a.tracker, secondes, res),
    )

    a.out.mkdir(parents=True, exist_ok=True)
    # ⚠ LE NOM SUIT LA CONVENTION DE `beats_for_file.py` (`.beats.kadans.json`) :
    # le worker cherche un fichier, pas une sortie standard.
    cible = a.out / f"{a.video.stem}.masques.kadans.json"
    cible.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"{a.video.name} → {cible.name} : {len(payload['regions'])} région(s), "
        f"{payload['echantillonnage_hz']:.2f} Hz, "
        f"recommandation « {payload['traitement_recommande']} »",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
