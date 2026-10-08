"""Le pont entre une extraction de pose et `masques.v1` — SANS la base R&D.

⚠⚠ POURQUOI CE MODULE EXISTE, ET CE QU'IL DÉBLOQUE
====================================================

`scripts/build_masques.py` lit la pose **déjà extraite** dans le SQLite du
corpus (`pose_keypoints`). Il ne sait donc produire un masque que pour une vidéo
du corpus R&D — **jamais pour la vidéo d'un utilisateur**.

Conséquence mesurée le 2026-08-31 : `kadans-jobs` sait masquer région par
région, `kadans-api` sait porter la règle du dévoilement… et **aucun pixel n'est
masqué**, parce que rien ne sait fabriquer le masque d'une vidéo réelle.

Or tout était là :

- `pipeline.pose.extract_pose(video_path, …)` prend **un chemin de fichier** ;
- `pipeline.masques.construire_masques(…)` prend des **`Echantillon`**, c'est-à-dire
  de la donnée pure — pas des lignes de base.

Il ne manquait que **l'adaptateur entre les deux**. C'est ce fichier.

⚠ Il ne contient AUCUNE logique de masquage : la géométrie, les marges, la
dilatation et le choix du traitement restent dans `pipeline/masques.py`. On
traduit une forme en une autre, rien de plus — c'est ce qui permet de le tester
sans installer un modèle de détection.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

from pipeline.masques import Echantillon, construire_masques


def md5_du_fichier(chemin: Path, bloc: int = 1 << 20) -> str:
    """L'identité pivot de `masques.v1`.

    ⚠ Lu par blocs : une vidéo d'entraînement pèse des centaines de Mo, et la
    charger en mémoire pour en prendre l'empreinte tuerait le worker sur un
    téléphone-source de 4 Go.
    """
    h = hashlib.md5()  # noqa: S324 — identité de contenu, pas une empreinte de sécurité
    with chemin.open("rb") as f:
        for morceau in iter(lambda: f.read(bloc), b""):
            h.update(morceau)
    return h.hexdigest()


def echantillons_depuis_frames(
    frame_poses: Iterable[object],
    *,
    fps: float,
    frame_decimation: int,
) -> list[Echantillon]:
    """`FramePose[]` → `Echantillon[]`.

    ⚠⚠ LE `person_id` VIENT DU `track_id` QUAND IL EXISTE, ET C'EST TOUT L'ENJEU.

    `masques.v1` promet un `person_id` **stable dans la vidéo** : c'est ce qui
    permet de dire « cette silhouette-là est la mienne » (RG-26) et de ne
    dévoiler QUE les régions créditées (RG-6). Le `person_id` brut de
    l'extraction change d'une frame à l'autre ; le `track_id` de BoT-SORT, non.

    ⚠ Sans tracker (`tracker="none"`), `track_id` est `None` et on retombe sur
    le `person_id` brut — **le masquage marche encore** (on masque toujours au
    bon endroit), mais l'attribution devient fausse. C'est pourquoi l'appelant
    doit garder le tracker, et pourquoi ce module ne le choisit pas à sa place.
    """
    if fps <= 0:
        raise ValueError("fps doit être > 0 : sans lui, aucun instant n'est calculable")
    if frame_decimation < 1:
        raise ValueError("frame_decimation doit être ≥ 1")

    out: list[Echantillon] = []
    for fp in frame_poses:
        # ⚠ L'INSTANT SE CALCULE SUR `frame_idx`, JAMAIS SUR LE RANG DANS LA
        # LISTE : une frame sans personne détectée n'y figure pas, et compter
        # les éléments décalerait tout ce qui suit.
        t_s = float(getattr(fp, "frame_idx")) / fps
        pid = getattr(fp, "track_id", None)
        if pid is None:
            pid = getattr(fp, "person_id")
        kps: Sequence[Sequence[float]] = [
            (k.x, k.y, k.score) for k in getattr(fp, "keypoints")
        ]
        out.append(Echantillon(t_s=t_s, person_id=int(pid), keypoints=kps))
    return out


def masques_pour_un_fichier(
    chemin: Path,
    *,
    resultat_pose,
    pipeline_version: str,
    rotation_deg: int | None = None,
    quality_flags: Sequence[str] = (),
) -> dict:
    """Le payload `masques.v1` d'une vidéo, à partir de sa pose déjà extraite.

    ⚠ La pose est passée en PARAMÈTRE plutôt que calculée ici : extraire la pose
    demande un modèle (YOLOv8 + RTMPose) et des minutes de CPU. Les séparer rend
    ce pont testable **sans installer quoi que ce soit** — et c'est ce qui permet
    à ce module d'avoir des tests dans un dépôt qui n'a pas de CI.
    """
    # ⚠⚠ LA ROTATION SE DÉDUIT DU RÉSULTAT — ELLE NE S'ATTEND PAS D'UN APPELANT.
    #
    # Ce paramètre valait `int = 0` par défaut, et personne ne le renseignait.
    # Le masque annonçait donc « aucune rotation » avec des boîtes calculées
    # dans un repère REDRESSÉ (`extract_pose` tourne les clips portrait de 90°
    # anti-horaire, cf. `detect_smart_rotation`, ADR-0027 §3).
    #
    # ⚠ Constaté sur une vraie vidéo de cours le 2026-09-01 : appliquées telles
    # quelles, les boîtes tombaient sur le PARQUET et sur les MURS — aucun des
    # sept visages n'était couvert. Et le schéma de `masques.v1` portait déjà
    # l'avertissement mot pour mot : « des coordonnées calculées dans un repère
    # et appliquées dans l'autre donnent un masque qui rate le visage SANS
    # lever la moindre erreur ». Le champ existait POUR fermer ce piège.
    #
    # ⚠ Un défaut qu'un appelant doit penser à corriger est un défaut qui
    # revient. On DÉDUIT ; `rotation_deg` explicite ne sert plus qu'à forcer,
    # et il faut alors le vouloir.
    if rotation_deg is None:
        rotation_deg = 270 if getattr(resultat_pose, "rotate_ccw", False) else 0

    fps = float(resultat_pose.fps)
    frames = list(resultat_pose.frame_poses)
    duree_s = float(resultat_pose.n_frames) / fps if fps > 0 else 0.0
    # ⚠ L'échantillonnage RÉEL, pas celui qu'on a demandé : si l'extraction a
    # sauté des frames, annoncer la cadence théorique ferait mentir le contrat.
    hz = fps / float(resultat_pose.frame_decimation) if resultat_pose.frame_decimation else fps

    return construire_masques(
        video_md5=md5_du_fichier(chemin),
        duree_s=duree_s,
        rotation_deg=rotation_deg,
        echantillonnage_hz=hz,
        echantillons=echantillons_depuis_frames(
            frames, fps=fps, frame_decimation=resultat_pose.frame_decimation
        ),
        pipeline_version=pipeline_version,
        quality_flags=list(quality_flags),
    )
