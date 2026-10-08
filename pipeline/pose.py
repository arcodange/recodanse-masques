"""Pose extraction RTMPose-WholeBody (V0.8) — voir [`adr/0016`](../adr/0016-pose-extraction-rtmpose.md).

Sortie : 133 keypoints par frame par personne, en coordonnées normalisées [0, 1]
par largeur/hauteur image. Frame decimation 1 sur 5 par défaut.

Stockage : table `pose_keypoints` (V0 plumbing déjà créée).
Cache JSON local `/tmp/video_analysis/pose/<canonical>/poses.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal


# Indices COCO-WholeBody — utiles pour les heuristiques de sélection / filtrage
KP_BODY = list(range(0, 17))
KP_FACE = list(range(23, 91))   # 17+6 (foot) à 91 — voir doc COCO-WholeBody
KP_LEFT_HAND = list(range(91, 112))
KP_RIGHT_HAND = list(range(112, 133))


@dataclass
class Keypoint:
    x: float       # normalisé [0, 1]
    y: float       # normalisé [0, 1]
    score: float

    def to_triplet(self) -> tuple[float, float, float]:
        return (round(self.x, 4), round(self.y, 4), round(self.score, 4))


@dataclass
class FramePose:
    frame_idx: int
    person_id: int           # raw au moment de l'extraction ; sera 0 après select_main_subject
    keypoints: list[Keypoint]  # 133 entrées dans l'ordre COCO-WholeBody
    track_id: int | None = None  # V1.3 — ID stable BoT-SORT/ByteTrack (None si tracker=none)

    def to_kp_json(self, version: int = 1) -> str:
        return json.dumps(
            {"v": version, "kp": [list(k.to_triplet()) for k in self.keypoints]},
            separators=(",", ":"),
        )

    def bbox(self, score_threshold: float = 0.3) -> tuple[float, float, float, float] | None:
        """Bounding box (x_min, y_min, x_max, y_max) en coords normalisées."""
        xs = [k.x for k in self.keypoints if k.score > score_threshold]
        ys = [k.y for k in self.keypoints if k.score > score_threshold]
        if not xs or not ys:
            return None
        return (min(xs), min(ys), max(xs), max(ys))

    def center(self, score_threshold: float = 0.3) -> tuple[float, float] | None:
        b = self.bbox(score_threshold)
        if b is None:
            return None
        x_min, y_min, x_max, y_max = b
        return ((x_min + x_max) / 2, (y_min + y_max) / 2)

    def bbox_area(self, score_threshold: float = 0.3) -> float:
        b = self.bbox(score_threshold)
        if b is None:
            return 0.0
        x_min, y_min, x_max, y_max = b
        return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)


@dataclass
class PoseQuality:
    """Métriques de qualité d'une extraction pose. Voir [`adr/0017`](../adr/0017-kinesthetic-style-classifier.md)."""

    coverage: float          # n_main_frames / n_frames_processed_expected ; cible ≥ 0.80
    body_score_mean: float   # moyenne des scores des 17 keypoints corps ; cible ≥ 0.50
    bbox_area_mean: float    # aire bbox normalisée moyenne ; cible ≥ 0.05
    bbox_area_min: float
    n_main_frames: int
    n_frames_processed_expected: int

    @property
    def usable_for_kine(self) -> bool:
        """Renvoie True si la qualité passe les seuils V0.9 kine."""
        return (
            self.coverage >= 0.80
            and self.body_score_mean >= 0.50
            and self.bbox_area_mean >= 0.05
        )


@dataclass
class PoseExtractionResult:
    video_path: Path
    fps: float
    width: int
    height: int
    n_frames: int
    frame_decimation: int
    frame_poses: list[FramePose]            # toutes personnes détectées
    main_subject_per_frame: list[FramePose] = field(default_factory=list)  # 1 par frame, person_id=0
    quality: PoseQuality | None = None      # rempli par compute_quality()
    # ⚠⚠ LA ROTATION APPLIQUÉE, ET POURQUOI ELLE DOIT SORTIR D'ICI.
    #
    # `extract_pose` auto-détecte l'orientation (`detect_smart_rotation`,
    # ADR-0027 §3), tourne chaque frame de 90° ANTI-HORAIRE quand il le faut, et
    # normalise les coordonnées sur `width`/`height` D'APRÈS rotation. Jusqu'au
    # 2026-09-01, cette information mourait ici : l'appelant recevait des
    # coordonnées dans un repère REDRESSÉ sans aucun moyen de le savoir.
    #
    # Constaté sur une vraie vidéo de cours : le masque `masques.v1` annonçait
    # `rotation_deg: 0` pendant que ses boîtes étaient dans le repère tourné.
    # Appliquées telles quelles, elles peignaient des rectangles sur le PARQUET
    # et sur les MURS — et pas un seul des six visages n'était couvert.
    # ⚠ Le schéma du contrat écrivait déjà l'avertissement, mot pour mot :
    # « des coordonnées calculées dans un repère et appliquées dans l'autre
    # donnent un masque qui rate le visage SANS lever la moindre erreur ».
    rotate_ccw: bool = False


def extract_pose(
    video_path: Path,
    frame_decimation: int = 5,
    mode: Literal["lightweight", "balanced", "performance"] = "balanced",
    device: str = "cpu",
    backend: str = "onnxruntime",
    tracker: Literal["none", "bytetrack", "botsort"] = "botsort",
    yolo_model_name: str = "yolov8n-pose.pt",
    rotate_ccw: bool | None = None,
) -> PoseExtractionResult:
    """Extrait la pose whole-body sur 1 frame sur `frame_decimation`.

    V1.3 (2026-05-24) : par défaut, utilise YOLOv8 + BoT-SORT pour produire
    des `track_id` stables (voir [`adr/0027`](../adr/0027-atomes-couple-multi-sujet-pose3d.md)
    §5 et le sprint V1.3 BoT-SORT). Le pipeline :

    1. YOLOv8-pose.track(frame, tracker='botsort.yaml', persist=True) → bboxes + track_ids
    2. RTMPose-Wholebody.pose_model(frame, bboxes=...) → 133 kp/personne tracked
    3. Stockage `FramePose(track_id=…)` pour persistance multi-sujet

    Args:
        video_path: chemin vers la vidéo (canonique mp4_file de préférence).
        frame_decimation: 1 frame sur N (default 5 → 6 fps si source 30 fps).
        mode: trade-off vitesse/qualité rtmlib. "balanced" en V0.8.
        device: "cpu" en V0.8 (MPS/CUDA à tester en V0.8.1).
        backend: "onnxruntime" recommandé.
        tracker: V1.3 — "botsort" (default, recommandé), "bytetrack" (legacy),
            ou "none" (rtmlib détection brute, person_idx incrémental par frame).
        yolo_model_name: poids YOLOv8 à charger (yolov8n-pose.pt = nano, ~6MB).
            Utilisé seulement si tracker != "none".
        rotate_ccw: V1.3 — Si True, rotate chaque frame de 90° anti-horaire
            avant détection (corrige métadonnées vidéo manquantes). Si None,
            auto-detect via `detect_smart_rotation()` au premier sample.

    Returns:
        `PoseExtractionResult` avec frame_poses (toutes personnes détectées).
        Si tracker != "none", chaque FramePose a un `track_id` stable.
        `main_subject_per_frame` reste vide ; appeler `select_main_subject()` ensuite.
    """
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # noqa: F401  (utilisé indirectement)
    from rtmlib import Wholebody  # type: ignore[import-not-found]

    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    wholebody = Wholebody(mode=mode, backend=backend, device=device)

    # V1.3 — Charger YOLOv8 tracker si activé
    yolo_tracker = None
    tracker_yaml: str | None = None
    if tracker != "none":
        from ultralytics import YOLO  # noqa: PLC0415
        yolo_tracker = YOLO(yolo_model_name)
        tracker_yaml = f"{tracker}.yaml"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width_native = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height_native = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # V1.3 — Auto-détection rotation (cf. detect_smart_rotation, ADR-0027 §3)
    if rotate_ccw is None and yolo_tracker is not None:
        try:
            rotate_ccw = detect_smart_rotation(video_path, yolo_tracker)
        except Exception:
            rotate_ccw = False
    elif rotate_ccw is None:
        rotate_ccw = False

    # Width/height après rotation pour normalisation finale
    width = height_native if rotate_ccw else width_native
    height = width_native if rotate_ccw else height_native

    frame_idx = 0
    frame_poses: list[FramePose] = []

    # V1.3 — boucle process TOUTES les frames pour préserver continuité tracker
    # (mais ne stocke que 1 sur frame_decimation pour rester cohérent avec V0.8)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if rotate_ccw:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if yolo_tracker is not None:
            # Track sur TOUTES les frames (pas seulement à frame_decimation) pour
            # préserver la continuité du tracker. La décimation se fait au storage.
            try:
                results = yolo_tracker.track(
                    frame, persist=True, tracker=tracker_yaml, verbose=False,
                )
            except Exception:
                frame_idx += 1
                continue

            if frame_idx % frame_decimation != 0:
                frame_idx += 1
                continue

            r0 = results[0] if results else None
            if r0 is None or r0.boxes is None or r0.boxes.id is None:
                frame_idx += 1
                continue

            boxes_xyxy = r0.boxes.xyxy.cpu().numpy()
            track_ids = r0.boxes.id.int().cpu().numpy()
            if len(boxes_xyxy) == 0:
                frame_idx += 1
                continue

            # RTMPose sur les bboxes tracked → 133 kp wholebody par sujet
            try:
                keypoints, scores = wholebody.pose_model(
                    frame, bboxes=boxes_xyxy.tolist()
                )
            except Exception:
                frame_idx += 1
                continue

            if keypoints is None or len(keypoints) == 0:
                frame_idx += 1
                continue

            for person_idx in range(len(keypoints)):
                kpts = keypoints[person_idx]
                scs = scores[person_idx]
                kp_list: list[Keypoint] = []
                for i in range(min(len(kpts), 133)):
                    x = float(kpts[i][0]) / max(1.0, float(width))
                    y = float(kpts[i][1]) / max(1.0, float(height))
                    s = float(scs[i])
                    kp_list.append(Keypoint(x=x, y=y, score=s))
                while len(kp_list) < 133:
                    kp_list.append(Keypoint(x=0.0, y=0.0, score=0.0))

                tid = int(track_ids[person_idx]) if person_idx < len(track_ids) else None
                frame_poses.append(
                    FramePose(
                        frame_idx=frame_idx,
                        person_id=person_idx,
                        keypoints=kp_list,
                        track_id=tid,
                    )
                )
        else:
            # Path legacy : rtmlib pur (pas de tracking, person_idx incrémental)
            if frame_idx % frame_decimation == 0:
                try:
                    keypoints, scores = wholebody(frame)
                except Exception:
                    frame_idx += 1
                    continue

                if keypoints is None or len(keypoints) == 0:
                    frame_idx += 1
                    continue

                for person_idx, (kpts, scs) in enumerate(zip(keypoints, scores)):
                    kp_list = []
                    for i in range(min(len(kpts), 133)):
                        x = float(kpts[i][0]) / max(1.0, float(width))
                        y = float(kpts[i][1]) / max(1.0, float(height))
                        s = float(scs[i])
                        kp_list.append(Keypoint(x=x, y=y, score=s))
                    while len(kp_list) < 133:
                        kp_list.append(Keypoint(x=0.0, y=0.0, score=0.0))

                    frame_poses.append(
                        FramePose(
                            frame_idx=frame_idx,
                            person_id=person_idx,
                            keypoints=kp_list,
                            track_id=None,
                        )
                    )

        frame_idx += 1

    cap.release()

    return PoseExtractionResult(
        video_path=video_path,
        fps=fps,
        width=width,
        height=height,
        n_frames=n_total,
        frame_decimation=frame_decimation,
        rotate_ccw=bool(rotate_ccw),
        frame_poses=frame_poses,
    )


def select_main_subject(
    poses: Iterable[FramePose],
    weight_bbox: float = 0.4,
    weight_center: float = 0.3,
    weight_continuity: float = 0.3,
    weight_static_penalty: float = 1.0,
    static_threshold: float = 0.005,
) -> list[FramePose]:
    """Sélectionne le danseur principal frame par frame.

    Heuristique multi-passes :
    1. Pour chaque frame, score les candidats par : bbox_area * w1 + center_proximity * w2
       + continuity_with_prev * w3 - static_penalty * w4.
    2. Garde le meilleur, pénalise les candidats stationnaires (audience).
    3. Retourne une liste de FramePose avec `person_id=0` (sujet principal).

    Args:
        poses: tous les FramePose extraits (potentiellement N personnes par frame).
        weight_*: poids du score multi-critère (cf. ADR-0016).
        static_threshold: si bbox_area < ce seuil, on considère que c'est de l'audience inactive.

    Returns:
        Liste de FramePose, 1 par frame_idx, avec `person_id=0`.
    """
    from collections import defaultdict

    by_frame: dict[int, list[FramePose]] = defaultdict(list)
    for fp in poses:
        by_frame[fp.frame_idx].append(fp)

    out: list[FramePose] = []
    prev_center: tuple[float, float] | None = None

    for frame_idx in sorted(by_frame.keys()):
        scored: list[tuple[float, FramePose, tuple[float, float]]] = []

        for cand in by_frame[frame_idx]:
            center = cand.center()
            area = cand.bbox_area()
            if center is None or area < static_threshold:
                continue

            cx, cy = center
            dist_center = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
            score = weight_bbox * area + weight_center * (1.0 - min(dist_center, 1.0))

            if prev_center is not None:
                dx = cx - prev_center[0]
                dy = cy - prev_center[1]
                dist_prev = (dx ** 2 + dy ** 2) ** 0.5
                continuity = max(0.0, 1.0 - 5.0 * dist_prev)  # pénalise jumps
                score += weight_continuity * continuity

            scored.append((score, cand, (cx, cy)))

        if not scored:
            continue

        best_score, best_fp, best_center = max(scored, key=lambda t: t[0])
        # Re-tag person_id=0 pour cohérence DB (sujet principal toujours 0 en V0.8).
        # Préserve track_id si présent (V1.3+).
        out.append(FramePose(
            frame_idx=best_fp.frame_idx,
            person_id=0,
            keypoints=best_fp.keypoints,
            track_id=best_fp.track_id,
        ))
        prev_center = best_center

    return out


def select_main_subjects(
    poses: Iterable[FramePose],
    n: int = 2,
    weight_bbox: float = 0.4,
    weight_center: float = 0.1,
    weight_continuity: float = 0.5,
    static_threshold: float = 0.005,
) -> list[list[FramePose]]:
    """V0.11 — Sélectionne les N danseurs principaux par frame, avec stabilité temporelle.

    Pour le corpus bachata 100j (2 danseurs côte à côte exécutant la même choré) :
    - N=2 : leader + follower, assignés par position relative gauche/droite.
    - Tracking par proximité avec la frame précédente (weight_continuity élevé).
    - Le `weight_center` est faible (vs `select_main_subject` mono-sujet) car on s'attend
      à 2 sujets pas tous centrés.

    Args:
        poses: tous les FramePose extraits (potentiellement N personnes par frame).
        n: nombre de sujets à suivre.
        weight_*: poids du score multi-critère.
        static_threshold: bbox_area minimale (filtre l'audience inactive).

    Returns:
        Liste de N listes (1 par sujet), chacune contenant les FramePose du sujet
        à travers les frames avec `person_id` ∈ [0, n) stable. Sujets ordonnés
        par position spatiale moyenne sur l'axe x (subject 0 = gauche, 1 = droite).
    """
    from collections import defaultdict
    import numpy as np

    by_frame: dict[int, list[FramePose]] = defaultdict(list)
    for fp in poses:
        by_frame[fp.frame_idx].append(fp)

    # Tracks : liste de N pistes, chacune {frame_idx: FramePose}
    tracks: list[dict[int, FramePose]] = [{} for _ in range(n)]
    prev_centers: list[tuple[float, float] | None] = [None] * n

    for frame_idx in sorted(by_frame.keys()):
        # Filtre les candidats valides
        candidates = []
        for cand in by_frame[frame_idx]:
            center = cand.center()
            area = cand.bbox_area()
            if center is None or area < static_threshold:
                continue
            candidates.append((cand, center, area))

        if not candidates:
            continue

        # Matrice de score : track_i × candidate_j
        # Si prev_centers[i] est None (pas encore initialisé), on use bbox+center
        # Sinon on use continuité
        score_matrix = np.zeros((n, len(candidates)))
        for i in range(n):
            for j, (_, center, area) in enumerate(candidates):
                cx, cy = center
                s = weight_bbox * area
                if prev_centers[i] is not None:
                    dx = cx - prev_centers[i][0]
                    dy = cy - prev_centers[i][1]
                    dist = (dx ** 2 + dy ** 2) ** 0.5
                    s += weight_continuity * max(0.0, 1.0 - 5.0 * dist)
                else:
                    # Initial : favorise centre (frame 0) ; ensuite on use continuité
                    dist_center = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
                    s += weight_center * (1.0 - min(dist_center, 1.0))
                score_matrix[i, j] = s

        # Assignation gloutonne : pour chaque track, prendre le meilleur candidat
        # disponible. Évite que 2 tracks pointent vers le même candidat.
        used = set()
        assignments = [None] * n  # type: ignore
        # Tri des tracks par "préférence" : track avec prev_centers défini d'abord
        track_order = sorted(range(n), key=lambda i: prev_centers[i] is None)
        for track_i in track_order:
            available = [j for j in range(len(candidates)) if j not in used]
            if not available:
                break
            best_j = max(available, key=lambda j: score_matrix[track_i, j])
            assignments[track_i] = best_j
            used.add(best_j)

        for track_i, j in enumerate(assignments):
            if j is None:
                continue
            cand, center, _ = candidates[j]
            cand_copy = FramePose(frame_idx=cand.frame_idx, person_id=track_i, keypoints=cand.keypoints)
            tracks[track_i][frame_idx] = cand_copy
            prev_centers[track_i] = center

    # Ré-ordonner les tracks par position spatiale moyenne (left → 0, right → 1, …)
    track_centers_x = []
    for ti in range(n):
        xs = [fp.center()[0] for fp in tracks[ti].values() if fp.center() is not None]
        track_centers_x.append((np.mean(xs) if xs else 0.5, ti))
    track_centers_x.sort()  # tri par x croissant

    out: list[list[FramePose]] = []
    for new_id, (_, old_id) in enumerate(track_centers_x):
        track_frames = sorted(tracks[old_id].items())
        # Réassigner person_id selon le nouvel ordre
        out.append([
            FramePose(frame_idx=f, person_id=new_id, keypoints=fp.keypoints)
            for f, fp in track_frames
        ])
    return out


def n_persons_median(poses: Iterable[FramePose]) -> int:
    """V1.3 — Compte médian de personnes détectées par frame.

    Utilisé pour auto-décider mono-sujet (`select_main_subject`) vs multi-sujet
    (`keep_all_subjects`) au build_pose. Sur bachata d01-d10 (couple côte à côte),
    cette valeur est typiquement 2. Sur une vidéo solo, c'est 1. Sur d07 (turns avec
    fragmentation ByteTrack), peut être 3-7 → BoT-SORT améliore ce comptage.

    Args:
        poses: liste des FramePose extraits (toutes personnes par frame).

    Returns:
        Médiane du nombre de personnes détectées par frame. 0 si pas de poses.
    """
    from collections import Counter
    counts = Counter()
    for fp in poses:
        counts[fp.frame_idx] += 1
    if not counts:
        return 0
    sorted_counts = sorted(counts.values())
    mid = len(sorted_counts) // 2
    return sorted_counts[mid]


def keep_all_subjects(
    poses: Iterable[FramePose],
    min_frames_persistence: int = 10,
    max_subjects: int = 2,
) -> list[FramePose]:
    """V1.3 — Garde les sujets persistants tracked par BoT-SORT (vs `select_main_subject`).

    Pour les vidéos couple (n_persons_median ≥ 2), on conserve les `max_subjects`
    `track_id` les **plus stables** (top-N par nombre de frames). Cela filtre
    les fragmentations BoT-SORT sur les turn-heavy (d07-d09) où le tracker
    peut produire 3-7 IDs au lieu de 2.

    Le `person_id` est dérivé du `track_id` : on remappe sur 0..N-1 pour
    cohérence DB en ordonnant par ordre spatial (gauche → droite à la
    **première** apparition de chaque track persistant retenu).

    Args:
        poses: tous les FramePose extraits (avec track_id du tracker).
        min_frames_persistence: nombre de frames minimum pour qu'un track_id
            soit considéré stable. Default 10 (sur 30-100 frames typiques).
        max_subjects: nombre maximum de sujets à garder (top-N par persistence).
            Default 2 (un couple). Mettre None pour garder tous les sujets stables.

    Returns:
        Liste de FramePose avec `person_id` ∈ [0, N) basé sur ordre spatial.
        Garde le `track_id` original pour traçabilité.
    """
    poses_list = list(poses)
    if not poses_list:
        return []

    # Compte la persistence (nombre de frames) par track_id
    from collections import Counter
    track_counts: Counter[int] = Counter()
    for fp in poses_list:
        if fp.track_id is not None:
            track_counts[fp.track_id] += 1

    # Filtre les tracks stables (≥ min_frames_persistence)
    stable_tracks = {
        tid for tid, cnt in track_counts.items()
        if cnt >= min_frames_persistence
    }

    # Garde top-N par persistence (les 2 plus persistants en cas de couple)
    if max_subjects is not None:
        top_n = sorted(
            stable_tracks, key=lambda tid: track_counts[tid], reverse=True
        )[:max_subjects]
        stable_tracks = set(top_n)

    # Si aucun track stable, fallback vers tous (cas tracker disabled ou tracks éphémères)
    if not stable_tracks:
        stable_tracks = set(track_counts.keys())

    # Calcule l'ordre des tracks STABLES selon position spatiale au 1er apparition
    first_x_by_track: dict[int, float] = {}
    for fp in sorted(poses_list, key=lambda f: f.frame_idx):
        if fp.track_id is None or fp.track_id not in stable_tracks:
            continue
        if fp.track_id in first_x_by_track:
            continue
        center = fp.center()
        if center is None:
            continue
        first_x_by_track[fp.track_id] = center[0]

    # Trie par x_min croissant (gauche → droite)
    sorted_tracks = sorted(first_x_by_track.items(), key=lambda kv: kv[1])
    track_to_pid: dict[int, int] = {tid: idx for idx, (tid, _) in enumerate(sorted_tracks)}

    # Filtre + remap : ne garde que les frames dont le track_id est stable
    out: list[FramePose] = []
    for fp in poses_list:
        if fp.track_id is None:
            # Tracker disabled : garde tel quel avec person_id raw
            out.append(FramePose(
                frame_idx=fp.frame_idx,
                person_id=fp.person_id,
                keypoints=fp.keypoints,
                track_id=None,
            ))
        elif fp.track_id in track_to_pid:
            out.append(FramePose(
                frame_idx=fp.frame_idx,
                person_id=track_to_pid[fp.track_id],
                keypoints=fp.keypoints,
                track_id=fp.track_id,
            ))
        # Sinon : track_id éphémère, on drop (filtré)
    return out


def select_subjects_auto(
    poses: Iterable[FramePose],
    force_main_subject: bool = False,
    couple_threshold: int = 2,
) -> tuple[list[FramePose], Literal["mono", "couple"]]:
    """V1.3 — Auto-décide mono-sujet vs multi-sujet selon n_persons médian.

    Args:
        poses: tous les FramePose extraits.
        force_main_subject: forcer mode mono-sujet (utile pour une vidéo solo
            même si occasionnellement >1 détection par fragmentation).
        couple_threshold: n_persons_median ≥ ce seuil → mode couple. Default 2.

    Returns:
        (frames, mode) où mode ∈ {"mono", "couple"}. En mode mono, person_id=0
        pour tous. En mode couple, person_id ∈ [0, N) ordonné gauche→droite.
    """
    poses_list = list(poses)
    if force_main_subject:
        return select_main_subject(poses_list), "mono"
    n_med = n_persons_median(poses_list)
    if n_med >= couple_threshold:
        return keep_all_subjects(poses_list), "couple"
    return select_main_subject(poses_list), "mono"


def compute_quality(result: PoseExtractionResult) -> PoseQuality:
    """Calcule les métriques qualité (voir ADR-0017)."""
    n_expected = max(1, result.n_frames // max(1, result.frame_decimation))
    main = result.main_subject_per_frame
    coverage = len(main) / n_expected

    body_scores: list[float] = []
    bbox_areas: list[float] = []
    for fp in main:
        for i in range(17):  # body indices COCO-WholeBody
            body_scores.append(fp.keypoints[i].score)
        area = fp.bbox_area()
        bbox_areas.append(area)

    body_score_mean = sum(body_scores) / len(body_scores) if body_scores else 0.0
    bbox_area_mean = sum(bbox_areas) / len(bbox_areas) if bbox_areas else 0.0
    bbox_area_min = min(bbox_areas) if bbox_areas else 0.0

    return PoseQuality(
        coverage=coverage,
        body_score_mean=body_score_mean,
        bbox_area_mean=bbox_area_mean,
        bbox_area_min=bbox_area_min,
        n_main_frames=len(main),
        n_frames_processed_expected=n_expected,
    )


def cache_path_for(video_path: Path) -> Path:
    """Path du JSON cache des poses pour une vidéo donnée.

    Clé = `<parent_dir>/<stem>.json`. Le stem est nécessaire car deux sources
    de vidéos coexistent :

    - **DanceVideos/raw → videos/** : chaque vidéo dans son dossier canonique
      unique, fichier toujours nommé `video.mp4`. Cache : `<canonical>/video.json`.
    - **un corpus de référence** : **plusieurs vidéos par
      dossier** (ex : `<corpus>/d01.mp4, d02.mp4`...). Cache :
      `<corpus>/<stem>.json` pour éviter les collisions.

    Sans le stem, les vidéos de ce corpus partageraient toutes le même cache (bug fix
    historique 2026-05-09).
    """
    p = video_path.resolve()
    return Path(f"/tmp/video_analysis/pose/{p.parent.name}/{p.stem}.json")


def save_cache(
    result: PoseExtractionResult,
    cache_path: Path,
    include_all_subjects: bool = False,
) -> None:
    """Sauvegarde le résultat (sujet principal + quality + optionnellement tous les sujets) en JSON cache.

    V0.11 (2026-05-13) : ajout du flag `include_all_subjects` pour persister `frame_poses`
    (toutes personnes détectées par RTMPose). Nécessaire pour tracking multi-sujets bachata
    (Leader + Follower côte à côte). Augmente la taille du cache (~×2-5 selon densité de sujets).
    Default False pour rétro-compatibilité.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if result.quality is None:
        result.quality = compute_quality(result)
    payload = {
        "video": str(result.video_path),
        "fps": result.fps,
        "width": result.width,
        "height": result.height,
        "n_frames": result.n_frames,
        "frame_decimation": result.frame_decimation,
        "quality": {
            "coverage": round(result.quality.coverage, 4),
            "body_score_mean": round(result.quality.body_score_mean, 4),
            "bbox_area_mean": round(result.quality.bbox_area_mean, 4),
            "bbox_area_min": round(result.quality.bbox_area_min, 4),
            "n_main_frames": result.quality.n_main_frames,
            "n_frames_processed_expected": result.quality.n_frames_processed_expected,
            "usable_for_kine": result.quality.usable_for_kine,
        },
        "main_subject_per_frame": [
            {
                "frame_idx": fp.frame_idx,
                "person_id": fp.person_id,
                "track_id": fp.track_id,
                "kp": [list(k.to_triplet()) for k in fp.keypoints],
            }
            for fp in (result.main_subject_per_frame or [])
        ],
    }
    if include_all_subjects and result.frame_poses:
        payload["frame_poses"] = [
            {
                "frame_idx": fp.frame_idx,
                "person_id": fp.person_id,
                "track_id": fp.track_id,
                "kp": [list(k.to_triplet()) for k in fp.keypoints],
            }
            for fp in result.frame_poses
        ]
    cache_path.write_text(json.dumps(payload, separators=(",", ":")))


def load_cache(cache_path: Path) -> PoseExtractionResult | None:
    if not cache_path.exists():
        return None
    data = json.loads(cache_path.read_text())
    main_frames = []
    for item in data.get("main_subject_per_frame", []):
        kps = [Keypoint(x=t[0], y=t[1], score=t[2]) for t in item["kp"]]
        main_frames.append(FramePose(
            frame_idx=item["frame_idx"],
            person_id=item.get("person_id", 0),
            keypoints=kps,
            track_id=item.get("track_id"),
        ))

    q_data = data.get("quality")
    quality = (
        PoseQuality(
            coverage=q_data.get("coverage", 0.0),
            body_score_mean=q_data.get("body_score_mean", 0.0),
            bbox_area_mean=q_data.get("bbox_area_mean", 0.0),
            bbox_area_min=q_data.get("bbox_area_min", 0.0),
            n_main_frames=q_data.get("n_main_frames", 0),
            n_frames_processed_expected=q_data.get("n_frames_processed_expected", 0),
        )
        if q_data
        else None
    )

    # V0.11 — frame_poses (toutes personnes) si persisté
    frame_poses = []
    for item in data.get("frame_poses", []):
        kps = [Keypoint(x=t[0], y=t[1], score=t[2]) for t in item["kp"]]
        frame_poses.append(FramePose(
            frame_idx=item["frame_idx"],
            person_id=item.get("person_id", 0),
            keypoints=kps,
            track_id=item.get("track_id"),
        ))

    return PoseExtractionResult(
        video_path=Path(data["video"]),
        fps=data.get("fps", 0.0),
        width=data.get("width", 0),
        height=data.get("height", 0),
        # ⚠ Un cache écrit AVANT ce champ ne le porte pas. On ne suppose donc
        # pas `False` : on le DÉDUIT de la forme, comme le ferait un
        # consommateur — width < height sur une source stockée en paysage veut
        # dire qu'on a redressé. C'est moins précis qu'un drapeau, et c'est
        # exactement pour ça qu'on écrit le drapeau désormais.
        rotate_ccw=bool(data.get("rotate_ccw", False)),
        n_frames=data.get("n_frames", 0),
        frame_decimation=data.get("frame_decimation", 5),
        frame_poses=frame_poses,
        main_subject_per_frame=main_frames,
        quality=quality,
    )


# ───────────────────────────────────────────────────────────────────────────
# Rotation detection (anatomy-based, V0.11)
# ───────────────────────────────────────────────────────────────────────────
# Voir [feedback_verify_video_rotation](../memory/feedback_verify_video_rotation.md).
# Le détecteur count-based (`n_persons_raw vs n_persons_rotated`) est
# non-déterministe (égalité → False par défaut). Sur d01, MediaPipe avait
# interprété un couple "couché", squelette compressé d'un facteur 15.
# Remplacement : `verticality_score = log(bbox_h / bbox_w) × confidence`.

def _verticality_score_yolo(img_bgr, yolo_model) -> float:
    """Score d'orientation verticale pour une image via YOLOv8-pose.

    Vertical human → bbox h >> w → score positif.
    Horizontal human → bbox w > h → score négatif.

    Args:
        img_bgr: image cv2 BGR
        yolo_model: ultralytics YOLO instance avec keypoints (e.g. yolov8n-pose)

    Returns:
        float : somme `log(h/w) × conf` sur les personnes détectées (conf>0.3).
    """
    import numpy as np  # noqa: PLC0415  (import local au helper)
    res = yolo_model(img_bgr, verbose=False)
    if not res or res[0].boxes is None:
        return 0.0
    boxes = res[0].boxes.xyxy.cpu().numpy()
    confs = res[0].boxes.conf.cpu().numpy()
    score = 0.0
    for i in range(boxes.shape[0]):
        if confs[i] < 0.3:
            continue
        x1, y1, x2, y2 = boxes[i]
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        score += float(np.log(h / w)) * float(confs[i])
    return score


def detect_smart_rotation(
    video_path: Path,
    yolo_model=None,
    n_samples: int = 3,
) -> bool:
    """Détection de rotation basée sur l'anatomie (humains debout vs couchés).

    Sample `n_samples` frames réparties (ratios 0.2 à 0.8), compare le score
    de verticalité bbox `raw` vs `cv2.ROTATE_90_COUNTERCLOCKWISE`. Retourne
    True si la version rotée donne un meilleur score.

    Convention du projet : les vidéos verticales d'un corpus de référence
    (`<corpus>__d*`) sont stockées en landscape → besoin de rotate
    CCW. Les vidéos longues filmées en landscape natif → rotate_ccw=False.

    Args:
        video_path: chemin vidéo
        yolo_model: instance ultralytics YOLO (chargée par appelant pour éviter
                    le coût d'init répété). Si None, charge yolov8n-pose.pt.
        n_samples: nombre de frames à sampler (3 = robuste mais rapide)

    Returns:
        True si rotate CCW recommandé, False sinon.

    Voir [ADR-0027](../adr/0027-atomes-couple-multi-sujet-pose3d.md) §3.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    if yolo_model is None:
        from ultralytics import YOLO  # noqa: PLC0415
        yolo_model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(str(video_path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_ratios = np.linspace(0.2, 0.8, n_samples)
    score_raw = 0.0
    score_ccw = 0.0
    for r in sample_ratios:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n_total * r))
        ok, frame = cap.read()
        if not ok:
            continue
        score_raw += _verticality_score_yolo(frame, yolo_model)
        score_ccw += _verticality_score_yolo(
            cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE), yolo_model)
    cap.release()
    return score_ccw > score_raw


# ───────────────────────────────────────────────────────────────────────────
# Migration : ajout colonne `track_id` à `pose_keypoints` (V0.11+)
# ───────────────────────────────────────────────────────────────────────────

def migrate_add_track_id_column(conn) -> bool:
    """Ajoute la colonne `track_id` à `pose_keypoints` si absente. Idempotent.

    Returns True si migration appliquée, False si déjà présente.
    Voir [ADR-0027](../adr/0027-atomes-couple-multi-sujet-pose3d.md) Migrations.
    """
    from sqlalchemy import text  # noqa: PLC0415
    existing = {
        row[1]
        for row in conn.execute(text("PRAGMA table_info(pose_keypoints)")).fetchall()
    }
    if "track_id" in existing:
        return False
    conn.execute(text("ALTER TABLE pose_keypoints ADD COLUMN track_id INTEGER"))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_pose_kp_track "
        "ON pose_keypoints(video_file_name, track_id)"
    ))
    return True
