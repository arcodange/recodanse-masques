"""Masques de visages pour ce qui devient PUBLIC — producteur du contrat `masques.v1` (#234).

Ce module dit **OÙ** masquer et **QUAND**. Il n'applique rien : `kadans-jobs`
applique avec ffmpeg (`boxblur` / `drawbox`) et **constate sur les pixels**
(kadans-dossier, ADR-0033). La frontière est le contrat, pas le conteneur.

═══════════════════════════════════════════════════════════════════════════════
IL N'Y A AUCUN DÉTECTEUR ICI, ET C'EST LE POINT

`pipeline/pose.py` produit déjà **133 keypoints COCO-WholeBody** par personne et
par frame, dont **68 pour le visage** (`KP_FACE = range(23, 91)`), en
coordonnées **déjà normalisées [0,1]** et **déjà redressées** (la rotation est
appliquée à l'extraction). Ce module en est un **consommateur** : il taille des
boîtes, il ne regarde pas d'image.

═══════════════════════════════════════════════════════════════════════════════
LA RÈGLE QUI GOUVERNE TOUS LES ARBITRAGES DE CE FICHIER

⚠ **DANS LE DOUTE, ON AGRANDIT.** Une boîte trop grande floute un bout d'épaule ;
une boîte trop petite laisse un visage nu sur une carte publique. Les deux
erreurs n'ont pas le même prix, donc aucune décision ici n'est symétrique :
l'union plutôt que l'interpolation, la dilatation plutôt que le collage, le
repli sur la boîte du visage entier plutôt que l'absence de barre.

⚠ **ET CE MODULE NE PROMET RIEN.** Il ne sait pas ce qu'il n'a pas vu : entre
deux échantillons il n'y a **aucune observation**, seulement de la marge. Le
**rappel** — quelle proportion des visages visibles est effectivement couverte —
est une mesure, publiée comme un **taux** (`notebooks/20-masquage-visages`),
jamais une garantie. Le produit ne dit jamais « anonyme ».
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from pipeline.pose import KP_BODY, KP_FACE  # ⚠ IMPORTER, ne pas redéfinir (#153)

# ── Les yeux dans les 68 points du visage ────────────────────────────────────
# Le bloc facial de COCO-WholeBody suit l'ordonnancement 68 points classique
# (300W/dlib) : mâchoire 0-16, sourcils 17-26, nez 27-35, **œil droit 36-41**,
# **œil gauche 42-47**, bouche 48-67. Les indices GLOBAUX s'en déduisent par le
# décalage du bloc.
_OFFSET_VISAGE = KP_FACE[0]                                   # 23
KP_OEIL_DROIT = [_OFFSET_VISAGE + i for i in range(36, 42)]   # 59..64
KP_OEIL_GAUCHE = [_OFFSET_VISAGE + i for i in range(42, 48)]  # 65..70
KP_YEUX = KP_OEIL_DROIT + KP_OEIL_GAUCHE

# Repli quand le bloc facial est trop faible : les 5 points de TÊTE du corps
# (COCO-17 : 0 nez, 1 œil gauche, 2 œil droit, 3 oreille gauche, 4 oreille
# droite). Beaucoup plus grossier, et beaucoup mieux que rien.
KP_TETE_CORPS = KP_BODY[0:5]
KP_YEUX_CORPS = KP_BODY[1:3]

# ── Les réglages, et ce que chacun coûte ─────────────────────────────────────
# ⚠ AUCUNE de ces valeurs n'est « raisonnable par principe » : ce sont des
# DÉFAUTS, et le notebook les mesure. Les changer sans remesurer, c'est
# remplacer une garantie par une intention.
SEUIL_POINT = 0.30          # en dessous, un keypoint ne dit rien de fiable
MIN_POINTS_VISAGE = 8       # moins que ça, le bloc facial n'a pas de forme
MIN_POINTS_YEUX = 4         # une barre se pose sur deux yeux, pas sur un reflet
ECARTS_ABERRANTS = 4.0      # au-delà de 4 écarts MÉDIANS du centre, un point ment
MARGE_VISAGE = 0.35         # +35 % de la plus grande dimension, sur les 4 côtés
MARGE_YEUX_X = 0.35         # la LARGEUR déborde : il faut couvrir les deux coins
                            # externes, et un regard qui part de côté
MARGE_YEUX_Y = 0.22         # ⚠ LA HAUTEUR, ELLE, SE RETIENT — retour fondateur
                            # 2026-08-29 : « peut-on rendre le masque des yeux
                            # plus fin ? ». Elle était à 0,60 comme la largeur,
                            # ce qui donnait une bande deux fois trop épaisse :
                            # une marge PROPORTIONNELLE à la plus grande
                            # dimension (la largeur, sur une bande plate) gonfle
                            # la hauteur du même montant absolu. Deux marges,
                            # parce que la bande des yeux n'est pas un carré.
HAUTEUR_MIN_YEUX = 0.022    # plancher en fraction de cadre. ⚠ Abaissé de 0,035 :
                            # sur un visage lointain il faisait à lui seul TOUTE
                            # la hauteur, donc une barre plus haute que le visage.
                            # Il reste, parce qu'une bande de 3 px ne masque rien
                            # — mais il borne, il ne dessine pas.
DILATATION_S = 0.25         # de chaque côté : survit à UNE détection perdue
FACTEUR_UNION_MAX = 1.6     # au-delà, l'union ne masque plus un visage : elle
                            # masque la danse — on coupe une nouvelle région
CADENCE_CIBLE_HZ = 6.0      # ⚠ L'ARBITRAGE CENTRAL DE CE MODULE, et il n'a pas
                            # de bonne réponse : PEU de régions ⇒ de GROSSES boîtes
                            # (l'union couvre le mouvement, donc masque la danse) ;
                            # BEAUCOUP de régions ⇒ un graphe ffmpeg ingérable chez
                            # le consommateur (une instance de filtre par région).
                            # Mesuré sur d07 : la pose y est à 30 Hz et TOUT
                            # fusionnait jusqu'au plafond — 378 régions de 0,19 × 0,13
                            # de cadre, soit la borne atteinte à chaque fois. À 6 Hz,
                            # le visage se déplace d'environ 0,025 de cadre entre deux
                            # échantillons (médiane 0,005 par 1/30 s), pour une boîte
                            # large de 0,065 : la MARGE de 35 % couvre ce déplacement,
                            # donc l'union n'a presque plus à grossir.
PLAFOND_DIMENSION = 0.22    # ⚠ PLAFOND ABSOLU, en fraction de cadre. Le facteur
                            # d'union seul ne suffit pas : il est RELATIF, donc une
                            # boîte individuelle aberrante relève le plafond qu'elle
                            # devrait déclencher. Mesuré sur d07 (2026-08-29) : la
                            # boîte médiane fait 0,065 × 0,075 de cadre, mais le
                            # max individuel monte à 0,27 — soit 11× l'aire médiane.
                            # Adossée à CE max, l'autorisation laissait passer une
                            # union de 0,50 × 0,21 : la moitié de l'écran, pour deux
                            # régions couvrant toute la vidéo. Une borne relative se
                            # fait toujours déborder par la queue qu'elle mesure


@dataclass(frozen=True)
class Boite:
    """Une boîte normalisée [0,1] du repère REDRESSÉ, origine en haut à gauche."""

    x: float
    y: float
    w: float
    h: float

    @property
    def aire(self) -> float:
        return self.w * self.h

    def union(self, autre: "Boite") -> "Boite":
        x0 = min(self.x, autre.x)
        y0 = min(self.y, autre.y)
        x1 = max(self.x + self.w, autre.x + autre.w)
        y1 = max(self.y + self.h, autre.y + autre.h)
        return Boite(x0, y0, x1 - x0, y1 - y0)

    def en_dict(self, decimales: int = 4) -> dict[str, float]:
        return {
            "x": round(self.x, decimales),
            "y": round(self.y, decimales),
            "w": round(self.w, decimales),
            "h": round(self.h, decimales),
        }


@dataclass(frozen=True)
class Echantillon:
    """Une personne, à un instant, telle que la pose l'a vue."""

    t_s: float
    person_id: int
    keypoints: Sequence[Sequence[float]]  # 133 × (x, y, score)


def _points_fiables(
    keypoints: Sequence[Sequence[float]], indices: Iterable[int], seuil: float
) -> list[tuple[float, float]]:
    pts = []
    for i in indices:
        if i >= len(keypoints):
            continue
        x, y, score = keypoints[i][0], keypoints[i][1], keypoints[i][2]
        if score is None or score <= seuil:
            continue
        pts.append((float(x), float(y)))
    return pts


def _sans_aberrants(
    pts: list[tuple[float, float]], seuil_ecarts: float = ECARTS_ABERRANTS
) -> list[tuple[float, float]]:
    """Jeter les points qui mentent, avant de prendre l'enveloppe.

    ⚠ MESURÉ, PAS SUPPOSÉ (2026-08-29, 14 504 visages du corpus) — ET LA MESURE
    A CORRIGÉ CE QU'ON CROYAIT. Le premier relevé comparait l'enveloppe min/max à
    une enveloppe p10/p90 et concluait « 1,8× trop haute en médiane » : c'était un
    artefact du repère, p10/p90 **rogne le vrai visage** (menton et front partent
    avec les 20 % écartés). Le cas courant n'est pas gonflé du tout.

    **C'est la QUEUE qui l'est, et elle seule.** Sur les 14 504 visages, l'écart
    médian entre boîte brute et boîte nettoyée est de **1,00×** — les bonnes
    détections ne bougent pas d'un cheveu — mais **27 boîtes couvraient plus de la
    moitié de la hauteur du cadre**, jusqu'à **36×** la taille réelle. Une boîte
    trop grande « couvre », donc elle a l'air sûre ; à cette taille elle ne masque
    plus un visage, elle **masque la danse**, et l'extrait ne sert plus à rien.
    Après écartement : **27 → 3**, et le maximum tombe de 0,72 à 0,64 de cadre.

    Un ou deux points partis à l'autre bout du cadre suffisent, et ⚠ **rien dans
    leur score ne les trahit** : ils passent le seuil de confiance comme les
    autres. C'est pourquoi le filtre est GÉOMÉTRIQUE et pas un seuil de plus.

    On écarte donc par la géométrie : au-delà de `seuil_ecarts` écarts MÉDIANS
    (MAD) du centre, un point ne décrit plus le même visage. La médiane est
    choisie plutôt que la moyenne précisément parce qu'elle ne se laisse pas
    tirer par ce qu'on cherche à jeter.

    ⚠ On ne jette JAMAIS jusqu'à vider : sous 60 % de survivants, le nuage n'a
    pas un aberrant, il a une forme qu'on ne comprend pas — et là on garde tout
    et on couvre par excès, fidèle à « dans le doute, on agrandit ».
    """
    if len(pts) < 5:
        return pts
    mx = _mediane([p[0] for p in pts])
    my = _mediane([p[1] for p in pts])
    dists = [abs(p[0] - mx) + abs(p[1] - my) for p in pts]
    mad = _mediane(dists)
    if mad <= 0:
        return pts
    gardes = [p for p, d in zip(pts, dists) if d <= seuil_ecarts * mad]
    if len(gardes) < 0.6 * len(pts) or len(gardes) < 4:
        return pts
    return gardes


def _mediane(valeurs: list[float]) -> float:
    v = sorted(valeurs)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def _borner(b: Boite) -> Boite:
    """Ramener la boîte DANS le cadre — le contrat exige x+w ≤ 1 et y+h ≤ 1."""
    x = min(max(b.x, 0.0), 1.0)
    y = min(max(b.y, 0.0), 1.0)
    w = min(b.w, 1.0 - x)
    h = min(b.h, 1.0 - y)
    return Boite(x, y, max(w, 1e-6), max(h, 1e-6))


def _boite_avec_marge(
    pts: list[tuple[float, float]], marge: float, marge_y: float | None = None
) -> Boite:
    """L'enveloppe des points, élargie.

    ⚠ LA MARGE EST PROPORTIONNELLE À LA PLUS GRANDE DIMENSION, jamais à chaque
    côté séparément. Un visage de profil donne une enveloppe très étroite : une
    marge proportionnelle à la largeur y ajouterait quelques pixels, là où il en
    faut autant que pour un visage de face à la même distance. C'est l'écart le
    plus coûteux qu'on puisse faire ici, parce qu'il frappe précisément le cas
    (le profil) où la détection est déjà la plus faible.
    """
    pts = _sans_aberrants(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    d = max(x1 - x0, y1 - y0)
    mx = d * marge
    # ⚠ SANS MARGE VERTICALE PROPRE, UNE BANDE PLATE DEVIENT UN CARRÉ. La marge
    # est proportionnelle à la plus GRANDE dimension — sur les yeux, la largeur.
    # Appliquée telle quelle en hauteur, elle ajoute en haut et en bas autant que
    # sur les côtés : la bande des yeux triplait d'épaisseur. Le visage, lui,
    # garde une marge isotrope (`marge_y is None`) : il EST à peu près carré.
    my = d * (marge if marge_y is None else marge_y)
    return _borner(Boite(x0 - mx, y0 - my, (x1 - x0) + 2 * mx, (y1 - y0) + 2 * my))


def boite_visage(
    keypoints: Sequence[Sequence[float]], seuil: float = SEUIL_POINT
) -> Boite | None:
    """La boîte du visage — bloc facial d'abord, points de tête du corps en repli."""
    pts = _points_fiables(keypoints, KP_FACE, seuil)
    if len(pts) >= MIN_POINTS_VISAGE:
        return _boite_avec_marge(pts, MARGE_VISAGE)
    # Repli : nez + yeux + oreilles. L'enveloppe est bien plus serrée que le
    # visage réel (pas de menton, pas de front), d'où la marge DOUBLÉE.
    pts = _points_fiables(keypoints, KP_TETE_CORPS, seuil)
    if len(pts) >= 2:
        return _boite_avec_marge(pts, MARGE_VISAGE * 2)
    return None


def boite_yeux(
    keypoints: Sequence[Sequence[float]], seuil: float = SEUIL_POINT
) -> Boite | None:
    """La bande des yeux — de quoi poser une barre.

    ⚠ Rendre `None` est un RÉSULTAT, pas un échec : de profil, de dos ou sous
    occlusion, les contours d'yeux ne se localisent pas. Le contrat le prévoit,
    et le consommateur a sa règle — barre demandée + `yeux` absent ⇒ on masque
    le VISAGE entier. On ne laisse jamais une région nue.
    """
    pts = _points_fiables(keypoints, KP_YEUX, seuil)
    if len(pts) < MIN_POINTS_YEUX:
        pts = _points_fiables(keypoints, KP_YEUX_CORPS, seuil)
        if len(pts) < 2:
            return None
    b = _boite_avec_marge(pts, MARGE_YEUX_X, MARGE_YEUX_Y)
    if b.h < HAUTEUR_MIN_YEUX:
        manque = HAUTEUR_MIN_YEUX - b.h
        b = _borner(Boite(b.x, b.y - manque / 2, b.w, HAUTEUR_MIN_YEUX))
    return b


@dataclass
class Region:
    t_start_s: float
    t_end_s: float
    person_id: int
    visage: Boite
    yeux: Boite | None
    confidence: float


def _confiance(keypoints: Sequence[Sequence[float]], indices: Iterable[int]) -> float:
    scores = [
        float(keypoints[i][2])
        for i in indices
        if i < len(keypoints) and keypoints[i][2] is not None
    ]
    if not scores:
        return 0.0
    return max(0.0, min(1.0, sum(scores) / len(scores)))


def assembler_regions(
    echantillons: Iterable[Echantillon],
    pas_s: float,
    dilatation_s: float = DILATATION_S,
    facteur_union_max: float = FACTEUR_UNION_MAX,
    plafond_dimension: float = PLAFOND_DIMENSION,
    seuil: float = SEUIL_POINT,
) -> list[Region]:
    """Des échantillons épars → des régions qui tiennent dans le temps.

    ⚠ ENTRE DEUX ÉCHANTILLONS, ON N'A RIEN VU. À 6 Hz, un visage parcourt une
    distance réelle entre deux observations. Deux façons de combler :

      · **interpoler** — supposer une trajectoire. Séduisant, et FAUX dès qu'une
        détection manque : on interpole alors par-dessus le trou, donc on affirme
        une position là où l'on n'a pas regardé, et le masque rate ;
      · **UNIONNER** — couvrir les deux positions observées ET tout l'espace
        entre elles. On ne suppose rien, et on couvre par excès.

    On unionne. Le coût est une boîte plus large ; le prix de l'autre erreur est
    un visage nu. ⚠ Mais l'union sans borne finirait par couvrir l'écran : on
    coupe une nouvelle région dès que l'union dépasse `facteur_union_max` fois la
    plus grande boîte individuelle — un plafond qui empêche un sujet qui traverse
    le cadre de faire flouter toute la danse.
    """
    par_personne: dict[int, list[Echantillon]] = {}
    for e in echantillons:
        par_personne.setdefault(e.person_id, []).append(e)

    regions: list[Region] = []
    for pid, ech in par_personne.items():
        ech.sort(key=lambda e: e.t_s)
        courant: Region | None = None
        aires: list[float] = []
        for e in ech:
            v = boite_visage(e.keypoints, seuil)
            if v is None:
                courant = None  # rien vu : on ne prolonge pas une région à l'aveugle
                aires = []
                continue
            y = boite_yeux(e.keypoints, seuil)
            c = _confiance(e.keypoints, KP_FACE)
            debut = max(0.0, e.t_s - dilatation_s)
            fin = e.t_s + pas_s + dilatation_s

            if courant is None or debut > courant.t_end_s:
                courant = Region(debut, fin, pid, v, y, c)
                aires = [v.aire]
                regions.append(courant)
                continue

            union = courant.visage.union(v)
            aires.append(v.aire)
            # ⚠ L'AUTORISATION SE MESURE SUR LA MÉDIANE, PAS SUR LE MAXIMUM.
            # Adossée au max, elle est fixée par la boîte la plus aberrante de la
            # région — c'est-à-dire que **plus une détection est mauvaise, plus on
            # s'autorise à grossir**. La médiane ne se laisse pas tirer par ce
            # qu'on cherche justement à ne pas suivre.
            reference = _mediane(aires)
            trop_grand = (
                union.aire > facteur_union_max * reference
                or union.w > plafond_dimension
                or union.h > plafond_dimension
            )
            if trop_grand:
                courant = Region(debut, fin, pid, v, y, c)
                aires = [v.aire]
                regions.append(courant)
                continue

            courant.t_end_s = max(courant.t_end_s, fin)
            courant.visage = union
            # ⚠ Les yeux s'unionnent AUSSI, et une région qui a vu des yeux les
            # garde : perdre la bande parce que le dernier échantillon était de
            # profil ferait retomber toute la région sur le visage entier.
            if y is not None:
                courant.yeux = y if courant.yeux is None else courant.yeux.union(y)
            courant.confidence = max(courant.confidence, c)

    regions.sort(key=lambda r: (r.t_start_s, r.person_id))
    return regions


def construire_masques(
    *,
    video_md5: str,
    duree_s: float,
    rotation_deg: int,
    echantillonnage_hz: float,
    echantillons: Iterable[Echantillon],
    pipeline_version: str,
    quality_flags: Sequence[str] = (),
    dilatation_s: float = DILATATION_S,
    seuil: float = SEUIL_POINT,
) -> dict:
    """Le payload `masques.v1`, prêt à valider contre le schéma."""
    if echantillonnage_hz <= 0:
        raise ValueError("echantillonnage_hz doit être strictement positif")
    if duree_s <= 0:
        raise ValueError("duree_s doit être strictement positive")
    pas_s = 1.0 / echantillonnage_hz
    regions = assembler_regions(
        echantillons, pas_s=pas_s, dilatation_s=dilatation_s, seuil=seuil
    )

    flags = list(dict.fromkeys(quality_flags))
    sans_yeux = sum(1 for r in regions if r.yeux is None)
    # ⚠ Le drapeau se pose sur un FAIT mesuré, pas sur une impression : quand
    # une région sur quatre n'a pas d'yeux localisables, une barre retombera
    # souvent sur le visage entier — le consommateur a le droit de le savoir
    # AVANT de choisir son traitement.
    if regions and sans_yeux / len(regions) >= 0.25 and "profil" not in flags:
        flags.append("profil")

    recommande = "flou" if regions and sans_yeux / len(regions) >= 0.5 else "barre_yeux"

    return {
        "schema_version": "masques.v1",
        "pipeline_version": pipeline_version,
        "video_md5": video_md5,
        "duree_s": round(float(duree_s), 3),
        "rotation_deg": int(rotation_deg),
        "echantillonnage_hz": round(float(echantillonnage_hz), 4),
        "traitement_recommande": recommande,
        "regions": [
            {
                "t_start_s": round(r.t_start_s, 3),
                "t_end_s": round(min(r.t_end_s, duree_s), 3),
                "person_id": int(r.person_id),
                "visage": r.visage.en_dict(),
                **({"yeux": r.yeux.en_dict()} if r.yeux is not None else {}),
                "confidence": round(r.confidence, 4),
            }
            for r in regions
            if r.t_start_s < duree_s
        ],
        "quality_flags": flags,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# QUI EST LE SUJET, ET QUI EST LA SALLE — le classement (fondateur, 2026-08-29)
# ═══════════════════════════════════════════════════════════════════════════════
#
# « Trouver une vidéo avec une foule, discriminer le/les danseur(s) et ne
# flouter/masquer que la foule. »
#
# ⚠⚠ ET C'EST UNE INVERSION DU DÉFAUT, PAS UN RÉGLAGE. Masquer TOUT LE MONDE et
# dévoiler LE PROF CRÉDITÉ n'est pas la même chose que masquer la salle : dans le
# premier cas, une erreur de classement laisse un élève masqué (sans gravité) ;
# dans le second, elle PUBLIE LE VISAGE de quelqu'un qui n'a rien demandé.
#
# ⚠⚠⚠ D'OÙ LA RÈGLE QUI GOUVERNE CE BLOC : **LA MACHINE PROPOSE, ELLE NE DÉSIGNE
# JAMAIS.** Aucune heuristique ne sait qui est « le prof » — elle sait qui est
# grand, qui bouge, qui est au centre. Un classement qui déciderait seul
# publierait un visage sur une corrélation. Le rang sert à PRÉ-SÉLECTIONNER dans
# un écran où quelqu'un confirme ; il ne lève jamais un masque tout seul. C'est
# la doctrine du produit appliquée telle quelle (RE-3 : monter est un GESTE ;
# RE-14 : l'arbitrage n'écrit jamais tout seul).
#
# Deux signaux, et ils sont indépendants l'un de l'autre :
#
#   · la TAILLE apparente — un sujet filmé est plus près, donc plus grand ;
#   · le MOUVEMENT — un danseur bouge, une salle assise ne bouge pas. C'est
#     exactement le signal de `audience_penalty`, LU À L'ENVERS : là-bas il sert
#     à ÉCARTER l'audience de l'analyse, ici à la RECONNAÎTRE pour la protéger.


@dataclass
class RangSujet:
    """Ce qu'on peut dire d'une personne suivie — sans jamais dire qui elle est."""

    person_id: int
    taille_mediane: float   # aire médiane de la boîte du corps, en fraction de cadre
    mouvement: float        # déplacement médian du centre entre deux échantillons
    echantillons: int
    score: float            # ⚠ un rang de PRÉ-SÉLECTION, jamais une décision


def classer_sujets(echantillons: Iterable[Echantillon]) -> list[RangSujet]:
    """Classer les personnes suivies, du plus « au premier plan » au moins.

    ⚠ CE QUE CE CLASSEMENT N'EST PAS : une identification. Il ne dit pas « c'est
    le prof » — il dit « celle-ci est grande et bouge, celle-là est petite et
    immobile ». Le produit s'en sert pour PROPOSER une sélection à confirmer.

    ⚠ ET IL NE REND JAMAIS UNE LISTE VIDE DE MASQUES. Un appelant qui prendrait
    le premier rang comme « le sujet » et laisserait tout le reste nu ferait
    exactement l'erreur que ce module existe pour empêcher : le défaut reste de
    masquer, et on ne DÉVOILE que ce qu'un humain a désigné.
    """
    par_personne: dict[int, list[Echantillon]] = {}
    for e in echantillons:
        par_personne.setdefault(e.person_id, []).append(e)

    rangs: list[RangSujet] = []
    for pid, ech in par_personne.items():
        ech.sort(key=lambda e: e.t_s)
        aires, centres = [], []
        for e in ech:
            pts = _points_fiables(e.keypoints, KP_BODY, SEUIL_POINT)
            if len(pts) < 5:
                continue
            pts = _sans_aberrants(pts)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            aires.append((max(xs) - min(xs)) * (max(ys) - min(ys)))
            centres.append(((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2))
        if not aires:
            continue
        depl = [
            abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(centres, centres[1:])
        ]
        taille = _mediane(aires)
        mouvement = _mediane(depl) if depl else 0.0
        # ⚠ LE SCORE EST DÉLIBÉRÉMENT GROSSIER. Un score fin donnerait l'illusion
        # d'une mesure ; ce n'est qu'un ORDRE d'affichage. La taille domine (elle
        # sépare le premier plan du fond, qui est la vraie question), le
        # mouvement départage à taille égale.
        rangs.append(RangSujet(
            person_id=pid, taille_mediane=taille, mouvement=mouvement,
            echantillons=len(aires), score=taille * (1.0 + mouvement),
        ))
    rangs.sort(key=lambda r: r.score, reverse=True)
    return rangs
