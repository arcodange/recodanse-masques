# RecoDanse — le suivi des personnes de la voie des masques

Ce dépôt publie le programme qui, sur le serveur de **RecoDanse**, repère et suit les
personnes d'une vidéo de danse pour calculer où se trouvent leurs visages. Il est publié
sous la **GNU Affero General Public License, version 3** (texte intégral : [`LICENSE`](LICENSE)),
parce qu'il s'appuie sur [ultralytics](https://github.com/ultralytics/ultralytics), lui-même
sous AGPL-3.0.

RecoDanse est une application web d'entraînement à la danse, éditée par **Arcodange**.

## Ce que fait ce programme

Quand une vidéo devient un extrait ou une affiche, le serveur de RecoDanse masque les
visages des personnes filmées. Ce programme fait la première moitié du travail : il dit
**où** et **quand** masquer. Il ne touche pas aux pixels.

1. `pipeline/pose.py` lit la vidéo avec OpenCV, la redresse si elle a été filmée en
   portrait, et suit chaque personne d'une image à l'autre avec **YOLOv8-pose** et le
   suivi **BoT-SORT** d'ultralytics ; puis **RTMW** (par [rtmlib](https://github.com/Tau-J/rtmlib),
   modèles d'OpenMMLab) estime 133 points du corps, du visage et des mains de chaque
   personne suivie.
2. `pipeline/masques_depuis_pose.py` traduit ces points en échantillons datés.
3. `pipeline/masques.py` en tire, pour chaque personne, une boîte autour du visage et une
   bande sur les yeux, élargies par une marge et tenues dans le temps.
4. `scripts/masques_for_file.py` écrit le résultat dans `<vidéo>.masques.kadans.json`
   (format `masques.v1` : la durée, la rotation appliquée, la cadence d'échantillonnage,
   puis une liste de régions `t_start_s`, `t_end_s`, `person_id`, `visage`, `yeux`
   facultatif, `confidence`, en coordonnées normalisées entre 0 et 1).

Le masque est une estimation : un visage que le détecteur n'a pas vu n'est pas dans la
liste.

## Ce que contient ce dépôt

| Fichier | Rôle |
| --- | --- |
| `scripts/masques_for_file.py` | Le point d'entrée : une vidéo en entrée, un fichier JSON en sortie |
| `pipeline/pose.py` | Lecture de la vidéo, redressement, suivi (ultralytics) et pose (rtmlib) |
| `pipeline/masques_depuis_pose.py` | Le pont entre la pose et les masques |
| `pipeline/masques.py` | La géométrie des boîtes (visage, yeux), les marges, la tenue dans le temps |
| `pipeline/__init__.py`, `pipeline/_suppress_objc_warnings.py` | Le paquet, et un filtre de messages de macOS (sans effet ailleurs) |
| `sitecustomize.py` | Borne les fils de calcul de torch, OpenCV et onnxruntime au quota du conteneur (variable `KADANS_FILS`) ; sans elle, ne fait rien |
| `requirements-torch.txt`, `requirements.lock.txt` | Les versions exactes des dépendances, celles qui tournent en production |
| `Dockerfile` | La construction de l'environnement complet, modèles compris |
| `LICENSE` | Le texte de l'AGPL-3.0 |

## D'où il vient dans RecoDanse

En production, ce programme tourne dans une image qui porte aussi d'autres traitements
de RecoDanse (le tempo de la musique, la transcription des paroles, la question « est-ce
de la danse ? »). Chacun est un **processus séparé** : aucun d'eux n'importe ultralytics,
et ils ne sont pas publiés ici. Un programme d'orchestration, lui aussi séparé, lance
celui-ci par une ligne de commande — `python scripts/masques_for_file.py <vidéo> --out <dossier>` —
puis lit le fichier JSON qu'il écrit et applique les masques avec ffmpeg.

Les fichiers de `pipeline/` et `scripts/` viennent du dépôt de recherche d'Arcodange, au
commit `5f00e0d` du 5 octobre 2026, la version déployée.

**Écart avec le code déployé.** Quatre passages de documentation de `pipeline/pose.py`
(dans les docstrings de `n_persons_median`, `select_subjects_auto`, `cache_path_for` et
`detect_smart_rotation`) nommaient un créateur de vidéos et un corpus de référence de la
recherche ; ils sont remplacés ici par des mots génériques. Aucune instruction n'a changé :
l'arbre syntaxique du fichier, docstrings mises à part, est identique à celui qui tourne.
Les autres fichiers de `pipeline/` et `scripts/`, et `sitecustomize.py`, sont identiques à
ceux qui tournent, octet pour octet (empreintes dans `SHA256SUMS`). Le `Dockerfile` et les deux
listes de dépendances sont tirés de ceux de la production, réduits à cette voie.

Les renvois des commentaires vers `adr/`, `notebooks/` ou `memory/` désignent des
documents de recherche qui ne sont pas publiés.

## Construire et lancer

Avec Docker (essayé en arm64 ; rien n'y est propre à une architecture, torch vient de
l'index « CPU » de PyTorch, qui publie amd64 et arm64) :

```sh
docker build -t recodanse-masques .
docker run --rm --network none -v "$PWD/videos:/videos" recodanse-masques \
  /videos/cours.mp4 --out /videos/sortie
```

L'image tourne sans réseau : les modèles y sont embarqués et `YOLO_OFFLINE=1` empêche
ultralytics d'en chercher un en ligne. Le résultat arrive dans
`videos/sortie/cours.masques.kadans.json` ; le dossier monté doit être accessible en
écriture à l'utilisateur `10001` du conteneur.

Sans Docker, avec Python 3.11 :

```sh
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-torch.txt
pip install --no-deps -r requirements.lock.txt && pip check
# les modèles : voir le tableau ci-dessous (yolov8n-pose.pt à la racine du dépôt,
# les deux .onnx dans $TORCH_HOME/hub/checkpoints)
YOLO_OFFLINE=1 python scripts/masques_for_file.py cours.mp4 --out sortie
```

Options : `--frame-decimation N` (une image sur N, 5 par défaut), `--device cpu`,
`--tracker botsort|bytetrack|none` (`botsort` par défaut ; sans suivi, l'identifiant
d'une personne change d'une image à l'autre).

## Les modèles

Aucun de ces modèles n'a été réentraîné par Arcodange. Le `Dockerfile` les télécharge
depuis leur source officielle et vérifie leur empreinte.

| Modèle | Licence | Source | SHA-256 |
| --- | --- | --- | --- |
| `yolov8n-pose.pt` (Ultralytics, version v8.3.0) | AGPL-3.0 | <https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt> | `c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212` |
| YOLOX-m HumanArt (OpenMMLab, détecteur de personnes de rtmlib) | Apache-2.0 | <https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip> | `3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23` (le `.onnx` extrait) |
| RTMW `rtmw-dw-x-l` 256×192 (OpenMMLab, pose du corps entier, 133 points) | Apache-2.0 | <https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip> | `9a9bc17c13ff0a1e37507f45a037a21bc410cd08374b0e664c57d0080843aa58` (le `.onnx` extrait) |

## Ce qui n'est pas ici

- **Aucune vidéo, aucun masque produit, aucune donnée de personne.** Ce dépôt ne contient
  que du code et des listes de versions.
- **Les autres programmes de RecoDanse** (l'application, son serveur, l'orchestration des
  analyses) : ils n'importent pas ultralytics et communiquent avec celui-ci par une ligne
  de commande et un fichier.
- **Le modèle qui dit « est-ce de la danse ? »** : c'est un modèle d'Arcodange, entraîné
  sur des vidéos du fondateur à partir de MobileNetV2 (timm, Apache-2.0, pré-entraîné sur
  ImageNet). Il n'utilise pas ultralytics.

## Licence

© 2026 Arcodange. Ce programme est un logiciel libre : vous pouvez le redistribuer et le
modifier selon les termes de la GNU Affero General Public License telle que publiée par la
Free Software Foundation, version 3. Il est distribué dans l'espoir qu'il sera utile, mais
**sans aucune garantie** ; voir [`LICENSE`](LICENSE).

ultralytics est © Ultralytics, sous AGPL-3.0. rtmlib, onnxruntime, OpenCV, torch et les
autres dépendances gardent leurs propres licences, installées avec chaque paquet.
