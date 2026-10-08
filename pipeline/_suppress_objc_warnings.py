"""Filtre stderr pour supprimer les warnings objc cv2/av libavdevice (macOS).

Contexte : `cv2` 4.13.0 bundle `libavdevice.61.3.100.dylib` et `av` 17.x bundle
`libavdevice.62.1.100.dylib`. Les deux déclarent les classes Objective-C
`AVFFrameReceiver` et `AVFAudioReceiver`. Le runtime macOS imprime un warning :

    objc[NNNN]: Class AVFFrameReceiver is implemented in both .../cv2/...
    .../av/.... This may cause spurious casting failures and mysterious
    crashes. One of the duplicates must be removed or renamed.

**Le warning est cosmétique** : le runtime garde la 1re classe chargée et
discarde la 2e. Aucun crash observé sur 600+ runs pytest + scripts pipeline.
Mais il pollue les logs uvicorn et les sorties des scripts CLI.

cv2 hard-link `libavdevice.61` (renommer le .dylib casse `import cv2`).
On ne peut donc pas éliminer le warning à la source — on filtre stderr.

## Comment ça marche

À l'import de ce module (qui DOIT être fait AVANT `import cv2` ou `import av`)
on :
1. Sauvegarde fd 2 (vrai stderr)
2. Crée un pipe
3. Redirige fd 2 vers l'écriture du pipe via `os.dup2`
4. Lance un thread daemon qui lit la lecture du pipe, filtre les lignes
   `objc[...]: Class X is implemented in both ...`, et passe le reste au
   vrai stderr (le fd sauvegardé).

Le filtre est strict : seules les lignes commençant par `objc[<digits>]:` ET
contenant `implemented in both` OU `must be removed or renamed` sont
supprimées. Tout autre output sur stderr (erreurs Python, warnings tier)
passe normalement.

## Usage

    # ✅ Bon : import EN PREMIER dans les entry points qui touchent cv2/av
    from pipeline import _suppress_objc_warnings  # noqa: F401
    import cv2
    import av

Pas besoin sur Linux/Windows : l'import est no-op.
"""
from __future__ import annotations

import atexit
import os
import sys
import threading


_OBJC_PATTERNS = (
    b"implemented in both",
    b"must be removed or renamed",
)


def _filter_stderr_loop(read_fd: int, orig_stderr_fd: int) -> None:
    """Thread daemon : read from pipe, filter, forward to orig stderr."""
    try:
        with os.fdopen(read_fd, "rb", buffering=0) as pipe:
            with os.fdopen(orig_stderr_fd, "wb", buffering=0, closefd=False) as orig:
                while True:
                    line = pipe.readline()
                    if not line:
                        break
                    # Drop objc duplicate-class warnings only
                    if line.startswith(b"objc[") and any(
                        pat in line for pat in _OBJC_PATTERNS
                    ):
                        continue
                    orig.write(line)
    except Exception:  # noqa: BLE001
        # Ne JAMAIS crash le pipeline pour un filtre stderr
        pass


def _install() -> None:
    if sys.platform != "darwin":
        return
    if os.environ.get("VIDEO_ANALYSIS_NO_STDERR_FILTER") == "1":
        return  # escape hatch pour debug
    if getattr(_install, "_done", False):
        return  # idempotent

    orig_stderr_fd = os.dup(2)
    r, w = os.pipe()
    os.dup2(w, 2)
    os.close(w)

    t = threading.Thread(
        target=_filter_stderr_loop,
        args=(r, orig_stderr_fd),
        daemon=True,
        name="objc-stderr-filter",
    )
    t.start()
    atexit.register(_vider_avant_la_sortie, orig_stderr_fd, t)
    _install._done = True  # type: ignore[attr-defined]


def _vider_avant_la_sortie(orig_stderr_fd: int, fil: threading.Thread) -> None:
    """Rend fd 2 au vrai stderr et laisse le fil finir de TRANSMETTRE.

    ⚠ Mesuré le 2026-09-14 : sans ceci, tout ce qu'un script écrit sur stderr
    juste avant `sys.exit` DISPARAÎT — le fil est un démon, tué à la sortie avant
    d'avoir relayé le contenu du pipe. `measure_dance_gate_baseline.py` sortait
    en code 2 avec ZÉRO octet sur ses deux flux, et le « ⛔ POINT D'ARRÊT n° 2 »
    de `train_dance_gate.py`, écrit sur stderr, aurait disparu de la même façon :
    un refus qui ne se lit pas.

    Restaurer fd 2 ferme la seule extrémité d'écriture du pipe : le fil lit la
    fin de fichier, vide ce qui restait, et sort.
    """
    try:
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    try:
        os.dup2(orig_stderr_fd, 2)
    except OSError:
        return
    fil.join(timeout=2.0)


_install()
