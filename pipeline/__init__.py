"""video_analysis pipeline.

Voir adr/ pour les décisions techniques (cross-link bidirectionnel doc/adr/notebooks/code).
Modules :
- audio_demix : Demucs htdemucs_ft → stems voix/musique
- beats       : madmom downbeat/tempo
- asr         : WhisperX word-level + diarization (voir adr/0003)
- pose        : RTMPose-WholeBody + tracking (voir adr/0001)
- segment     : V0 règles → V1 TCN (voir adr/0002)
- label       : comptes / patterns / styles
- crop        : crops dynamiques sur body parts (V1)
- export_decoupe : sérialisation JSON pour DanceVideos/decoupe/decoupe.sh
"""

# macOS : supprime les warnings objc cv2/av libavdevice (cosmétique mais
# bruyant). Importé AVANT tout `import cv2`/`import av` dans le sous-package.
# Cf. pipeline/_suppress_objc_warnings.py pour le détail. No-op hors macOS.
from . import _suppress_objc_warnings  # noqa: F401
