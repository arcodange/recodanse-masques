"""analyses/sitecustomize.py — LES FILS DE CALCUL BORNÉS AU QUOTA DU JOB (K1, kadans-jobs#54).

Un `Job` d'analyse a 2 vCPU (ADR-0036 §13.6). Mais torch, OpenCV et onnxruntime
dimensionnent leurs fils sur les cœurs VISIBLES de la machine — tous, quota ou
pas — et rtmlib crée sa session onnxruntime sans options (rtmlib/tools/base.py).
Sous un quota CFS de 2 CPU, une douzaine de fils qui se disputent deux cœurs
coûtent bien plus que deux fils : mesuré au banc k3d le 2026-10-06, 26,6 min de
masques pour une vidéo de 49 s.

Python importe ce module au démarrage (il est sur sys.path) : rien à changer
dans le code R&D copié. Sans `KADANS_FILS`, il ne fait RIEN — le Mac garde ses
réglages.
"""
import os

_fils = os.environ.get("KADANS_FILS", "").strip()
if _fils.isdigit() and int(_fils) > 0:
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(_var, _fils)
    _n = int(_fils)

    try:  # onnxruntime : rtmlib n'expose pas les options de session — on les impose.
        import onnxruntime as _ort

        _Session = _ort.InferenceSession

        class _SessionBornee(_Session):
            def __init__(self, *args, **kwargs):
                opts = kwargs.get("sess_options")
                if opts is None:
                    opts = _ort.SessionOptions()
                    kwargs["sess_options"] = opts
                opts.intra_op_num_threads = _n
                opts.inter_op_num_threads = 1
                super().__init__(*args, **kwargs)

        _ort.InferenceSession = _SessionBornee
    except ImportError:
        pass

    try:  # OpenCV : son pool interne suit les cœurs visibles, lui aussi.
        import cv2 as _cv2

        _cv2.setNumThreads(_n)
    except ImportError:
        pass

    try:  # torch : OMP_NUM_THREADS suffit s'il est posé avant l'import ; on le confirme.
        import torch as _torch

        _torch.set_num_threads(_n)
    except ImportError:
        pass
