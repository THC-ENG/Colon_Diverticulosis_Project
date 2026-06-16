import json
import os
from importlib import import_module, metadata
from pathlib import Path
from urllib.parse import unquote, urlparse


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _editable_project_root(dist_name: str) -> Path | None:
    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        return None
    text = dist.read_text("direct_url.json")
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    dir_info = payload.get("dir_info", {})
    if not bool(dir_info.get("editable")):
        return None
    url = str(payload.get("url", "")).strip()
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    raw_path = unquote(parsed.path or "")
    if os.name == "nt" and raw_path.startswith("/") and len(raw_path) >= 3 and raw_path[2] == ":":
        raw_path = raw_path[1:]
    if not raw_path:
        return None
    return Path(raw_path).resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def load_sam_components(caller: str = ""):
    medsam_ver = _dist_version("medsam")
    if medsam_ver is None:
        src = f" ({caller})" if caller else ""
        raise RuntimeError(
            "MedSAM runtime check failed%s: package `medsam` is not installed.\n"
            "Fix: python -m pip install -e \"D:\\Embodied AI\\SAM\\MedSAM\"" % src
        )

    plain_sam_ver = _dist_version("segment-anything")
    if plain_sam_ver is not None:
        src = f" ({caller})" if caller else ""
        raise RuntimeError(
            "MedSAM runtime check failed%s: detected conflicting package "
            "`segment-anything==%s`.\n"
            "Fix: python -m pip uninstall -y segment-anything" % (src, plain_sam_ver)
        )

    module = import_module("segment_anything")
    module_file = Path(str(getattr(module, "__file__", ""))).resolve()

    editable_root = _editable_project_root("medsam")
    if editable_root is not None and not _is_within(module_file, editable_root):
        src = f" ({caller})" if caller else ""
        raise RuntimeError(
            "MedSAM runtime check failed%s: imported `segment_anything` from `%s`, "
            "but editable MedSAM root is `%s`." % (src, module_file, editable_root)
        )

    predictor = getattr(module, "SamPredictor", None)
    registry = getattr(module, "sam_model_registry", None)
    if predictor is None or registry is None:
        src = f" ({caller})" if caller else ""
        raise RuntimeError(
            "MedSAM runtime check failed%s: `segment_anything` missing "
            "`SamPredictor` or `sam_model_registry`." % src
        )

    info = {
        "medsam_version": medsam_ver,
        "segment_anything_file": str(module_file),
        "editable_root": str(editable_root) if editable_root else "",
    }
    return predictor, registry, info
