#!/usr/bin/env python3
"""
Runs inside build/venv-export on the Jetson, and as the ONNX export stage of docker/Dockerfile
on x86. ONNX is architecture-independent, which is why this step is the one part of the model
pipeline that can be baked into an image; the TensorRT engines cannot be.

Downloads the PPE and fire/smoke YOLO checkpoints, exports each to ONNX, and reports the two
facts that determine everything downstream:

  1. the class list  -> labels.txt + num-detected-classes, and which classes rules.py can use
  2. the output shape -> which bbox parser and cluster-mode the nvinfer config needs

    {N, 8400}  pre-NMS  (v8/v11)   -> custom NMS parser,   cluster-mode=2
    {300, 6}   post-NMS (v10/v26+) -> passthrough parser,  cluster-mode=4

Nothing here is guessed: the shape is read back off the exported ONNX. Guessing this wrong is
the single most common way a DeepStream YOLO integration silently produces zero or garbage boxes.

    ./build/venv-export/bin/python3 scripts/export_models.py [ppe|fire|all]
"""

import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMGSZ = 640
OPSET = 17
MAX_BATCH = 20
DOWNLOAD_ATTEMPTS = 4

MODELS = {
    "ppe": {
        "repo": "melihuzunoglu/ppe-detection",
        "file": "best.pt",
        "note": "YOLOv11 PPE — AGPL-3.0 (see the licensing section in README.md)",
        "min_bytes": 3_000_000,
    },
    "fire": {
        "repo": "SalahALHaismawi/yolov26-fire-detection",
        "file": "best.pt",
        "note": "YOLOv26-S fire/smoke — MIT",
        # Hub file is 20.3 MB. A truncated pull (seen as 9.6 MB) loads as a zip with no
        # central directory and Ultralytics reports a "corrupted checkpoint".
        "min_bytes": 15_000_000,
    },
}


def _looks_like_ckpt(path: Path, min_bytes: int) -> bool:
    """Torch checkpoints are zip files (`PK`). HTML, LFS pointers and truncated Xet pulls are not."""
    if not path.exists() or path.stat().st_size < min_bytes:
        return False
    with path.open("rb") as fh:
        return fh.read(2) == b"PK"


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _download_hf_hub(repo: str, fname: str, dst: Path, *, force: bool = False) -> None:
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id=repo, filename=fname, local_dir=str(dst.parent), force_download=force,
    )
    src = Path(cached)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


def _download_urllib(repo: str, fname: str, dst: Path, min_bytes: int) -> None:
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}?download=true"
    req = urllib.request.Request(url, headers={"User-Agent": "isms-export/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dst, "wb") as out:
        expected = resp.headers.get("Content-Length")
        shutil.copyfileobj(resp, out)
        wrote = out.tell()
    if expected and int(expected) != wrote:
        raise OSError(f"truncated download: got {wrote} bytes, Content-Length {expected}")
    if wrote < min_bytes:
        raise OSError(f"truncated download: got {wrote} bytes, need at least {min_bytes}")


def download(repo: str, fname: str, dst: Path, min_bytes: int) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _looks_like_ckpt(dst, min_bytes):
        print(f"    {dst.name} already present ({dst.stat().st_size/1e6:.1f} MB)")
        return dst
    _unlink(dst)

    last_err: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        print(f"    downloading {repo}/{fname} (attempt {attempt}/{DOWNLOAD_ATTEMPTS})")
        try:
            try:
                _download_hf_hub(repo, fname, dst, force=attempt > 1)
            except ImportError:
                _download_urllib(repo, fname, dst, min_bytes)
            if not _looks_like_ckpt(dst, min_bytes):
                size = dst.stat().st_size if dst.exists() else 0
                raise OSError(f"{dst.name} is not a torch checkpoint ({size} bytes)")
            print(f"    -> {dst} ({dst.stat().st_size/1e6:.1f} MB)")
            return dst
        except Exception as exc:  # noqa: BLE001 — retry the whole pull, then fail clearly
            last_err = exc
            print(f"    !! {exc}")
            _unlink(dst)
            sibling = dst.parent / fname
            if sibling != dst:
                _unlink(sibling)
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(2 * attempt)
    raise SystemExit(f"could not download {repo}/{fname}: {last_err}")


def export(name: str, spec: dict) -> dict:
    import torch
    from ultralytics import YOLO

    # oneDNN is disabled for the whole export, which costs nothing and prevents a hard crash.
    #
    # On an AMD EPYC 9554 (Zen 4) with torch 2.13.0+cpu, a plain `nn.Conv2d` over a
    # (20, 3, 640, 640) tensor dies with SIGFPE inside `_conv_forward` — no Python traceback, just
    # "Floating point exception (core dumped)" and exit 136. Reduced to four lines of torch with no
    # ultralytics involved, so it is a oneDNN convolution bug and not anything in this file. It
    # surfaced first in the FLOPs counter that `fuse()` calls, which made it look like a
    # bookkeeping problem; the real export hits the same kernel a moment later.
    #
    # This changes which convolution implementation runs, never what is traced: the ONNX graph is
    # a record of the operations, not of their numerics, so the exported file is byte-identical
    # either way. TensorRT does every real inference, and the reference kernels are quick enough
    # for two nano-scale models exported once at image build time.
    torch.backends.mkldnn.enabled = False

    print(f"\n=== {name} — {spec['repo']} ===")
    print(f"    {spec['note']}")
    outdir = ROOT / "models" / name / "model"
    pt = download(spec["repo"], spec["file"], outdir / f"{name}.pt", spec["min_bytes"])

    model = YOLO(str(pt))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    classes = [names[i] for i in sorted(names)]
    print(f"    {len(classes)} classes: {classes}")

    labels = outdir.parent / "config" / "labels.txt"
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text("\n".join(classes) + "\n")
    print(f"    labels -> {labels}")

    onnx_path = outdir / f"{name}.onnx"
    if not onnx_path.exists():
        print(f"    exporting ONNX (imgsz={IMGSZ}, opset={OPSET}, dynamic, batch={MAX_BATCH})")
        produced = model.export(
            format="onnx", imgsz=IMGSZ, opset=OPSET,
            dynamic=True, simplify=True, batch=MAX_BATCH,
        )
        Path(produced).replace(onnx_path)
    print(f"    onnx -> {onnx_path}")

    info = inspect(onnx_path, num_classes=len(classes))
    info.update(name=name, repo=spec["repo"], classes=classes, num_classes=len(classes),
                onnx=str(onnx_path), labels=str(labels))
    return info


def inspect(onnx_path: Path, num_classes: int | None = None) -> dict:
    """Read input/output shapes back off the exported graph and derive the parser choice.

    Ultralytics exports with `dynamic=True` leave the anchor axis symbolic (`'anchors'`), so the
    classification cannot rely on integer dims alone — it keys off the *channel* axis instead,
    which stays concrete.
    """
    import onnx

    m = onnx.load(str(onnx_path))

    def shape(vi):
        return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                for d in vi.type.tensor_type.shape.dim]

    inputs = {vi.name: shape(vi) for vi in m.graph.input}
    outputs = {vi.name: shape(vi) for vi in m.graph.output}
    print(f"    inputs : {inputs}")
    print(f"    outputs: {outputs}")

    oname, oshape = next(iter(outputs.items()))
    tail = oshape[1:]  # drop batch

    # Post-NMS graphs emit [max_det, 6]: (x1,y1,x2,y2,conf,cls) in pixel coords, already clustered.
    # Pre-NMS graphs emit [4+num_classes, anchors]: raw cx/cy/w/h + per-class scores.
    expected_ch = (4 + num_classes) if num_classes else None
    if len(tail) == 2 and tail[-1] == 6 and isinstance(tail[0], int) and tail[0] >= 100:
        parser, cluster, kind = "passthrough", 4, "post-NMS (v10/v26+ style)"
    elif len(tail) == 2 and expected_ch is not None and tail[0] == expected_ch:
        parser, cluster, kind = "custom-NMS", 2, f"pre-NMS (v8/v11 style, 4+{num_classes} channels)"
    elif len(tail) == 2 and isinstance(tail[0], int) and tail[0] < 100:
        parser, cluster, kind = "custom-NMS", 2, "pre-NMS (v8/v11 style, channel-major)"
    else:
        parser, cluster, kind = "UNKNOWN", None, f"unrecognised output shape {oshape}"

    print(f"    output '{oname}' {oshape} -> {kind}")
    print(f"    => parser={parser}  cluster-mode={cluster}")
    if parser == "UNKNOWN":
        print("    !! Could not classify this head. Inspect manually before writing the config.")

    dynamic = any(not isinstance(d, int) for d in next(iter(inputs.values())))
    if dynamic:
        print(f"    => dynamic input axes present, nvinfer config MUST set infer-dims=3;{IMGSZ};{IMGSZ}")

    return dict(output_name=oname, output_shape=oshape, input_shape=next(iter(inputs.values())),
                parser=parser, cluster_mode=cluster, head_kind=kind,
                needs_infer_dims=dynamic, imgsz=IMGSZ)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = MODELS if which == "all" else {which: MODELS[which]}

    results = {}
    for name, spec in targets.items():
        results[name] = export(name, spec)

    summary = ROOT / "models" / "export_summary.json"
    existing = json.loads(summary.read_text()) if summary.exists() else {}
    existing.update(results)
    summary.write_text(json.dumps(existing, indent=2))

    print("\n" + "=" * 62)
    print(" SUMMARY — these values go straight into the nvinfer configs")
    print("=" * 62)
    for n, r in existing.items():
        print(f" {n:5s} classes={r['num_classes']:<3} cluster-mode={r['cluster_mode']} "
              f"parser={r['parser']}")
        print(f"       {r['classes']}")
    print(f"\n -> {summary}")


if __name__ == "__main__":
    main()
