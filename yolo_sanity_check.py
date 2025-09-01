import argparse, os, sys, json, hashlib, re
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass, field

try:
    import yaml
except Exception:
    yaml = None
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

@dataclass
class LabelIssue:
    kind: str
    detail: str = ""

@dataclass
class ImageReport:
    split: str
    image_path: str
    label_path: str | None
    issues: List[LabelIssue] = field(default_factory=list)
    objects: int = 0
    width: int = 0
    height: int = 0
    md5: str = ""

def md5_of_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def read_yaml_names(yaml_path: Path) -> List[str] | None:
    if not yaml_path.exists() or yaml is None:
        return None
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        # v8 sometimes stores {0: 'cls0', 1: 'cls1', ...}
        names = [names[i] for i in sorted(names.keys())]
    return names if isinstance(names, list) else None

def parse_label_line(line: str) -> Tuple[int, float, float, float, float] | None:
    parts = re.split(r"\s+", line.strip())
    if len(parts) != 5:
        return None
    try:
        cid = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
    except Exception:
        return None
    return cid, x, y, w, h

def clip_box(x, y, w, h):
    # convert to corners, clip, then back to center
    x1, y1 = x - w/2, y - h/2
    x2, y2 = x + w/2, y + h/2
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    w2, h2 = x2 - x1, y2 - y1
    if w2 <= 0 or h2 <= 0:
        return None
    xc, yc = (x1 + x2)/2, (y1 + y2)/2
    return xc, yc, w2, h2

def scan_split(root: Path, split: str, class_names: List[str] | None, fix: str | None, quarantine_dir: Path | None) -> List[ImageReport]:
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"

    img_map: Dict[str, Path] = {}
    if img_dir.exists():
        for p in img_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                img_map[p.stem] = p

    lbl_map: Dict[str, Path] = {}
    if lbl_dir.exists():
        for p in lbl_dir.rglob("*.txt"):
            lbl_map[p.stem] = p

    reports: List[ImageReport] = []

    # images with/without labels
    all_stems = set(img_map) | set(lbl_map)
    for stem in sorted(all_stems):
        img_p = img_map.get(stem)
        lbl_p = lbl_map.get(stem)
        rep = ImageReport(split=split, image_path=str(img_p) if img_p else "", label_path=str(lbl_p) if lbl_p else None)

        if img_p is None:
            rep.issues.append(LabelIssue("label_without_image"))
        else:
            try:
                with Image.open(img_p) as im:
                    w, h = im.size
                rep.width, rep.height = w, h
                rep.md5 = md5_of_file(img_p)
            except Exception as e:
                rep.issues.append(LabelIssue("corrupted_image", str(e)))

        if lbl_p is None:
            rep.issues.append(LabelIssue("image_without_label"))
        else:
            try:
                text = Path(lbl_p).read_text(encoding="utf-8").strip()
            except Exception as e:
                rep.issues.append(LabelIssue("label_read_error", str(e)))
                text = ""

            if text == "":
                # empty label file → غالبًا صورة بلا أجسام. مسموح لكنه يستحق المراجعة.
                rep.issues.append(LabelIssue("empty_label_file"))

            valid_lines = []
            new_lines = []
            for i, line in enumerate(text.splitlines(), 1):
                parsed = parse_label_line(line)
                if parsed is None:
                    rep.issues.append(LabelIssue("malformed_line", f"line {i}: {line[:40]}"))
                    continue
                cid, x, y, w, h = parsed
                if class_names is not None and not (0 <= cid < len(class_names)):
                    rep.issues.append(LabelIssue("bad_class_id", f"got {cid}"))
                    continue

                # numeric checks
                numeric_ok = all(map(lambda v: isinstance(v, float) and (v == v), [x,y,w,h]))
                if not numeric_ok:
                    rep.issues.append(LabelIssue("nan_or_non_numeric", f"line {i}"))
                    continue
                if w <= 0 or h <= 0:
                    rep.issues.append(LabelIssue("zero_area_box", f"line {i}"))
                    continue

                in01 = (0 <= x <= 1) and (0 <= y <= 1) and (0 <= w <= 1) and (0 <= h <= 1)
                if not in01 or (x - w/2 < 0) or (y - h/2 < 0) or (x + w/2 > 1) or (y + h/2 > 1):
                    rep.issues.append(LabelIssue("out_of_bounds", f"line {i}"))
                    if fix == "clip":
                        clipped = clip_box(x,y,w,h)
                        if clipped is None:
                            rep.issues.append(LabelIssue("box_dropped_after_clip", f"line {i}"))
                            continue
                        x,y,w,h = clipped
                        new_lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    else:
                        # keep as-is if not fixing
                        new_lines.append(line.strip())
                else:
                    new_lines.append(line.strip())

                valid_lines.append((cid,x,y,w,h))

            rep.objects = len(valid_lines)

            # write fixed labels if needed
            if fix == "clip" and lbl_p is not None:
                fixed_text = "\n".join(new_lines)
                if fixed_text.strip() != text.strip():
                    Path(lbl_p).write_text(fixed_text + ("\n" if fixed_text else ""), encoding="utf-8")

        # quarantine files that are fundamentally broken
        if quarantine_dir is not None:
            need_quarantine = any(iss.kind in {"corrupted_image"} for iss in rep.issues)
            if need_quarantine and img_p is not None:
                q_img = quarantine_dir / Path(rep.image_path).name
                q_img.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.replace(rep.image_path, q_img)
                    rep.issues.append(LabelIssue("quarantined_image", str(q_img)))
                except Exception as e:
                    rep.issues.append(LabelIssue("quarantine_failed", str(e)))

        reports.append(rep)

    return reports

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="dataset", help="dataset root containing train/valid/test")
    ap.add_argument("--fix", type=str, choices=["clip"], default=None, help="optional auto-fixes")
    ap.add_argument("--quarantine", type=str, default=None, help="quarantine folder (relative to root)")
    ap.add_argument("--out", type=str, default="sanity_report.json", help="output JSON report file")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERR] Root not found: {root}", file=sys.stderr); sys.exit(1)

    names = read_yaml_names(root / "data.yaml")
    if names is None:
        print("[WARN] Couldn't read class names from data.yaml. Class-id range checks will be skipped.")

    quarantine_dir = None
    if args.quarantine:
        quarantine_dir = (root / args.quarantine).resolve()
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    all_reports: List[ImageReport] = []
    for split in ["train", "valid", "test"]:
        rep = scan_split(root, split, names, args.fix, quarantine_dir)
        all_reports.extend(rep)

    # duplicate detection across splits
    md5_to_entries: Dict[str, List[ImageReport]] = {}
    for r in all_reports:
        if r.md5:
            md5_to_entries.setdefault(r.md5, []).append(r)
    leak_pairs = []
    for md5, entries in md5_to_entries.items():
        splits = {e.split for e in entries}
        if len(splits) > 1:
            leak_pairs.append([(e.split, e.image_path) for e in entries])

    # build summary
    summary = {}
    for split in ["train", "valid", "test"]:
        subset = [r for r in all_reports if r.split == split]
        imgs = sum(1 for r in subset if r.image_path)
        lbls = sum(1 for r in subset if r.label_path)
        total_objs = sum(r.objects for r in subset)
        issue_counts: Dict[str, int] = {}
        for r in subset:
            for iss in r.issues:
                issue_counts[iss.kind] = issue_counts.get(iss.kind, 0) + 1
        summary[split] = {
            "images": imgs,
            "labels": lbls,
            "objects": total_objs,
            "issues": dict(sorted(issue_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        }

    out_path = Path(args.out)
    payload = {
        "summary": summary,
        "leaks": leak_pairs,
        "reports": [
            {
                "split": r.split,
                "image_path": r.image_path,
                "label_path": r.label_path,
                "objects": r.objects,
                "width": r.width,
                "height": r.height,
                "issues": [{"kind": i.kind, "detail": i.detail} for i in r.issues],
            }
            for r in all_reports
        ],
        "class_names": names,
        "root": str(root.resolve()),
        "fix": args.fix,
        "quarantine": str(quarantine_dir) if quarantine_dir else None,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n✅ Sanity report saved to: {out_path}")
    print("📌 Quick summary:")
    for split, info in summary.items():
        print(f"  - {split}: images={info['images']} labels={info['labels']} objects={info['objects']}")
        top_issues = list(info["issues"].items())[:5]
        print(f"    issues(top): {top_issues if top_issues else 'none'}")

    if leak_pairs:
        print("\n⚠️ Potential data leakage (duplicate images across splits):")
        for group in leak_pairs[:10]:
            print("   -> " + " | ".join([f"{sp}:{Path(p).name}" for sp, p in group]))
        if len(leak_pairs) > 10:
            print(f"   (+{len(leak_pairs)-10} more)")

if __name__ == "__main__":
    main()
