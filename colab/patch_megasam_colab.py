import re
from pathlib import Path


ROOT = Path("SLAM/mega-sam/base")
EXTENSIONS = {".cu", ".cpp", ".h", ".hpp"}

REPLACEMENTS = [
    (re.compile(r"\.type\s*\(\s*\)\s*\.scalarType\s*\(\s*\)"), ".scalar_type()"),
    (re.compile(r"\.type\s*\(\s*\)\s*\.device\s*\(\s*\)"), ".device()"),
    (re.compile(r"\.type\s*\(\s*\)\s*\.is_cuda\s*\(\s*\)"), ".is_cuda()"),
    (re.compile(r"\.scalar_type\s*\(\s*\)\s*\.scalarType\s*\(\s*\)"), ".scalar_type()"),
    (re.compile(r"\.scalar_type\s*\(\s*\)\s*\.device\s*\(\s*\)"), ".device()"),
    (re.compile(r"\.scalar_type\s*\(\s*\)\s*\.is_cuda\s*\(\s*\)"), ".is_cuda()"),
]

BAD_PATTERNS = [
    re.compile(r"\.scalar_type\s*\(\s*\)\s*\.device\s*\("),
    re.compile(r"\.scalar_type\s*\(\s*\)\s*\.is_cuda\s*\("),
    re.compile(r"\.scalar_type\s*\(\s*\)\s*\.scalarType\s*\("),
]


def patch_text(text: str) -> str:
    patched = text
    for pattern, replacement in REPLACEMENTS:
        patched = pattern.sub(replacement, patched)
    return patched


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"Missing Mega-SAM base directory: {ROOT}")

    changed = []
    for path in ROOT.rglob("*"):
        if path.suffix not in EXTENSIONS or not path.is_file():
            continue

        text = path.read_text(errors="ignore")
        patched = patch_text(text)
        if patched != text:
            path.write_text(patched)
            changed.append(str(path))

    if changed:
        print("Patched Mega-SAM Torch API compatibility in:")
        for path in changed:
            print(f"  {path}")
    else:
        print("Mega-SAM Torch API compatibility patch: no source changes needed.")

    bad_hits = []
    for path in ROOT.rglob("*"):
        if path.suffix not in EXTENSIONS or not path.is_file():
            continue

        text = path.read_text(errors="ignore")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in BAD_PATTERNS):
                bad_hits.append(f"{path}:{line_number}: {line.strip()}")

    if bad_hits:
        print("ERROR: bad scalar_type chained access remains:")
        for hit in bad_hits[:40]:
            print(hit)
        raise SystemExit(1)

    print("Mega-SAM Torch API compatibility verification passed.")


if __name__ == "__main__":
    main()

