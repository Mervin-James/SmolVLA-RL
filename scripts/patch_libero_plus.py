import argparse
from pathlib import Path

PATCHES = {
    "libero/libero/envs/env_wrapper.py": [
        ("dtype=np.float_)", "dtype=np.float64)"),
        ("np.fromstring(x.make_blob(), np.uint8)", "np.frombuffer(x.make_blob(), np.uint8)"),
        ("plasma_fractal(wibbledecay=c[1])",
         "plasma_fractal(mapsize=max(256, 1 << (max(height_x, weight_x) - 1).bit_length()), wibbledecay=c[1])"),
    ],
    "libero/libero/benchmark/__init__.py": [
        ('            language = " ".join(x.split("_"))\n',
         '            language = " ".join(re.sub(r"_(?:view|light|add|moved|table|tb)_.*(?=\\.bddl$)", "", x).split("_"))\n'),
    ],
}


def patch(clone: Path) -> None:
    for relative, replacements in PATCHES.items():
        path = clone / relative
        source = path.read_text()
        for old, new in replacements:
            if new in source:
                continue
            assert source.count(old) == 1, f"{old!r} not found exactly once in {path}"
            source = source.replace(old, new)
        path.write_text(source)
        print(f"patched {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clone", type=Path)
    args = parser.parse_args()
    patch(args.clone)


if __name__ == "__main__":
    main()
