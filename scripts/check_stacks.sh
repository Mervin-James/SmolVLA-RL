set -u
VLA="${VLA:-$HOME/venvs/vla}"
PLUS="${PLUS:-$HOME/venvs/vlaplus}"

report () {
  "$1/bin/python" - "$2" <<'PY'
import importlib.metadata as md
import sys

PACKAGES = ["lerobot", "robosuite", "mujoco", "numpy", "torch", "transformers", "hf-libero", "libero"]


def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "-"


print(f"{sys.argv[1]:8s} " + "  ".join(f"{n} {version(n)}" for n in PACKAGES))
PY
}

report "$VLA" "vla"
report "$PLUS" "vlaplus"
echo
echo "CP0 ran in vla, CP3 runs in vlaplus. A difference in robosuite, mujoco or the libero package means"
echo "the two evaluation distributions are not simulated by the same thing, which voids the comparison."
