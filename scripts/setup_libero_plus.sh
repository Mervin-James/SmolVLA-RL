set -eu
VENV="${VENV:-$HOME/venvs/vlaplus}"
CLONE="${CLONE:-$HOME/libero-plus}"
SHA="${SHA:-4976dc3}"
CONFIG_DIR="${CONFIG_DIR:-$HOME/.libero-plus}"
ROOT="$CLONE/libero/libero"

echo "== 1. venv with torch cu128"
[ -d "$VENV" ] || uv venv "$VENV" --python 3.12
uv pip install --python "$VENV" torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VENV" "lerobot[smolvla,dataset]==0.6.1" \
  "robosuite==1.4.0" "bddl==1.0.1" "easydict==1.13" "mujoco==3.8.1" \
  "Wand==0.6.13" "scikit-image==0.25.2" "gym==0.26.2" pyyaml future \
  matplotlib opencv-python einops cloudpickle thop hydra-core wandb

echo "== 2. clone LIBERO-plus @ $SHA"
[ -d "$CLONE" ] || git clone https://github.com/sylvestf/LIBERO-plus.git "$CLONE"
git -C "$CLONE" checkout "$SHA"
uv pip install --python "$VENV" --no-deps -e "$CLONE"
"$VENV/bin/python" "$(dirname "$0")/patch_libero_plus.py" "$CLONE"

echo "== 3. assets (6.4 GB)"
if [ ! -d "$ROOT/assets" ]; then
  "$VENV/bin/python" -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Sylvest/LIBERO-plus', repo_type='dataset', filename='assets.zip', local_dir='/tmp/libero-plus-dl')"
  unzip -q /tmp/libero-plus-dl/assets.zip -d /tmp/libero-plus-dl/extract
  mv "$(find /tmp/libero-plus-dl/extract -type d -name assets | head -1)" "$ROOT/assets"
  rm -rf /tmp/libero-plus-dl
fi

echo "== 4. isolated config"
mkdir -p "$CONFIG_DIR"
printf "assets: %s/assets\nbddl_files: %s/bddl_files\ndatasets: %s/../datasets\ninit_states: %s/init_files\n" "$ROOT" "$ROOT" "$ROOT" "$ROOT" > "$CONFIG_DIR/config.yaml"
cat "$CONFIG_DIR/config.yaml"

echo "== 5. check: suites and task counts"
PYTHONPATH="$CLONE" LIBERO_CONFIG_PATH="$CONFIG_DIR" "$VENV/bin/python" -c "
from libero.libero import benchmark
d = benchmark.get_benchmark_dict()
print('suites:', sorted(d))
for name in sorted(d):
    if 'goal' in name:
        suite = d[name]()
        print(name, 'tasks:', suite.n_tasks)
        print('example:', suite.get_task(0).language)
"
echo "== done. standard LIBERO in ~/venvs/vla is untouched"
