#!/usr/bin/env bash
# =============================================================
# setup_data.sh — Download Kaggle dataset, train model, deploy weights
# =============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$ROOT/ml_pipeline"
BACKEND_WEIGHTS="$ROOT/backend/app/ml/weights"
DATA_DIR="$ML_DIR/data/chest_xray"
MODELS_DIR="$ML_DIR/models"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Step 1: Kaggle credentials ────────────────────────────────
setup_kaggle() {
  info "Checking Kaggle credentials..."
  if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo ""
    warn "Kaggle API key not found at ~/.kaggle/kaggle.json"
    echo "  1. Go to https://www.kaggle.com/settings/account"
    echo "  2. Scroll to 'API' section → 'Create New Token'"
    echo "  3. kaggle.json file download hobe"
    echo "  4. Run:"
    echo "     mkdir -p ~/.kaggle"
    echo "     cp ~/Downloads/kaggle.json ~/.kaggle/"
    echo "     chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo ""
    read -rp "kaggle.json ~/.kaggle/-এ রাখার পর Enter চাপো... " _
    if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
      error "~/.kaggle/kaggle.json এখনো নাই। আগে রাখো তারপর আবার run করো।"
    fi
  fi
  chmod 600 "$HOME/.kaggle/kaggle.json"
  info "Kaggle credentials OK"
}

# ── Step 2: Install kaggle CLI ────────────────────────────────
install_kaggle_cli() {
  if ! command -v kaggle &>/dev/null; then
    info "Installing kaggle CLI..."
    cd "$ROOT/backend" && source venv/bin/activate
    pip install kaggle -q
    deactivate
  fi
}

# ── Step 3: Download dataset ──────────────────────────────────
download_data() {
  if [ -d "$DATA_DIR/train" ] && [ "$(ls -A "$DATA_DIR/train")" ]; then
    warn "Data already exists at $DATA_DIR — skipping download"
    return
  fi

  info "Downloading chest-xray-pneumonia dataset (~2GB)..."
  mkdir -p "$ML_DIR/data"
  cd "$ML_DIR/data"

  source "$ROOT/backend/venv/bin/activate"
  kaggle datasets download -d paultimothymooney/chest-xray-pneumonia --unzip
  deactivate

  # Kaggle unzips to chest_xray/ inside data/
  if [ -d "$ML_DIR/data/chest_xray/chest_xray" ]; then
    mv "$ML_DIR/data/chest_xray/chest_xray"/* "$DATA_DIR/"
    rmdir "$ML_DIR/data/chest_xray/chest_xray"
  fi

  info "Dataset ready at $DATA_DIR"
}

# ── Step 4: Install ml_pipeline deps ─────────────────────────
install_ml_deps() {
  info "Installing ml_pipeline dependencies..."
  source "$ROOT/backend/venv/bin/activate"
  pip install -r "$ML_DIR/requirements.txt" -q
  deactivate
}

# ── Step 5: Train ─────────────────────────────────────────────
train_model() {
  info "Starting training (this may take 30-90 min on CPU)..."
  mkdir -p "$MODELS_DIR"

  source "$ROOT/backend/venv/bin/activate"
  cd "$ML_DIR"
  DATA_ROOT="$DATA_DIR" SAVE_DIR="$MODELS_DIR" python3 run_pipeline.py \
    --skip-eda \
    --epochs "${EPOCHS:-5}"
  deactivate

  info "Training complete. Model at $MODELS_DIR/best_model.pth"
}

# ── Step 6: Copy weights to backend ──────────────────────────
deploy_weights() {
  if [ ! -f "$MODELS_DIR/best_model.pth" ]; then
    error "best_model.pth not found. Training may have failed."
  fi

  mkdir -p "$BACKEND_WEIGHTS"
  cp "$MODELS_DIR/best_model.pth" "$BACKEND_WEIGHTS/best_model.pth"
  info "Weights deployed to $BACKEND_WEIGHTS/best_model.pth"
}

# ── Step 7: Install backend DB deps ──────────────────────────
install_backend_db_deps() {
  info "Installing backend DB dependencies (pymysql, sqlalchemy)..."
  source "$ROOT/backend/venv/bin/activate"
  pip install pymysql==1.1.1 sqlalchemy==2.0.30 cryptography==42.0.8 -q
  deactivate
}

# ── Main ──────────────────────────────────────────────────────
main() {
  echo ""
  echo "============================================"
  echo "  Pneumonia XAI — Data + Train Setup"
  echo "============================================"

  setup_kaggle
  install_kaggle_cli
  download_data
  install_ml_deps
  train_model
  deploy_weights
  install_backend_db_deps

  echo ""
  echo "============================================"
  echo "  DONE! Now run the app:"
  echo "    ./run.sh"
  echo "  Test at: http://localhost:8000/docs"
  echo "============================================"
}

main "$@"
