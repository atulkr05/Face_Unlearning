#!/bin/bash
# run_experiment.sh — End-to-end face identity unlearning experiment runner
# 
# Usage:
#   bash face_unlearning/run_experiment.sh [SPLIT_NAME]
#
# Examples:
#   bash face_unlearning/run_experiment.sh "Face Set 1"
#   bash face_unlearning/run_experiment.sh "Face Set 2"
#   bash face_unlearning/run_experiment.sh all     ← runs both sets

set -e

PYTHON="/DATA2/Atul/2027/challenge/face_env/bin/python"
SCRIPT_DIR="/DATA2/Atul/2027/challenge/face_unlearning"
CHALLENGE_DIR="/DATA2/Atul/2027/challenge"

DATA_DIR="$CHALLENGE_DIR/CelebAHQ/Img/img_celeba"
IDENTITY_FILE="$CHALLENGE_DIR/CelebAHQ/Anno/identity_CelebA.txt"
SPLITS_FILE="$SCRIPT_DIR/validation-splits.json"
EMBEDDING_CACHE="$SCRIPT_DIR/embeddings"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"

SPLIT_NAME="${1:-Face Set 1}"
GPU_ID="${2:-0}"

# ── Pre-flight checks ────────────────────────────────────────────────────────
echo "============================================"
echo " Face Identity Unlearning — Experiment Runner"
echo "============================================"
echo ""

if [ ! -f "$IDENTITY_FILE" ]; then
    echo "[ERROR] identity_CelebA.txt not found at: $IDENTITY_FILE"
    echo ""
    echo "Download instructions:"
    echo "  Option A (gdown): gdown https://drive.google.com/uc?id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS -O '$IDENTITY_FILE'"
    echo "  Option B: Visit https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html → Anno → identity_CelebA.txt"
    echo ""
    echo "Attempting auto-download with gdown..."
    $PYTHON -m pip install gdown -q
    $PYTHON -m gdown "https://drive.google.com/uc?id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS" \
        -O "$IDENTITY_FILE" || {
        echo "[ERROR] Auto-download failed. Please download manually."
        exit 1
    }
fi

echo "[OK] identity_CelebA.txt found at $IDENTITY_FILE"
echo ""

# ── Check GPU ────────────────────────────────────────────────────────────────
$PYTHON -c "
import torch
print(f'GPUs available: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory // 1024**3
    print(f'  GPU {i}: {name} ({mem} GB)')
"
echo ""

# ── Run for one or both face sets ────────────────────────────────────────────
run_split() {
    local SPLIT="$1"
    local GPU="${2:-0}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running: $SPLIT (GPU $GPU)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SCRIPT_DIR/run_unlearning.py" \
        --split_name "$SPLIT" \
        --img_dir "$DATA_DIR" \
        --identity_file "$IDENTITY_FILE" \
        --splits_file "$SPLITS_FILE" \
        --embedding_cache "$EMBEDDING_CACHE" \
        --ip_adapter_path "$CHALLENGE_DIR/face_unlearning/checkpoints/ip_adapter_faceid/ip-adapter-faceid_sd15.bin" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --epochs 20 \
        --lr 5e-5 \
        --batch_size 2 \
        --lora_rank 16 \
        --alpha 1.0 \
        --beta 5.0 \
        --delta 0.01 \
        --log_every 5 \
        --gpu 0

    echo ""
    echo "[Done] $SPLIT complete."
}

if [ "$SPLIT_NAME" = "all" ]; then
    echo "Running BOTH Face Set 1 and Face Set 2..."
    # Run Set 1 on GPU 0, Set 2 on GPU 1 in parallel
    run_split "Face Set 1" 0 &
    PID1=$!
    run_split "Face Set 2" 1 &
    PID2=$!
    wait $PID1
    wait $PID2
    echo ""
    echo "Both face sets complete!"
else
    run_split "$SPLIT_NAME" "$GPU_ID"
fi

# ── Collect all results ───────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "           Combined Results Summary"
echo "============================================"
$PYTHON - <<'PYEOF'
import json, glob, os

results_files = glob.glob("/DATA2/Atul/2027/challenge/face_unlearning/checkpoints/*/results_summary.json")
if not results_files:
    print("No results found yet.")
else:
    print(f"{'Set':<15} {'FA':>6} {'EA':>6} {'RA':>6} {'ERB':>8} {'GP':>6} {'AR':>6}")
    print("-" * 55)
    for rf in sorted(results_files):
        with open(rf) as f:
            r = json.load(f)
        set_name = os.path.basename(os.path.dirname(rf)).replace("_", " ")
        print(f"{set_name:<15} {r['FA']:>6.4f} {r['EA']:>6.4f} {r['RA']:>6.4f} {r['ERB']:>8.4f} {r['GP']:>6.4f} {r['AR']:>6.4f}")
PYEOF

echo ""
echo "Experiment complete. Checkpoints at: $CHECKPOINT_DIR"
