#!/bin/bash
# One on-policy GKD round: rollout -> teacher-score -> KL train -> probe eval.
# Usage: bash experiments/gkd/run_full.sh <round_number> <init_checkpoint>
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
ROUND=$1
INIT=$2
PY="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
NCCL_ENV="NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_SOCKET_IFNAME=lo"
RD=outputs/gkd/round$ROUND
mkdir -p $RD

echo "=== GKD round $ROUND from $INIT ==="

echo "--- phase A: rollouts (9 GPUs) ---"
for s in 0 1 2 3 4 5 6 7 8; do
  CUDA_VISIBLE_DEVICES=$s "$PY" training/programs/gkd_rollout.py \
    --checkpoint $INIT --out $RD/rollouts --seed $ROUND \
    --shard-id $s --num-shards 9 > $RD/rollout_shard$s.log 2>&1 &
done
wait
echo "rollouts done: $(cat $RD/rollouts/rollouts_shard*.jsonl | wc -l) traces"

echo "--- phase B: teacher scoring (9 GPUs) ---"
for s in 0 1 2 3 4 5 6 7 8; do
  CUDA_VISIBLE_DEVICES=$s "$PY" training/programs/gkd_score.py \
    --rollouts $RD/rollouts/rollouts_shard$s.jsonl --out $RD/scored \
    --shard-id $s > $RD/score_shard$s.log 2>&1 &
done
wait
echo "scoring done"

echo "--- phase C: KL training (8 GPUs) ---"
env $NCCL_ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 \
  "$TORCHRUN" --standalone --nproc_per_node=8 training/programs/gkd_train.py \
  --init $INIT --scored "$RD/scored/scored_shard*.pt" \
  --out-dir $RD/model > $RD/train.log 2>&1
echo "training done -> $RD/model/final"

echo "--- phase D: probe eval (9 GPUs, validation[0:100]) ---"
for s in 0 1 2 3 4 5 6 7 8; do
  CUDA_VISIBLE_DEVICES=$s "$PY" training/validation/eval_latent_reader.py \
    --checkpoint $RD/model/final --max-samples 100 \
    --shard-id $s --num-shards 9 --tag gkd_r${ROUND}_probe \
    > $RD/probe_shard$s.log 2>&1 &
done
wait
"$PY" - <<EOF
import glob, json
n = c = 0
for p in glob.glob("data/latent_reader_eval/gkd_r${ROUND}_probe_shard*.json"):
    d = json.load(open(p)); n += d["n"]; c += d["correct"]
print(json.dumps({"round": $ROUND, "probe_n": n, "probe_acc": c / max(1, n)}))
EOF
echo "=== round $ROUND complete ==="
