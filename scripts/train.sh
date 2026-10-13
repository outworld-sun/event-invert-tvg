#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

# Edit this section, as in Invert4TVG's grpo_3b.sh. Environment variables override it.
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-12371}"
PYTHON_BIN="${PYTHON_BIN:-python}"

BRANCH="${BRANCH:-E5}"
LOAD_MODEL_PATH="${LOAD_MODEL_PATH:-/data/models/Qwen2.5-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:-/data/timer1/annotations/train_2k5.json}"
VIDEO_FOLDER="${VIDEO_FOLDER:-/data/timer1/videos}"
DATASET_NAME="${DATASET_NAME:-timer1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/${DATASET_NAME}}"
SPACY_MODEL="${SPACY_MODEL:-en_core_web_sm}"
PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-}"
RUN_NAME="${RUN_NAME:-${BRANCH}_768_g8_seed3407}"
OUTDIR="${OUTDIR:-outputs/${RUN_NAME}}"
LOGGING_DIR="${LOGGING_DIR:-${OUTDIR}/tensorboard}"

if [[ "$BRANCH" == "E5" ]]; then
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
else
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
fi
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16384}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-false}"
PREPARE_DATA="${PREPARE_DATA:-auto}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero3_offload.json}"

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# One candidate per rank gives G8. ZeRO-3 offloads parameters and optimizer to CPU.
training_args=(
  --deepspeed "$DEEPSPEED_CONFIG"
  --branch "$BRANCH"
  --world_size "$NPROC_PER_NODE"
  --output_dir "$OUTDIR"
  --model_name_or_path "$LOAD_MODEL_PATH"
  --train_data_path "$DATA_PATH"
  --video_folder "$VIDEO_FOLDER"
  --dataset_name "$DATASET_NAME"
  --artifact_root "$ARTIFACT_ROOT"
  --spacy_model "$SPACY_MODEL"
  --prepare_data "$PREPARE_DATA"
  --max_prompt_length "$MAX_PROMPT_LENGTH"
  --max_completion_length "$MAX_COMPLETION_LENGTH"
  --num_generations "$NUM_GENERATIONS"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --logging_steps 1
  --bf16
  --torch_dtype bfloat16
  --seed 3407
  --data_seed 3417
  --gradient_checkpointing true
  --attn_implementation "$ATTN_IMPLEMENTATION"
  --fix_vit true
  --slide_window false
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --run_name "$RUN_NAME"
  --report_to tensorboard
  --reward_funcs iou format
  --temperature 1.0
  --prompt_type event_core_v1
  --is_curriculum_learning false
  --logging_dir "$LOGGING_DIR"
  --save_strategy "$SAVE_STRATEGY"
  --save_steps "$SAVE_STEPS"
  --save_total_limit 3
  --is_early_stopping false
  --save_only_model "$SAVE_ONLY_MODEL"
  --max_frames 768
  --min_frames 4
  --fps 2.0
  --min_pixels 12544
  --max_pixels 602112
  --total_pixels 2809856
  --visual_token_cap 9216
)
if [[ -n "$PARENT_CHECKPOINT" ]]; then
  training_args+=(--parent_checkpoint "$PARENT_CHECKPOINT")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  training_args+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

# Inspection and data preparation run once and do not initialize a GPU process group.
for argument in "$@"; do
  case "$argument" in
    --help|-h|--print_config|--dry_run|--dry-run|--prepare_only)
      exec "$PYTHON_BIN" main.py "${training_args[@]}" "$@"
      ;;
  esac
done

exec "$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  main.py "${training_args[@]}" "$@"
