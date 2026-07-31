export A3_SHARED_PYTHON_ENV="/workspace/a3/python-env"
export PATH="/workspace/a3/bin:${A3_SHARED_PYTHON_ENV}/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export HF_HOME="/workspace/a3/cache/huggingface"
export TORCH_HOME="/workspace/a3/cache/torch"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset PYTHONNOUSERSITE

if [ -n "${USER:-}" ]; then
    export A3_STAGING_ROOT="/workspace/a3/staging/${USER}"
    export A3_RUN_ROOT="/workspace/a3/runs/${USER}"
fi
