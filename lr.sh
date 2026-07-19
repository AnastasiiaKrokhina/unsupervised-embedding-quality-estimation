#!/usr/bin/env bash
#lr      0.1   0.2   0.3   0.4   0.5
#LR_TAG  0.1   0.2   0.3   0.4   0.5
set -euo pipefail

GPU_ID=3
LR=0.1
LR_TAG=0.1
WD=1e-6
WD_TAG=1e-6
COV=5
EPOCHS=100
SEED=0

RUN_NAME="vicreg-c100-rn18-lr${LR_TAG}-wd${WD_TAG}-cov${COV}-seed${SEED}-e${EPOCHS}"
CKPT_DIR="./trained_models_lr/${RUN_NAME}"
LOG_FILE="./logs/${RUN_NAME}.log"

mkdir -p "${CKPT_DIR}" ./logs

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python -u main_pretrain.py \
  --config-path scripts/pretrain/cifar \
  --config-name vicreg.yaml \
  name="${RUN_NAME}" \
  data.dataset=cifar100 \
  data.train_path=./datasets \
  data.val_path=./datasets \
  backbone.name=resnet18 \
  method_kwargs.proj_hidden_dim=2048 \
  method_kwargs.proj_output_dim=2048 \
  method_kwargs.sim_loss_weight=25 \
  method_kwargs.var_loss_weight=25 \
  method_kwargs.cov_loss_weight="${COV}" \
  optimizer.name=lars \
  optimizer.batch_size=256 \
  optimizer.lr="${LR}" \
  optimizer.weight_decay="${WD}" \
  optimizer.kwargs.eta=0.02 \
  optimizer.kwargs.clip_lr=True \
  optimizer.kwargs.exclude_bias_n_norm=True \
  scheduler.name=warmup_cosine \
  max_epochs="${EPOCHS}" \
  'devices=[0]' \
  strategy=auto \
  sync_batchnorm=False \
  precision=16-mixed \
  wandb.enabled=False \
  auto_resume.enabled=True \
  checkpoint.enabled=True \
  checkpoint.dir="${CKPT_DIR}" \
  checkpoint.frequency=1 \
  ++checkpoint.keep_prev=False \
  ++seed="${SEED}" \
  > "${LOG_FILE}" 2>&1 &

echo "Started PID: $!"
echo "Log: ${LOG_FILE}"