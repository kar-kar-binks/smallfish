#!/usr/bin/env bash
set -e

DEST="$(dirname "$0")/simlingo_checkpoint"
mkdir -p "$DEST/model/.hydra"
mkdir -p "$DEST/model/checkpoints/epoch=013.ckpt"

curl -L -o "$DEST/model/.hydra/config.yaml" \
  "https://huggingface.co/RenzKa/simlingo/resolve/main/simlingo/.hydra/config.yaml"

curl -L -o "$DEST/model/checkpoints/epoch=013.ckpt/pytorch_model.pt" \
  "https://huggingface.co/RenzKa/simlingo/resolve/main/simlingo/checkpoints/epoch%3D013.ckpt/pytorch_model.pt"

echo "checkpoint fetched to $DEST"
