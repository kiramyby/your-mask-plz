#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP_PATH="${SCRIPT_DIR}/face-mask-dataset-yolo-format.zip"
DEST_DIR="${SCRIPT_DIR}"

mkdir -p "${DEST_DIR}"

curl -L -x http://127.0.0.1:7897 -o "${ZIP_PATH}" \
  "https://www.kaggle.com/api/v1/datasets/download/aditya276/face-mask-dataset-yolo-format"
unzip -o "${ZIP_PATH}" -d "${DEST_DIR}"

# Ensure labels are in Ultralytics-expected layout
DATA_DIR="${SCRIPT_DIR}/dataset"
for split in train valid test; do
  mkdir -p "${DATA_DIR}/labels/${split}"
  if compgen -G "${DATA_DIR}/images/${split}/*.txt" > /dev/null; then
    find "${DATA_DIR}/images/${split}" -type f -name '*.txt' -print0 \
      | xargs -0 -I{} cp -n "{}" "${DATA_DIR}/labels/${split}/"
  fi
done

rm -f "${ZIP_PATH}"