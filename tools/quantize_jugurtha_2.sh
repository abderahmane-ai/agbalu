#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/jovyan/envs/agbalu/bin/python"
LLAMA_DIR="/home/jovyan/llama.cpp"
LLAMA_BIN="/home/jovyan/llama-bin/llama-b11718"
MODEL_DIR="/home/jovyan/agbalu/artifacts/runs/jugurtha-9b-sft/model"
OUT_DIR="/home/jovyan/agbalu/artifacts/runs/jugurtha-2-9b-gguf"
BF16_GGUF="${OUT_DIR}/Jugurtha-2-9B-BF16.gguf"

export LD_LIBRARY_PATH="/home/jovyan/envs/agbalu/lib:/home/jovyan/llama-bin/llama-b11718:${LD_LIBRARY_PATH:-}"

mkdir -p "${OUT_DIR}"

echo "=== [1/5] Converting HF safetensors to BF16 GGUF ==="
if [ ! -f "${BF16_GGUF}" ]; then
    "${PYTHON}" "${LLAMA_DIR}/convert_hf_to_gguf.py" "${MODEL_DIR}" \
        --outfile "${BF16_GGUF}" \
        --outtype bf16
    echo "BF16 GGUF created: $(ls -lh "${BF16_GGUF}")"
else
    echo "Found existing ${BF16_GGUF}, skipping conversion."
fi

QUANTS=("Q4_K_M" "Q5_K_M" "Q6_K" "Q8_0" "Q3_K_M")

for q in "${QUANTS[@]}"; do
    target="${OUT_DIR}/Jugurtha-2-9B-${q}.gguf"
    echo "=== Quantizing to ${q} -> ${target} ==="
    if [ ! -f "${target}" ]; then
        "${LLAMA_BIN}/llama-quantize" "${BF16_GGUF}" "${target}" "${q}" 8
        echo "Finished ${q}: $(ls -lh "${target}")"
    else
        echo "Found existing ${target}, skipping."
    fi
done

echo "=== Cleaning intermediate BF16 GGUF ==="
rm -f "${BF16_GGUF}"

echo "=== Placing metadata and Modelfile ==="
cp /home/jovyan/agbalu/resources/llm/Modelfile "${OUT_DIR}/Modelfile"

cat << 'EOF' > "${OUT_DIR}/.gitattributes"
*.gguf filter=lfs diff=lfs merge=lfs -text
EOF

echo "=== All quants completed successfully! ==="
ls -lh "${OUT_DIR}"
