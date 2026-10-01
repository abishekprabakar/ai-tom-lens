#!/usr/bin/env bash
# Run the full analysis on three real open-weight LLMs.
# Needs a machine that can reach huggingface.co (about 5 GB of downloads).
# On an Apple-silicon Mac it uses the GPU through MPS automatically.
#
#   bash scripts/run_real_llms.sh            # ~150 story groups = 2,250 prompts per model
#   MAX_GROUPS=400 bash scripts/run_real_llms.sh   # the full corpus
set -euo pipefail
cd "$(dirname "$0")/.."

MAX_GROUPS="${MAX_GROUPS:-150}"
MODELS=(
  "Qwen/Qwen3-0.6B"                              # Qwen3, hybrid thinking model
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"    # R1 reasoning traces distilled into Qwen2.5
  "HuggingFaceTB/SmolLM2-360M-Instruct"          # Llama architecture, ungated
)
# Gated alternatives (accept the licence on huggingface.co, then `huggingface-cli login`):
#   meta-llama/Llama-3.2-1B-Instruct   google/gemma-3-1b-it

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

python -m tomlens.run --hf "${MODELS[@]}" --max-groups "$MAX_GROUPS" ${DTYPE:+--dtype "$DTYPE"}
python -m tomlens.report
echo
echo "Done. Start the viewer with:  python -m webapp.app   (then open http://127.0.0.1:5000)"
