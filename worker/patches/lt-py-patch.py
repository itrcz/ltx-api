"""Patch ComfyUI's LTXAV text-encoder memory estimator.

LTXVGemmaEnhancePrompt calls clip.load_model() before any tokenize(), which
makes memory_estimation_function receive {"gemma3_12b": []} → min([]) ValueError.
We insert an early-return for the empty-input case.

Run at image build time as: `python3 /lt-py-patch.py`.
Validated against ComfyUI master at:
    /comfyui/comfy/text_encoders/lt.py
"""
import sys

P = "/comfyui/comfy/text_encoders/lt.py"
OLD = (
    '        token_weight_pairs = token_weight_pairs.get("gemma3_12b", [])\n'
    '        m = min([sum(1 for _ in itertools.takewhile(lambda x: x[0] == 0, sub)) for sub in token_weight_pairs])'
)
NEW = (
    '        token_weight_pairs = token_weight_pairs.get("gemma3_12b", [])\n'
    '        if not token_weight_pairs:\n'
    '            return 642 * constant * 1024 * 1024\n'
    '        m = min([sum(1 for _ in itertools.takewhile(lambda x: x[0] == 0, sub)) for sub in token_weight_pairs])'
)

s = open(P).read()
if NEW.split("\n")[1] in s:
    print("lt.py already patched")
    sys.exit(0)
if OLD not in s:
    sys.exit("lt.py patch target not found — upstream changed?")
open(P, "w").write(s.replace(OLD, NEW))
print("lt.py patched")
