"""
Run this once to print the SAM3 processor API.
Tells us exactly what arguments are supported for prompts.
"""
import inspect
from Segmentation import SAM3Segmenter

seg = SAM3Segmenter()
seg._ensure_loaded()   # triggers model load
proc = seg._processor

print("\n=== set_image ===")
print(inspect.signature(proc.set_image))

print("\n=== reset_all_prompts ===")
print(inspect.signature(proc.reset_all_prompts))

print("\n=== set_text_prompt ===")
print(inspect.signature(proc.set_text_prompt))

print("\n=== add_geometric_prompt ===")
print(inspect.signature(proc.add_geometric_prompt))

print("\n=== all public methods ===")
for name in dir(proc):
    if not name.startswith("_"):
        print(f"  {name}")
