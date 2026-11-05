import json
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from megatron.core import dist_checkpointing
from safetensors import safe_open

root = Path(sys.argv[1])
step = int((root / "latest_checkpointed_iteration.txt").read_text())
folder = root / f"iter_{step:07d}"
meta = dcp.FileSystemReader(folder).read_metadata()
keys = [
    "model.visual.patch_embed.proj.weight",
    "model.visual.merger.linear_fc2.bias",
    "language_model.decoder.final_layernorm.weight",
]
mapping = {key: key for key in keys}
mapping[keys[-1]] = "model.language_model.norm.weight"
names = keys + ["optimizer.state." + p + "." + key for key in keys for p in ("param", "exp_avg", "exp_avg_sq")]
state = {
    key: torch.empty(meta.state_dict_metadata[key].size, dtype=meta.state_dict_metadata[key].properties.dtype)
    for key in names
}
dcp.load(state, checkpoint_id=folder)
report = {
    "iteration": step,
    "tensors": {},
    "rng_metadata_entries": sum(k.startswith("rng_state/") for k in meta.state_dict_metadata),
}
with safe_open(sys.argv[2], framework="pt", device="cpu") as hf:
    for key in keys:
        initial = hf.get_tensor(mapping[key]).to(state[key].dtype).float()
        current = state[key].float()
        master = state["optimizer.state.param." + key].float()
        moment = state["optimizer.state.exp_avg." + key].float()
        assert torch.isfinite(current).all() and torch.isfinite(master).all() and torch.isfinite(moment).all()
        report["tensors"][key] = {
            "elements": current.numel(),
            "model_l1_change_from_hf": (current - initial).abs().sum().item(),
            "model_changed_elements": int((current != initial).sum()),
            "master_l1_change_from_hf": (master - initial).abs().sum().item(),
            "adam_exp_avg_l1": moment.abs().sum().item(),
        }
common = dist_checkpointing.load_common_state_dict(folder)
report["scheduler"] = common["opt_param_scheduler"]
cursor = torch.load(root / "rollout" / f"global_dataset_state_dict_{step}.pt", map_location="cpu", weights_only=False)
report["dataset_cursor"] = {k: v for k, v in cursor.items() if isinstance(v, (int, float, bool, type(None)))}
report["optimizer_common_keys"] = list(common["optimizer"]["optimizer"])
Path(sys.argv[3]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
