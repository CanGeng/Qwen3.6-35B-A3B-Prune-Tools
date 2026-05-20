from moe_prune_distill.prune.config_editor import write_student_config
from moe_prune_distill.prune.slicer import prune_state_dict_sharded

__all__ = ["write_student_config", "prune_state_dict_sharded"]
