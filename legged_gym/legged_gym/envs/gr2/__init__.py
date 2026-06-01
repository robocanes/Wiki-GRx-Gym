from legged_gym.utils.task_registry import task_registry

# ------------------------------------------------------------

from legged_gym.envs.gr2.gr2_code import GR2
from legged_gym.envs.gr2.gr2_reach_code import GR2Reach
from legged_gym.envs.gr2.gr2_config_dynamic_walk import GR2DynamicWalkCfg, GR2DynamicWalkCfgPPO
from legged_gym.envs.gr2.gr2_config_main_body import GR2MainBodyCfg, GR2MainBodyCfgPPO
from legged_gym.envs.gr2.gr2_config_reach import GR2ReachCfg, GR2ReachCfgPPO
from legged_gym.envs.gr2.gr2_config_upper_body import GR2UpperBodyCfg, GR2UpperBodyCfgPPO

task_registry.register("GR2", GR2, GR2MainBodyCfg(), GR2MainBodyCfgPPO(), )
task_registry.register("GR2DynamicWalk", GR2, GR2DynamicWalkCfg(), GR2DynamicWalkCfgPPO(), )
task_registry.register("GR2UpperBody", GR2, GR2UpperBodyCfg(), GR2UpperBodyCfgPPO(), )
task_registry.register("GR2Reach", GR2Reach, GR2ReachCfg(), GR2ReachCfgPPO(), )
