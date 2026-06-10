from legged_gym.utils.task_registry import task_registry

# ------------------------------------------------------------

from legged_gym.envs.gr2.gr2_code import GR2, GR2DynamicWalk12
from legged_gym.envs.gr2.gr2_reach_code import GR2DynamicWalkReachPretrain, GR2Reach
from legged_gym.envs.gr2.gr2_config_dynamic_walk import GR2DynamicWalkCfg, GR2DynamicWalkCfgPPO
from legged_gym.envs.gr2.gr2_config_dynamic_walk_12 import GR2DynamicWalk12Cfg, GR2DynamicWalk12CfgPPO
from legged_gym.envs.gr2.gr2_config_dynamic_walk_reach_pretrain import (
    GR2DynamicWalkReachPretrainCfg,
    GR2DynamicWalkReachPretrainCfgPPO,
)
from legged_gym.envs.gr2.gr2_config_main_body import GR2MainBodyCfg, GR2MainBodyCfgPPO
from legged_gym.envs.gr2.gr2_config_reach import GR2ReachCfg, GR2ReachCfgPPO
from legged_gym.envs.gr2.gr2_config_upper_body import GR2UpperBodyCfg, GR2UpperBodyCfgPPO

task_registry.register("GR2", GR2, GR2MainBodyCfg(), GR2MainBodyCfgPPO(), )
task_registry.register("GR2DynamicWalk", GR2, GR2DynamicWalkCfg(), GR2DynamicWalkCfgPPO(), )
task_registry.register("GR2DynamicWalk12", GR2DynamicWalk12, GR2DynamicWalk12Cfg(), GR2DynamicWalk12CfgPPO(), )
task_registry.register(
    "GR2DynamicWalkReachPretrain",
    GR2DynamicWalkReachPretrain,
    GR2DynamicWalkReachPretrainCfg(),
    GR2DynamicWalkReachPretrainCfgPPO(),
)
task_registry.register("GR2UpperBody", GR2, GR2UpperBodyCfg(), GR2UpperBodyCfgPPO(), )
task_registry.register("GR2Reach", GR2Reach, GR2ReachCfg(), GR2ReachCfgPPO(), )
