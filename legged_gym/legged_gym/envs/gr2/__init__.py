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
from legged_gym.envs.gr2.gr2_config_reach_expanded_stable import (
    GR2ReachExpandedStableCfg,
    GR2ReachExpandedStableCfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_pickup_high import (
    GR2ReachPickupHighCfg,
    GR2ReachPickupHighCfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_pickup import (
    GR2ReachPlantedPickup,
    GR2ReachPlantedPickupCfg,
    GR2ReachPlantedPickupCfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum import (
    GR2ReachPlantedCurriculumZ1Cfg,
    GR2ReachPlantedCurriculumZ1CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z2 import (
    GR2ReachPlantedCurriculumZ2Cfg,
    GR2ReachPlantedCurriculumZ2CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z3 import (
    GR2ReachPlantedCurriculumZ3Cfg,
    GR2ReachPlantedCurriculumZ3CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z4 import (
    GR2ReachPlantedCurriculumZ4Cfg,
    GR2ReachPlantedCurriculumZ4CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z5 import (
    GR2ReachPlantedCurriculumZ5Cfg,
    GR2ReachPlantedCurriculumZ5CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z6 import (
    GR2ReachPlantedCurriculumZ6Cfg,
    GR2ReachPlantedCurriculumZ6CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z7 import (
    GR2ReachPlantedCurriculumZ7Cfg,
    GR2ReachPlantedCurriculumZ7CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z8 import (
    GR2ReachPlantedCurriculumZ8Cfg,
    GR2ReachPlantedCurriculumZ8CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z9 import (
    GR2ReachPlantedCurriculumZ9Cfg,
    GR2ReachPlantedCurriculumZ9CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z10 import (
    GR2ReachPlantedCurriculumZ10Cfg,
    GR2ReachPlantedCurriculumZ10CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z11 import (
    GR2ReachPlantedCurriculumZ11Cfg,
    GR2ReachPlantedCurriculumZ11CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z12 import (
    GR2ReachPlantedCurriculumZ12Cfg,
    GR2ReachPlantedCurriculumZ12CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z13 import (
    GR2ReachPlantedCurriculumZ13Cfg,
    GR2ReachPlantedCurriculumZ13CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z14 import (
    GR2ReachPlantedCurriculumZ14Cfg,
    GR2ReachPlantedCurriculumZ14CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z15 import (
    GR2ReachPlantedCurriculumZ15Cfg,
    GR2ReachPlantedCurriculumZ15CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z16 import (
    GR2ReachPlantedCurriculumZ16Cfg,
    GR2ReachPlantedCurriculumZ16CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z17 import (
    GR2ReachPlantedCurriculumZ17Cfg,
    GR2ReachPlantedCurriculumZ17CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z18 import (
    GR2ReachPlantedCurriculumZ18Cfg,
    GR2ReachPlantedCurriculumZ18CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z19 import (
    GR2ReachPlantedCurriculumZ19Cfg,
    GR2ReachPlantedCurriculumZ19CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z20 import (
    GR2ReachPlantedCurriculumZ20Cfg,
    GR2ReachPlantedCurriculumZ20CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z21 import (
    GR2ReachPlantedCurriculumZ21Cfg,
    GR2ReachPlantedCurriculumZ21CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z22 import (
    GR2ReachPlantedCurriculumZ22Cfg,
    GR2ReachPlantedCurriculumZ22CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z23 import (
    GR2ReachPlantedCurriculumZ23Cfg,
    GR2ReachPlantedCurriculumZ23CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z24 import (
    GR2ReachPlantedCurriculumZ24Cfg,
    GR2ReachPlantedCurriculumZ24CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z25 import (
    GR2ReachPlantedCurriculumZ25Cfg,
    GR2ReachPlantedCurriculumZ25CfgPPO,
)
from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z26 import (
    GR2ReachPlantedCurriculumZ26Cfg,
    GR2ReachPlantedCurriculumZ26CfgPPO,
)
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
task_registry.register("GR2ReachExpandedStable", GR2Reach, GR2ReachExpandedStableCfg(), GR2ReachExpandedStableCfgPPO(), )
task_registry.register("GR2ReachPickupHigh", GR2Reach, GR2ReachPickupHighCfg(), GR2ReachPickupHighCfgPPO(), )
task_registry.register("GR2ReachPlantedPickup", GR2ReachPlantedPickup, GR2ReachPlantedPickupCfg(), GR2ReachPlantedPickupCfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ1", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ1Cfg(), GR2ReachPlantedCurriculumZ1CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ2", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ2Cfg(), GR2ReachPlantedCurriculumZ2CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ3", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ3Cfg(), GR2ReachPlantedCurriculumZ3CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ4", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ4Cfg(), GR2ReachPlantedCurriculumZ4CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ5", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ5Cfg(), GR2ReachPlantedCurriculumZ5CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ6", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ6Cfg(), GR2ReachPlantedCurriculumZ6CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ7", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ7Cfg(), GR2ReachPlantedCurriculumZ7CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ8", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ8Cfg(), GR2ReachPlantedCurriculumZ8CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ9", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ9Cfg(), GR2ReachPlantedCurriculumZ9CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ10", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ10Cfg(), GR2ReachPlantedCurriculumZ10CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ11", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ11Cfg(), GR2ReachPlantedCurriculumZ11CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ12", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ12Cfg(), GR2ReachPlantedCurriculumZ12CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ13", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ13Cfg(), GR2ReachPlantedCurriculumZ13CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ14", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ14Cfg(), GR2ReachPlantedCurriculumZ14CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ15", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ15Cfg(), GR2ReachPlantedCurriculumZ15CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ16", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ16Cfg(), GR2ReachPlantedCurriculumZ16CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ17", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ17Cfg(), GR2ReachPlantedCurriculumZ17CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ18", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ18Cfg(), GR2ReachPlantedCurriculumZ18CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ19", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ19Cfg(), GR2ReachPlantedCurriculumZ19CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ20", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ20Cfg(), GR2ReachPlantedCurriculumZ20CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ21", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ21Cfg(), GR2ReachPlantedCurriculumZ21CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ22", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ22Cfg(), GR2ReachPlantedCurriculumZ22CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ23", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ23Cfg(), GR2ReachPlantedCurriculumZ23CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ24", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ24Cfg(), GR2ReachPlantedCurriculumZ24CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ25", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ25Cfg(), GR2ReachPlantedCurriculumZ25CfgPPO(), )
task_registry.register("GR2ReachPlantedCurriculumZ26", GR2ReachPlantedPickup, GR2ReachPlantedCurriculumZ26Cfg(), GR2ReachPlantedCurriculumZ26CfgPPO(), )
