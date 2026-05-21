from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import ReachTarget

print("Creating observation config...")
obs_config = ObservationConfig()
obs_config.set_all(False)

print("Creating action mode...")
action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(),
    gripper_action_mode=Discrete()
)

print("Creating env...")
env = Environment(
    action_mode=action_mode,
    obs_config=obs_config,
    headless=True
)

print("Launching env...")
env.launch()

print("Loading task...")
task = env.get_task(ReachTarget)

print("Resetting task...")
descriptions, obs = task.reset()

print("Descriptions:", descriptions)
print("Observation type:", type(obs))

env.shutdown()
print("RLBench smoke test OK")
