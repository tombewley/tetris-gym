import argparse
import gym, tetris
from rlutils import make, train # https://github.com/tombewley/rlutils
from rlutils.observers.loggers import EpLengthLogger

parser = argparse.ArgumentParser()
parser.add_argument("agent_class", type=str)
args = parser.parse_args()

env = gym.make("Tetris-v0", vector_obs=True, render_mode="rgb_array")

if args.agent_class == "actor_critic":
    agent = make("actor_critic", env, {
        "net_pi": [(None, 256), "R", (256, 128), "R", (128, 64), "R", (64, None), "S"],
        "net_V": [(None, 256), "R", (256, 128), "R", (128, 64), "R", (64, None)],
        "gamma": 0.95
    })
elif args.agent_class == "reinforce":
    agent = make("reinforce", env, {
        "net_pi": [(None, 256), "R", (256, 128), "R", (128, 64), "R", (64, None), "S"],
        "net_V": [(None, 256), "R", (256, 128), "R", (128, 64), "R", (64, None)]
    })
elif args.agent_class == "dqn":
    agent = make("dqn", env, {
        "net_Q": [(None, 256), "R", (256, 128), "R", (128, 64), "R", (64, None)],
        "replay_capacity": 20*5000,
        "epsilon_start": 1.0, # NOTE: Random exploration
        "epsilon_end": 1.0,
    })

train(agent, {
    "agent": args.agent_class,
    "num_episodes": int(1e6),
    "checkpoint_freq": 5000,
    "video_freq": 500,
    "video_to_wandb": True,
    "render_freq": 0,
    "project_name": "tetris",
    "wandb_monitor": True,
    },
    observers={"": EpLengthLogger()}
)
