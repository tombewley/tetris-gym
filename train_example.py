import gym, tetris
from rlutils import make, train # https://github.com/tombewley/rlutils
from rlutils.observers.loggers import EpLengthLogger

env = gym.make("Tetris-v0", vector_obs=True, render_mode="rgb_array")

agent = make("actor_critic", env, {
    "net_pi": [(None, 256), "R", (256, 128), "R", (128, None), "S"],
    "net_V": [(None, 256), "R", (256, 128), "R", (128, None)]
})

train(agent, {
    "num_episodes": int(1e6),
    "checkpoint_freq": 1000,
    "video_freq": 200,
    "video_to_wandb": True,
    "render_freq": 0,
    "project_name": "tetris",
    "wandb_monitor": True,
    },
    observers={"": EpLengthLogger()}
)
