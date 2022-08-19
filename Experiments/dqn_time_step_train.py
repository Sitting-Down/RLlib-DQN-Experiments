import os
import itertools
import sys
import ray
import sumo_rl
import supersuit

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from ray.rllib.agents.dqn.dqn import DQNTrainer
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env

SAVE_PATH = 'D:/0-School/DQN-RLlib-SumoRL-Experiments/checkpoints/test'
LOAD_PATH = 'D:/0-School/DQN-RLlib-SumoRL-Experiments/checkpoints/test/checkpoint_000025/checkpoint-25'

D: / 0 - School / DQN - RLlib - SumoRL - Experiments / Simulation

if __name__ == '__main__':
    # Supposedly speeds up the sumo simu but haven't noticed, kept just in case
    LIBSUMO_AS_TRACI = 1
    ray.init()
    env = sumo_rl.parallel_env(
        route_file='D:/0-School/DQN-RLlib-SumoRL-Experiments/Simulation/routes/day1-7V1.rou.xml',
        net_file='D:/0-School/DQN-RLlib-SumoRL-Experiments/Simulation/nets/mainv2.net.xml',
        out_csv_name='D:/0-School/DQN-RLlib-SumoRL-Experiments/Outputs/dqn',
        single_agent=False,
        use_gui=False,
        fixed_ts=False,
        num_seconds=100800,
        max_depart_delay=100800,
        waiting_time_memory=1000000,
        min_green=5,
        max_green=60,
        yellow_time=4,
        delta_time=5,
        sumo_warnings=True,
        reward_fn='diff-waiting-time'
    )

    env = supersuit.pad_observations_v0(env)
    env = supersuit.pad_action_space_v0(env)
    env = ParallelPettingZooEnv(env)

    padded_action_space = env.action_space
    padded_observation_space = env.observation_space

    register_env("dqn_3_step_learning_intersections", lambda config: env)
    trainer = DQNTrainer(env="dqn_3_step_learning_intersections", config={
        "multiagent": {
            "policies": {
                'test': (DQNTFPolicy, padded_observation_space, padded_action_space, {}),
            },
            "policy_mapping_fn": (lambda _: 'test')
        },
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 80640,
        },
        "lr": 0.001,
        'log_level': 'INFO',
        'num_workers': 1,
        'num_gpus': 1,
        "no_done_at_end": True,
        'record_env': True,
        "batch_mode": "complete_episodes"

                      'create_env_on_driver': True,
                                              'evaluation_interval': 1,
    "evaluation_duration": 1,
    # "evaluation_num_workers": 1,
    "train_batch_size": 35,
    "rollout_fragment_length": 5,

    "replay_buffer_config": {
        "_enable_replay_buffer_api": True,
        "learning_starts": 0,
        "type": "MultiAgentReplayBuffer",
        "replay_batch_size": 35,
        "replay_sequence_length": 1,
    },

    })
    print('loaded')
    trainer.restore(LOAD_PATH)
    for step in itertools.count():
        print(step)
        trainer.train()  # distributed training step
        print('checkpointing')
        trainer.save(SAVE_PATH)
        print('done checkpointing')

