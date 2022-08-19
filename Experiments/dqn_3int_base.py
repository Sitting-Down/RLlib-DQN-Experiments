import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import sumo_rl
from sumo_rl import SumoEnvironment

#Meant to use Sumo-RL to get data where Eclipse sumo uses the default traffic light configurations given to it
if __name__ == '__main__':
    LIBSUMO_AS_TRACI = 1
    #Check the data if Max Number of Steps in Eclipse Sumo is set to [time]
    time = [100800,
            500000]
    #2 Routes to check data on
    routes = ['day1-7', 'day8-14']
    for x in time:
        print(x)
        for route in routes:
            print(route)
            runs = 2
            out_csv_name = f'D:/0-School/DQN-RLlib-SumoRL-Experiments/Outputs/base/{x}/{route}/dqn'
            env = SumoEnvironment(
                        route_file=f'D:/0-School/DQN-RLlib-SumoRL-Experiments/Simulation/routes/{route}.rou.xml',
                        net_file='D:/0-School/DQN-RLlib-SumoRL-Experiments/Simulation/nets/mainV2.net.xml',
                        out_csv_name=out_csv_name,
                        single_agent=False,
                        use_gui=False,
                        fixed_ts=True,
                        num_seconds=x,
                        max_depart_delay=x,
                        waiting_time_memory=x,
                        min_green=5,
                        max_green=60,
                        yellow_time=4,
                        delta_time= 5,
                        sumo_warnings=True,
                        reward_fn='diff-waiting-time'
                        )

            for run in range(1, runs+1):
                initial_states = env.reset()
                print(run)
                info = []
                done = {'__all__': False}
                while not done['__all__']:
                    s, r, done, info = env.step(action=None)

                env.save_csv(out_csv_name, run)
                print('saved')
                env.close()


