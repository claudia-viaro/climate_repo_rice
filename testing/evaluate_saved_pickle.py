import pickle
from pathlib import Path

import pickle

with open("outputs/1761147954/eval/episode_summaries.pkl", "rb") as f:
    episodes = pickle.load(f)

climate_info = episodes[0]["climate_info"]

print(climate_info.keys())
# → dict_keys(['timestep', 'rl_summary', 'bau_history'])

print(len(climate_info["rl_summary"]), "timesteps in RL summary")
print(len(climate_info["bau_history"]), "timesteps in BAU")
