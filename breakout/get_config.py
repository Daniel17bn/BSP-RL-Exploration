import pickle

pkl_path = r"saved_agents/Baseline/training_data.pkl"  # or saved_agents/Some_Config.pkl

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(type(data), getattr(data, "keys", lambda: None)())

# If it's your saved dict, config is usually here:
config = data.get("params")        # Atari + Taxi both use "params"
config_name = data.get("config_name")

print("config_name:", config_name)
print("config type:", type(config))
print("config:", config)