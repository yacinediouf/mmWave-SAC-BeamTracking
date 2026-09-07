import numpy as np
from stable_baselines3 import SAC
from Environnement import RayMobTimeEnv

# 1. Charger l'environnement et le MEILLEUR modèle
env = RayMobTimeEnv()
model = SAC.load("models/best_model/best_model.zip")

# 2. Tester le modèle sur plusieurs épisodes
nb_episodes = 10
recompenses_totales = []

for ep in range(nb_episodes):
    obs, info = env.reset()
    done = False
    ep_reward = 0

    while not done:
        # deterministic=True : On demande à l'agent de donner SA MEILLEURE action sans explorer
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward

    recompenses_totales.append(ep_reward)
    print(f"Épisode {ep + 1}: Récompense = {ep_reward:.2f}")

print(f"\n🎯 Récompense moyenne sur {nb_episodes} tests : {np.mean(recompenses_totales):.2f}")