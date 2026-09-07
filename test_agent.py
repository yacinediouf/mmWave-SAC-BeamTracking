import numpy as np
from stable_baselines3 import SAC
from Environnement import RayMobTimeEnv

def evaluer_agent_vs_aleatoire(num_episodes=5):
    # 1. Instancier l'environnement de test
    env = RayMobTimeEnv()
    
    # 2. Charger l'agent SAC entraîné
    print("🔄 Chargement de l'agent SAC entraîné...")
    model = SAC.load("models/sac_beamforming_final", env=env)
    
    rewards_sac = []
    rewards_random = []
    
    print(f"\n🚀 Début de l'évaluation sur {num_episodes} épisodes...")
    
    for ep in range(num_episodes):
        # --- TEST AGENT SAC ---
        obs, _ = env.reset()
        done = False
        ep_rew_sac = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True) # Mode déterministe pour le test
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rew_sac += reward
            done = terminated or truncated
        rewards_sac.append(ep_rew_sac)
        
        # --- TEST SÉLECTION ALÉATOIRE ---
        env.reset()
        done = False
        ep_rew_random = 0
        while not done:
            action_aleatoire = env.action_space.sample() # Choix au hasard
            _, reward, terminated, truncated, _ = env.step(action_aleatoire)
            ep_rew_random += reward
            done = terminated or truncated
        rewards_random.append(ep_rew_random)
        
        print(f"Épisode {ep+1} | SAC: {ep_rew_sac:.2f} Mbps | Aléatoire: {ep_rew_random:.2f} Mbps")
        
    print("\n" + "="*50)
    print("📊 BILAN COMPARATIF FINAL")
    print("="*50)
    print(f"🏆 Débit Moyen Cumulé SAC        : {np.mean(rewards_sac):.2f} Mbps")
    print(f"🎲 Débit Moyen Cumulé Aléatoire : {np.mean(rewards_random):.2f} Mbps")
    gain = ((np.mean(rewards_sac) - np.mean(rewards_random)) / np.mean(rewards_random)) * 100
    print(f"📈 Gain de performance grâce au SAC : +{gain:.2f}%")
    print("="*50)

if __name__ == "__main__":
    evaluer_agent_vs_aleatoire()