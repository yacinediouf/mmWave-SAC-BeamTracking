import numpy as np
# On importe la classe que tu viens de coder dans Environnement.py
from Environnement import RayMobTimeEnv

print("="*50)
print("🚀 Lancement du test de l'environnement Gym...")
print("="*50)

try:
    # 1. Initialisation de l'environnement
    env = RayMobTimeEnv()
    
    # 2. Reset de l'environnement pour obtenir le premier état (t=0)
    state, info = env.reset()
    print(f"✅ Environnement initialisé avec succès !")
    print(f"📊 Taille de l'espace d'observations (State Shape) : {state.shape}")
    print(f"🎛️ Espace d'actions configuré : {env.action_space}")
    print("-"*50)

    # 3. Simulation d'un épisode complet avec des actions aléatoires
    print("🏃 Simulation de l'épisode en cours (50 pas de temps)...")
    total_reward = 0
    steps = 0
    terminated = False
    
    while not terminated:
        # L'environnement génère une action aléatoire entre -1.0 et 1.0 pour chaque utilisateur
        action_aleatoire = env.action_space.sample()
        
        # On exécute l'action dans l'environnement
        next_state, reward, terminated, truncated, info = env.step(action_aleatoire)
        
        total_reward += reward
        steps += 1
        
        # On affiche un aperçu pour les premiers pas
        if steps <= 3:
            print(f"   [Pas {steps}] Récompense (Sum-Rate) obtenue : {reward:.2f} Mbps")

    print("-"*50)
    print(f"🏁 Fin de la simulation ! Nombre de pas exécutés : {steps}/{env.num_steps}")
    print(f"🏆 Récompense cumulée sur l'épisode : {total_reward:.2f} Mbps")
    print("🎉 Félicitations, ton environnement fonctionne à la perfection !")
    print("="*50)

except Exception as e:
    print(f"❌ Erreur lors de l'exécution de l'environnement :")
    print(str(e))
    print("="*50)