import numpy as np
from Environnement import RayMobTimeEnv

# 1. Charger tes vraies données CSI (adapte le chemin selon ton projet)
# Si tes données sont sous forme de fichier .npy ou .npz :
csi_data = np.load("./csi_data.npy")  # <--- Mettre le bon nom/chemin de ton fichier de données

# 2. Instancier l'environnement avec les données
env = RayMobTimeEnv(csi_data=csi_data)

# 3. Réinitialiser et faire UN pas de test
obs, _ = env.reset()
action_aleatoire = env.action_space.sample()
env.step(action_aleatoire)