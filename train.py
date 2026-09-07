# train.py
import h5py
import numpy as np
from Environnement import RayMobTimeEnv
from Agent_SAC import creer_agent_sac, entrainer_agent

# 1. Chargement de la matrice 4D en RAM
dataset_path = "datasets/beijing_mobile_60GHz_ts0.5s_V_e0.hdf5"
print(f"--> Chargement du dataset en RAM : {dataset_path}...")

with h5py.File(dataset_path, "r") as f:
    # On extrait directement le tableau 4D de forme (50, 10, 100, 8)
    raw_csi_matrix = np.array(f['allEpisodeData'])

print(f"--> Dataset chargé avec succès ! Shape = {raw_csi_matrix.shape}")

# Instanciation des 3 environnements distincts (70% / 20% / 10%)
train_env = RayMobTimeEnv(csi_data=raw_csi_matrix, mode='train', train_ratio=0.70, val_ratio=0.20)
val_env   = RayMobTimeEnv(csi_data=raw_csi_matrix, mode='val',   train_ratio=0.70, val_ratio=0.20)
test_env  = RayMobTimeEnv(csi_data=raw_csi_matrix, mode='test',  train_ratio=0.70, val_ratio=0.20)

# 3. Création et lancement de l'entraînement SAC
model = creer_agent_sac(train_env)
entrainer_agent(model, train_env, val_env, total_timesteps=100000)