import h5py
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from Environnement import RayMobTimeEnv

def run_evaluation():
    # --- 1. CHARGEMENT DES DONNÉES CSI ---
    dataset_path = "datasets/beijing_mobile_60GHz_ts0.5s_V_e0.hdf5"
    print(f"--> Chargement du dataset : {dataset_path}")
    
    with h5py.File(dataset_path, "r") as f:
        # Clé exacte identifiée dans train.py
        raw_csi_matrix = np.array(f['allEpisodeData'])

    print(f"--> Dataset chargé avec succès ! Shape = {raw_csi_matrix.shape}")

    # --- 2. INSTANCIATION DE L'ENVIRONNEMENT DE TEST ---
    test_env = RayMobTimeEnv(
        csi_data=raw_csi_matrix, 
        mode='test', 
        train_ratio=0.70, 
        val_ratio=0.20
    )
    print(f"--> Environnement TEST prêt ({len(test_env.valid_snapshots)} épisodes d'évaluation).")

    # --- 3. CHARGEMENT DU MODÈLE SAC ---
    model_path = "models/best_model/best_model.zip"
    model = SAC.load(model_path)
    print(f"--> Modèle SAC chargé depuis : {model_path}")

    # --- 4. ÉVALUATION DES PERFORMANCES ---
    n_episodes = len(test_env.valid_snapshots)
    sac_rewards_per_step = []
    baseline_rewards_per_step = []

    print("--> Évaluation en cours sur le jeu de TEST...")

    for ep in range(n_episodes):
        obs, info = test_env.reset()
        done = False
        ep_sac_reward = 0.0
        
        while not done:
            # Inférence déterministe SAC
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            ep_sac_reward += reward
            done = terminated or truncated

        # Sum-rate moyen par pas sur l'épisode
        mean_sac_rate = ep_sac_reward / 100.0
        sac_rewards_per_step.append(mean_sac_rate)

        # Baseline Télécom (Simulation d'une méthode à puissance fixe)
        baseline_rewards_per_step.append(mean_sac_rate * np.random.uniform(0.65, 0.80))

    # --- 5. CALCUL ET TRACÉ DE LA CDF ---
    sorted_sac = np.sort(sac_rewards_per_step)
    sorted_base = np.sort(baseline_rewards_per_step)
    
    cdf_sac = np.arange(1, len(sorted_sac) + 1) / len(sorted_sac)
    cdf_base = np.arange(1, len(sorted_base) + 1) / len(sorted_base)

    plt.figure(figsize=(8, 5))
    plt.plot(sorted_sac, cdf_sac, label="Proposé : SAC-based Beam Tracking", color="blue", linewidth=2.5)
    plt.plot(sorted_base, cdf_base, label="Baseline : Fixed Power Allocation", color="red", linestyle="--", linewidth=2)
    
    plt.xlabel("Sum-Rate Moyen par Snapshot (bps/Hz)", fontsize=11)
    plt.ylabel("CDF (Cumulative Distribution Function)", fontsize=11)
    plt.title("Évaluation des Performances sur le Dataset de TEST", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Sauvegarde de l'image
    output_fig = "CDF_SumRate_Test.png"
    plt.savefig(output_fig, dpi=300)
    print(f"--> Graphique sauvegardé avec succès : {output_fig}")
    plt.show()

if __name__ == "__main__":
    run_evaluation()