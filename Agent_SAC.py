# agent_sac.py
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

def creer_agent_sac(train_env):
    """Initialise et configure le modèle SAC avec les hyperparamètres."""
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))

    model = SAC(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=5e-5,     # Baisse à 1e-4 pour lisser les mises à jour
        buffer_size=50000,      # Mémoire de 50 000 pas
        learning_starts=500,    # Phase de remplissage initial du buffer
        batch_size=256,
        tau=0.005,
        gamma=0.95,
        ent_coef="auto",
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        train_freq=1,
        gradient_steps=1,
    )
    return model


def entrainer_agent(model, train_env, val_env, total_timesteps=100000):
    """Exécute l'apprentissage tout en évaluant sur l'environnement de validation."""
    os.makedirs("./models/best_model/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)

    # 🎯 CRUCIAL : EvalCallback évalue UNIQUEMENT sur val_env (snapshots inédits)
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path="./models/best_model/",
        log_path="./logs/",
        eval_freq=1000,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path="./models/checkpoints/", 
        name_prefix="sac_beam"
    )

    print(f"🚀 Début de l'entraînement pour {total_timesteps} pas...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
    )

    model.save("models/sac_beamforming_final")
    print("💾 Entraînement terminé ! Le meilleur modèle est sauvegardé dans 'models/best_model/'.")