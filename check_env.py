from Environnement import RayMobTimeEnv

# Instancier l'environnement
env = RayMobTimeEnv()

# 1. Vérifier l'espace d'action (Action Space)
print("=" * 50)
print("📐 ESPACE D'ACTION (Action Space) :")
print(env.action_space)
print(f"Forme de l'action (shape) : {env.action_space.shape}")
print(f"Valeurs min/max : [{env.action_space.low}, {env.action_space.high}]")

# 2. Vérifier s'il y a des attributs d'antennes enregistrés
print("\n📡 INFORMATIONS ANTENNES :")
if hasattr(env, "num_antennas"):
    print(f"Nombre d'antennes (num_antennas) : {env.num_antennas}")
elif hasattr(env, "n_antennas"):
    print(f"Nombre d'antennes (n_antennas) : {env.n_antennas}")
else:
    print(
        "Attribut d'antenne non trouvé directement (regarde dans Environnement.py)."
    )
print("=" * 50)