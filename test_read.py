import h5py
import numpy as np

# Le chemin pointe vers le premier fichier d'épisode (e0)
dataset_path = "datasets/beijing_mobile_60GHz_ts0.5s_V_e0.hdf5"

try:
    with h5py.File(dataset_path, 'r') as f:
        print("==================================================")
        print("🎉 Connexion réussie à l'Épisode 0 de RayMobTime !")
        print("==================================================")
        
        # Afficher la structure réelle du fichier pour voir les variables
        print("Variables disponibles dans ce fichier d'épisode :")
        print(list(f.keys()))
        print("--------------------------------------------------\n")
        
        # Recherche automatique de la clé contenant les matrices de canaux
        # Dans RayMobTime, c'est souvent 'channel' ou 'channel_matrix' ou 'output'
        clés = list(f.keys())
        nom_cle_canal = clés[0] # On prend la première par défaut pour inspecter
        
        donnees_canal = f[nom_cle_canal]
        print(f"Inspection de la variable '{nom_cle_canal}' :")
        print(f" - Forme des données (Shape) : {donnees_canal.shape}")
        print("--------------------------------------------------")
        
        # Si c'est un tableau de données, on affiche le premier pas de temps t=0
        if len(donnees_canal.shape) > 1:
            print("📡 EXTRAIT DES COEFFICIENTS DU CANAL (t = 0) :")
            print(donnees_canal[0])
        else:
            print("Données brutes de la variable :")
            print(donnees_canal[:5]) # Affiche les 5 premiers éléments
            
        print("--------------------------------------------------")
        print("Succès : Le terminal est prêt à alimenter ton environnement RL !")

except FileNotFoundError:
    print(f"❌ Erreur : Place le fichier 'beijing_mobile_60GHz_ts0.5s_V_e0.hdf5' dans le dossier 'datasets/'")
except Exception as e:
    print(f"❌ Une erreur est survenue : {e}")