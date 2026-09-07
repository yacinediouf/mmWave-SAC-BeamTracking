import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RayMobTimeEnv(gym.Env):
    def __init__(self, csi_data, mode='train', train_ratio=0.70, val_ratio=0.20):
        super(RayMobTimeEnv, self).__init__()

        self.csi_data = csi_data  # Shape: (50, 10, 100, 8)
        self.mode = mode.lower()

        # --- 1. SÉPARATION DES SNAPSHOTS (70% Train / 20% Val / 10% Test) ---
        num_total_snapshots = self.csi_data.shape[0]  # Ex: 50
        
        train_end = int(train_ratio * num_total_snapshots)                # 35 épisodes (0 à 34)
        val_end = int((train_ratio + val_ratio) * num_total_snapshots)   # 45 épisodes (35 à 44)

        if self.mode == 'train':
            self.valid_snapshots = list(range(0, train_end))
        elif self.mode == 'val':
            self.valid_snapshots = list(range(train_end, val_end))
        elif self.mode == 'test':
            self.valid_snapshots = list(range(val_end, num_total_snapshots))
        else:
            raise ValueError("Le mode doit être 'train', 'val' ou 'test'")
        
        # --- PARAMÈTRES PHYSIQUES DU SYSTÈME ---
        self.num_users = 10       # N = 10 utilisateurs
        self.num_beams = 64      # Taille du codebook (64 faisceaux)
        self.N_t = 64            # Nombre d'antennes à l'émission (gNB)
        self.N_r = 1             # Nombre d'antennes à la réception (UE SIMO/MISO)

        # Dans __init__() de Environnement.py
        self.tx_power_dBm = 30.0  # 30 dBm = 1 Watt (Valeur standard en mmWave)
        self.tx_power = 10**((self.tx_power_dBm - 30) / 10)  # Converti en Watts (= 1.0)

        # Bande passante de 100 MHz et Facteur de bruit de 6 dB
        self.bandwidth = 100e6 
        noise_figure_dB = 6.0
        noise_dBm = -174 + 10 * np.log10(self.bandwidth) + noise_figure_dB  # -88 dBm
        self.noise_power = 10**((noise_dBm - 30) / 10)  # ~1.58e-12 Watts

        # Codebook de faisceaux (steering vectors)
        self.codebook = self._build_dft_codebook(self.N_t, self.num_beams)

        # --- ESPACES GYMNASIUM ---
        # Action continue ramenée à l'orientation des faisceaux pour 10 UEs
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_users,), dtype=np.float32
        )
        
        # Observation : Caractéristiques du canal pour chaque utilisateur
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_users, 2), dtype=np.float32
        )

        self.current_snapshot_idx = None
         
        # Paramètres de gestion des épisodes
        self.max_steps_per_episode = 100
        self.current_step = 0

    def _build_dft_codebook(self, N_antennas, N_beams):
        """ Génère les steering vectors a_t(phi) de l'équation (1). """
        angles = np.linspace(-np.pi/2, np.pi/2, N_beams)
        codebook = np.zeros((N_antennas, N_beams), dtype=complex)
        for i in range(N_beams):
            # Vector a_t(phi) = (1/sqrt(N_t)) * [1, e^(-j*pi*sin(phi)), ...]
            codebook[:, i] = np.exp(-1j * np.pi * np.arange(N_antennas) * np.sin(angles[i])) / np.sqrt(N_antennas)
        return codebook

    def _compute_channel_H(self, csi_user_paths):
        """
        ÉQUATION (1) DE NOTRE ARTICLE :
        H_t = sqrt(N_t * N_r / L) * sum_{l=1}^{L} [ alpha_{u,l} * a_r(theta_{l,t}) * a_t^H(phi_{l,t}) ]
        """
        # Exfiltration des trajets depuis le dataset (100, 8)
        alpha = csi_user_paths[:, 0] + 1j * csi_user_paths[:, 1]  # Gains complexes alpha_{u,l}
        phi_aod = csi_user_paths[:, 3]                           # Angles AoD (phi_{l,t})
        
        # Filtrage des trajets non nuls (L trajets réels)
        valid_indices = np.abs(alpha) > 1e-12
        alpha_l = alpha[valid_indices]
        phi_l = phi_aod[valid_indices]
        L = len(alpha_l)
        
        if L == 0:
            return np.zeros(self.N_t, dtype=complex)
        
        # Matrice des steering vectors a_t(phi_{l,t})
        idx = np.arange(self.N_t)[:, np.newaxis]
        A_t = np.exp(-1j * np.pi * idx * np.sin(phi_l)) / np.sqrt(self.N_t) # (N_t, L)
        
        # Équation (1) : Somme pondérée par alpha_l et normalisée par sqrt(N_t * N_r / L)
        scaling_factor = np.sqrt((self.N_t * self.N_r) / L)
        H_u = scaling_factor * (A_t @ alpha_l)  # (N_t,)
        
        return H_u

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

       # 1. Tirer un épisode au hasard parmi les snapshots valides de ce mode
        selected_episode = np.random.choice(self.valid_snapshots)

       # 2. On démarre TOUJOURS au premier pas (index 0) de cet épisode 
       # pour garantir un épisode complet de 100 snapshots
        self.current_snapshot_idx = selected_episode

       # 3. Récupération du premier snapshot CSI
        current_csi = self.csi_data[self.current_snapshot_idx]

       # 4. Construction de l'observation initiale
        observation = self._get_observation(current_csi)
        info = {"snapshot_index": self.current_snapshot_idx, "mode": self.mode}

        return observation, info

    def step(self, action):
    # 1. Utiliser le snapshot CSI courant
        current_csi = self.csi_data[self.current_snapshot_idx]

        # --- Calculs physiques Télécom ---
        # 1. Reconstitution de H (K x Nt)
        H_list = []
        for u in range(self.num_users):
            H_u = self._compute_channel_H(current_csi[u])
            if H_u.ndim == 2:
                H_u = H_u[0, :]  # Prend le vecteur MISO (1 x Nt)
            H_list.append(H_u)
    
        H = np.array(H_list)  # Forme : (num_users, N_t)

         # 2. Pseudo-inverse Zero-Forcing : W_zf = H^H * (H * H^H)^(-1)
        H_H = H.conj().T
        try:
            W_zf = H_H @ np.linalg.inv(H @ H_H)
        except np.linalg.LinAlgError:
            W_zf = np.linalg.pinv(H)

        # 3. Normalisation et application de l'action SAC
        W = []
        for u in range(self.num_users):
            w_u = W_zf[:, u]  # Vecteur colonne pour l'utilisateur u
            norm_w = np.linalg.norm(w_u)
            if norm_w > 0:
               w_u = w_u / norm_w
        
            # Action SAC [0, 1]
            scale = (action[u] + 1.0) / 2.0 if hasattr(action, '__len__') and len(action) >= self.num_users else 1.0
            W.append(w_u * scale)
        
        W = np.array(W)  # Forme (num_users, N_t)

        # 4. Calcul du SINR et du Sum-Rate
        sinr_list = []
        for u in range(self.num_users):
            signal_power = self.tx_power * np.abs(np.vdot(H[u], W[u]))**2
        
            interference = sum(
                self.tx_power * np.abs(np.vdot(H[u], W[j]))**2
                for j in range(self.num_users) if j != u
            )
        
            sinr_u = signal_power / (self.noise_power + interference)
            sinr_list.append(sinr_u)

        r_t = sum(np.log2(1 + sinr_u) for sinr_u in sinr_list)
        reward = float(r_t)

    # --- Gestion du temps et des pas (AVANCE D'UN PAS DANS LE CANAL) ---
        # --- 2. GESTION DU TEMPS ET DES PAS ---
        self.current_step += 1

        # L'épisode s'arrête exactement au bout des 100 pas (max_steps_per_episode)
        if self.current_step >= self.max_steps_per_episode:
            terminated = True
        else:
            terminated = False

        truncated = False

        # --- 3. RÉCUPÉRATION DE LA PROCHAINE OBSERVATION (t+1) ---
        current_csi = self.csi_data[self.current_snapshot_idx]
        observation = self._get_observation(current_csi)

        info = {
            "sum_rate_bps_hz": reward, 
            "selected_beams": []
        }

        return observation, reward, terminated, truncated, info

            #if u == 0:
            #    print(f"--- PAS DE TEST ZF ---")
            #    print(f"Signal Utile: {signal_power:.4f}")
            #    print(f"Interference: {interference:.12f}")
            #    print(f"Noise Power: {self.noise_power}")
            #    print(f"SINR (linéaire): {sinr_u:.4f}")
                
    def _action_to_beams(self, action):
        scaled = (action + 1.0) / 2.0
        beam_indices = np.floor(scaled * self.num_beams).astype(int)
        return np.clip(beam_indices, 0, self.num_beams - 1)

    def _get_observation(self, csi_snapshot):
        obs = np.zeros((self.num_users, 2), dtype=np.float32)
        for u in range(self.num_users):
            H_u = self._compute_channel_H(csi_snapshot[u])
            obs[u, 0] = np.abs(np.mean(H_u))
            obs[u, 1] = np.angle(np.mean(H_u))
        return obs