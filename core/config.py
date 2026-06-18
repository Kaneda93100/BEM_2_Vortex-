"""
=============================================================================
 CONFIGURATION GLOBALE - BEM 2 VORTEX
 Centralisation des hyperparamètres, constantes physiques et règles d'entraînement.
=============================================================================
"""

# ==========================================
# 1. CONSTANTES PHYSIQUES (Anciennement dans physics.py)
# ==========================================
RHO = 1.198                 # Densité de l'air [kg/m3]
U_INFTY = 12.52             # Vitesse du vent (TSR 8) [m/s]
PITCH_RAD = -0.040143       # Angle de pitch en radians (-2.3 degrés)
R_ROTOR = 2.25              # Rayon du rotor [m]
OMEGA = 44.5163679          # Vitesse de rotation [rad/s]

# ==========================================
# 2. PARAMÈTRES D'ENTRAÎNEMENT (TRAINER)
# ==========================================
# Nombre d'époques
EPOCHS_OPTUNA = 150         # Époques pendant la recherche Optuna
EPOCHS_FINAL = 1000         # Époques pour l'entraînement final (et CV final)
EPOCHS_AE = 500             # Époques pour l'Auto-Encodeur

# Paramètres généraux
RANDOM_SEED = 42
CV_SPLITS = 3               # Nombre de K-Folds pour la validation croisée
DEVICE = "cuda"             

# ==========================================
# 3. RECHERCHE OPTUNA (ESPACES DES HYPERPARAMÈTRES)
# ==========================================
# Nombres d'essais (Trials)
TRIALS_GV = 150             # Stratégies Globales Vectorielles (MLP)
TRIALS_GM = 50              # Stratégies Globales Matricielles (CNN)
TRIALS_AE = 10              # Auto-Encodeurs

# Espaces de recherche continus
LR_BOUNDS = (1e-4, 5e-3)    # Limites du Learning Rate (Log scale)
DROPOUT_BOUNDS = (0.0, 0.4) # Limites du Dropout

# Espaces de recherche discrets (MLP)
MLP_LAYERS_BOUNDS = (2, 5)
MLP_NEURONS_CHOICES = [128, 192, 256, 320, 384, 448, 512]

# Espaces de recherche discrets (CNN)
CNN_LAYERS_BOUNDS = (2, 5)
CNN_FILTERS_CHOICES = [16, 32, 64]
KERNEL_SIZE = 3             # Taille du noyau CNN fixe
PADDING_R = 1               # Padding classique
PADDING_THETA = 1           # Padding circulaire

# ==========================================
# 4. FONCTIONS DE PERTE (LOSS) ET MODÉLISATION
# ==========================================
# Pondération pour la PhysicsInformedLoss (Stratégie 'v')
LAMBDA_INGENIEUR = 0.5      

# Dimensions latentes autorisées
LATENT_DIMS_GV = [0, 32, 64, 128, 256]
LATENT_DIMS_GM = [0, 128, 256, 512]