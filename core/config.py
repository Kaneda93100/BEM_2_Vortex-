"""
=============================================================================
 CONFIGURATION GLOBALE - BEM 2 VORTEX
=============================================================================
"""

# ==========================================
# 1. CONSTANTES PHYSIQUES
# ==========================================
<<<<<<< Updated upstream
RHO = 1.198                 
U_INFTY = 12.52             
PITCH_RAD = -0.040143       
R_ROTOR = 2.25              
OMEGA = 44.5163679          
=======
RHO = 1.198                 # Densité de l'air [kg/m3]
U_INFTY = 12.52             # Vitesse du vent (TSR 8) [m/s]
PITCH_RAD = -0.040143       # Angle de pitch en radians (-2.3 degrés)
R_ROTOR = 2.25              # Rayon du rotor [m]
OMEGA = 44.5163679          # Vitesse de rotation [rad/s]
Dist_R = [0.2119407,  0.21968875, 0.23512587, 0.25813459, 0.28853979, 0.32611007, # Distribution des rayons de la pale
 0.3705595,  0.42154979, 0.47869288, 0.54155386, 0.60965434, 0.68247602,
 0.75946469 ,0.84003441, 0.92357201, 1.00944172, 1.09699   , 1.18555057,
 1.27444943 ,1.36301   , 1.45055828, 1.53642799, 1.61996559, 1.70053531,
 1.77752398 ,1.85034566, 1.91844614, 1.98130712, 2.03845021, 2.0894405,
 2.13388993 ,2.17146021, 2.20186541, 2.22487413, 2.24031125, 2.2480593 ]
>>>>>>> Stashed changes

# ==========================================
# 2. PARAMÈTRES D'ENTRAÎNEMENT (TRAINER)
# ==========================================
EPOCHS_OPTUNA = 1000         
EPOCHS_FINAL = 1000         
EPOCHS_AE = 500            

RANDOM_SEED = 42
CV_SPLITS = 5               
DEVICE = "cuda"             

# ==========================================
# 3. RECHERCHE OPTUNA
# ==========================================
TRIALS_GV = 150             
TRIALS_GM = 50              
TRIALS_AE = 15              

LR_BOUNDS = (1e-4, 5e-3)    
DROPOUT_BOUNDS = (0.0, 0.4) 

MLP_LAYERS_BOUNDS = (1, 4)
MLP_NEURONS_CHOICES = [128, 192, 256, 320, 384, 448, 512]

CNN_LAYERS_BOUNDS = (1, 4)
CNN_FILTERS_CHOICES = [16, 32, 64]
KERNEL_SIZE = 3             
PADDING_R = 1               
PADDING_THETA = 1           

# =========================================================================
# GRILLE DE RECHERCHE ARCHITECTURALE
# =========================================================================
ENTREES = ['GV', 'GM']
RESIDUELLES = ['0', '1', '2', '2+']
INTERMS = ['f', 'v']
OPTIONS = ['A', 'B'] 

<<<<<<< Updated upstream
AE_NATURES = ['V', 'M']  
AE_DIMS = [16, 32, 64, 128, 256, 512, 1024]

# =========================================================================
# HYPERPARAMÈTRES D'ENTRAÎNEMENT & OPTUNA
# =========================================================================
PRUNER_WARMUP = 150      
RATIO_THRESHOLD = 2.5    

AE_JSON_PATH = "training/hyperparametres/ae_hyperparameters.json"
AE_WEIGHTS_DIR = "training/models/ae/"
=======
# Dimensions latentes autorisées
LATENT_DIMS_GV = [0, 32, 64, 128, 256]
LATENT_DIMS_GM = [0, 128, 256, 512]
>>>>>>> Stashed changes
