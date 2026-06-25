"""
=============================================================================
 CONFIGURATION GLOBALE - BEM 2 VORTEX
=============================================================================
"""

# ==========================================
# 1. CONSTANTES PHYSIQUES
# ==========================================
RHO = 1.198                 
U_INFTY = 12.52             
PITCH_RAD = -0.040143       
R_ROTOR = 2.25              
OMEGA = 44.5163679          

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

AE_NATURES = ['V', 'M']  
AE_DIMS = [16, 32, 64, 128, 256, 512, 1024]

# =========================================================================
# HYPERPARAMÈTRES D'ENTRAÎNEMENT & OPTUNA
# =========================================================================
PRUNER_WARMUP = 150      
RATIO_THRESHOLD = 2.5    

AE_JSON_PATH = "training/hyperparametres/ae_hyperparameters.json"
AE_WEIGHTS_DIR = "training/models/ae/"
