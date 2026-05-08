import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d, RegularGridInterpolator

# ==========================================
# CONSTANTES PHYSIQUES ET OPÉRATIONNELLES
# ==========================================
RHO = 1.198                 # Densité de l'air kg/m3
U_INFTY = 12.52             # m/s (TSR 8)
PITCH_RAD = -0.040143       # Angle de pitch en radians (-2.3 degrés)

# ==========================================
# GESTIONNAIRE DE GÉOMÉTRIE
# ==========================================
class BladeGeometry:
    def __init__(self, geom_file="geometry/blade_geom.csv", airfoils_file="geometry/airfoils.csv"):
        self.geom_file = geom_file
        self.airfoils_file = airfoils_file
        
        self._load_blade_geometry()
        self._load_airfoils()

    def _load_blade_geometry(self):
        if not os.path.exists(self.geom_file):
            raise FileNotFoundError(f"Fichier géométrie introuvable: {self.geom_file}")

        df_geom = pd.read_csv(self.geom_file)
        self.nodesRadius = df_geom['r'].values
        self.nodesChord = df_geom['chord'].values
        self.nodesTwist_deg = df_geom['beta_deg'].values

        self.get_chord = interp1d(self.nodesRadius, self.nodesChord, kind='linear', fill_value="extrapolate")
        self.get_twist_rad = interp1d(self.nodesRadius, np.radians(self.nodesTwist_deg), kind='linear', fill_value="extrapolate")

    def _load_airfoils(self):
        if not os.path.exists(self.airfoils_file):
            raise FileNotFoundError(f"Fichier polaires introuvable: {self.airfoils_file}")

        df_airfoils = pd.read_csv(self.airfoils_file)
        
        alphas_deg = np.sort(df_airfoils['alpha_deg'].unique())
        radii = np.sort(df_airfoils['r'].unique())
        
        df_cl = df_airfoils.pivot(index='alpha_deg', columns='r', values='Cl')
        df_cd = df_airfoils.pivot(index='alpha_deg', columns='r', values='Cd')
        
        cl_matrix = df_cl.loc[alphas_deg, radii].values
        cd_matrix = df_cd.loc[alphas_deg, radii].values
        
        self._interp_cl = RegularGridInterpolator((alphas_deg, radii), cl_matrix, method='linear', bounds_error=False, fill_value=None)
        self._interp_cd = RegularGridInterpolator((alphas_deg, radii), cd_matrix, method='linear', bounds_error=False, fill_value=None)

    def get_cl_cd(self, r, alpha_deg):
        """ 
        Retourne Cl et Cd interpolés. 
        ATTENTION: alpha_deg doit être en DEGRÉS.
        """
        cl = self._interp_cl((alpha_deg, r))
        cd = self._interp_cd((alpha_deg, r))
        return float(cl), float(cd)


geom_db = None
def get_geometry():
    global geom_db
    if geom_db is None:
        geom_db = BladeGeometry()
    return geom_db


# ==========================================
# FONCTIONS DE CONVERSIONS PHYSIQUES
# ==========================================

def convert_v_to_f(V_eff, alpha_deg, r):
    """
    Convertit v (V_eff, alpha) en efforts (Fn, Ft) en [N/m].
    Prend en entrée alpha en DEGRÉS.
    """
    geom = get_geometry()
    
    # Conversion de alpha en radians juste pour np.cos et np.sin
    alpha_rad = np.radians(alpha_deg)
    
    if isinstance(r, (list, np.ndarray, pd.Series)):
        c = np.array([geom.get_chord(ri) for ri in r])
        # On passe alpha_deg à get_cl_cd
        cl_cd = [geom.get_cl_cd(ri, ai) for ri, ai in zip(r, alpha_deg)]
        Cl = np.array([item[0] for item in cl_cd])
        Cd = np.array([item[1] for item in cl_cd])
    else:
        c = geom.get_chord(r)
        Cl, Cd = geom.get_cl_cd(r, alpha_deg)
    
    # Projections géométriques (Nécessite alpha en radians)
    Cn = Cl * np.cos(alpha_rad) + Cd * np.sin(alpha_rad)
    Ct = Cl * np.sin(alpha_rad) - Cd * np.cos(alpha_rad)
    
    # Pression dynamique et Forces
    q = 0.5 * RHO * (V_eff**2) * abs(c)
    Fn = q * Cn
    Ft =- q * Ct
    
    return Fn, Ft


def convert_u_to_f(a, phi_deg, r):
    """
    Pipeline complète.
    """
    V_eff, alpha_deg = convert_u_to_v(a, phi_deg, r)
    Fn, Ft = convert_v_to_f(V_eff, alpha_deg, r)
    return Fn, Ft