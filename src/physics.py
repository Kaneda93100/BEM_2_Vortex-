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
    Ft =- q * Ct #signe moins pour retrouver la même convention
    
    return Fn, Ft

def compute_cp(df, col_fn, col_ft, R_rotor=2.25, Nb_pales=3, omega=44.5163679):
    """
    Calcule le coefficient de puissance (Cp) de l'éolienne à partir des champs de forces.
    Gère dynamiquement les variations de yaw et de TSR.
    """
    # On travaille sur une copie pour ne pas altérer le DataFrame original
    df_calc = df.copy()
    geom = get_geometry()
    
    # 1. Calcul des longueurs de sections (dl)
    r_unique = np.sort(df_calc['r'].unique())
    nodes = [0.21] # Rayon du moyeu (hub)
    for i in range(len(r_unique)):
        next_node = 2 * r_unique[i] - nodes[-1]
        nodes.append(next_node)
    
    dl_map = dict(zip(r_unique, np.diff(nodes)))
    df_calc['dl'] = df_calc['r'].map(dl_map)
    
    # 2. Projection (calcul de phi en radians)
    df_calc['phi_rad'] = PITCH_RAD + geom.get_twist_rad(df_calc['r'])
    
    # 3. Calcul du couple élémentaire (dQ)
    Ft_corrige = df_calc[col_ft] * -1
    cos_phi = np.cos(df_calc['phi_rad'])
    sin_phi = np.sin(df_calc['phi_rad'])
    
    df_calc['dQ_r'] = df_calc[col_fn] * sin_phi + Ft_corrige * cos_phi
    df_calc['dQ'] = df_calc['dQ_r'] * df_calc['r'] * df_calc['dl']
    
    # 4. Préparation du GroupBy
    group_cols = ['yaw']
    if 'TSR' in df_calc.columns:
        group_cols.append('TSR')
        
    results = []
    model_name = col_fn.replace('Fn_', '') # Extrait 'pred', 'SVEN', 'BEM', etc.
    
    for name, group in df_calc.groupby(group_cols):
        if 'TSR' in group_cols:
            yaw_val = name[0] if isinstance(name, tuple) else name
            tsr_val = name[1] if isinstance(name, tuple) else 8.0
        else:
            yaw_val = name[0] if isinstance(name, tuple) else name
            tsr_val = 8.0 # Valeur de TSR par défaut
            
  
        # Omega est fixé, on calcule le vent théorique correspondant au TSR
        u_vent = (omega * R_rotor) / tsr_val
        
        # Intégrations
        Q_theta = group.groupby('theta')['dQ'].sum()  # Somme sur le rayon
        Q_moyen = Q_theta.mean()                      # Moyenne sur l'azimut
        
        # Puissances
        P_meca = Nb_pales * Q_moyen * omega
        P_vent = 0.5 * RHO * np.pi * (R_rotor**2) * (u_vent**3)
        
        Cp = P_meca / P_vent
        
        res_dict = {
            'yaw': yaw_val,
            'TSR': tsr_val,
            f'Cp_{model_name}': Cp
        }
        results.append(res_dict)
        
    return pd.DataFrame(results)