from src.data_loader import load_clean_data, get_splits
from src.optimize import optimize
from src.evaluate import evaluator

def main():
    # 1. Chargement global une seule fois en mémoire
    df_full = load_clean_data()
    
    # 2. Les stratégies à explorer
    entrees = ['L', 'GR', 'GA']
    residuelles = ['0', '1']
    inters = ['f', 'v', 'u']
    
    # 3. Boucles d'expérimentation
    for e in entrees:
        # --> LE SPLIT SE FAIT ICI, en fonction de 'e' <--
        df_train, df_test = get_splits(df_full, entree=e)
        
        for r in residuelles:
            for i in inters:
                print(f"\n=== Modèle : {e}_{r}_{i} ===")
                
                # Optimisation Optuna
                optimize(df_train, entree=e, residuelle=r, inter=i, n_trials=50)
                
                # Évaluation et sauvegarde des perfs
                evaluator(df_train, df_test, entree=e, residuelle=r, inter=i)

if __name__ == "__main__":
    main()