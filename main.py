import importlib
import os

def run_all_models():
    # Liste des modèles à exécuter
    models = [
        'models.transformers.cpu',
        'models.rnn.cpu',
        'models.cnn.cpu',
        'models.mlp.cpu',
        # 'models.transformers.gpu',
        # 'models.rnn.gpu',
        # 'models.cnn.gpu',
        # 'models.mlp.gpu',
    ]
    
    # Exécuter chaque modèle
    for model in models:
        print(f"\n{'='*50}")
        print(f"Exécution de {model}")
        print(f"{'='*50}\n")
        
        try:
            # Importer dynamiquement le module
            module = importlib.import_module(model)
            
            # Exécuter la fonction main
            if hasattr(module, 'main'):
                module.main()
            else:
                print(f"Pas de fonction main trouvée dans {model}")
                
        except Exception as e:
            print(f"Erreur lors de l'exécution de {model}: {str(e)}")
            continue

if __name__ == "__main__":
    # Créer le dossier exports s'il n'existe pas
    os.makedirs("exports", exist_ok=True)
    
    # Lancer tous les modèles
    run_all_models()