import pyodbc
import pandas as pd
from sqlalchemy import create_engine, text

# --- CONFIGURATION DE LA BASE DE DONNÉES ---

# Correction du SyntaxWarning: L'utilisation de '\\' est nécessaire en Python
SERVER_NAME = 'CIYGSG9030DK\\SQLEXPRESS' 
DATABASE_NAME = 'GLPI_DWH'
DRIVER = '{ODBC Driver 17 for SQL Server}'
CONNECTION_STRING = f'DRIVER={DRIVER};SERVER={SERVER_NAME};DATABASE={DATABASE_NAME};Trusted_Connection=yes;'

# Création du moteur SQLAlchemy
try:
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={CONNECTION_STRING}")
except Exception as e:
    print(f"Erreur de connexion SQLAlchemy : {e}")
    engine = None

# --- FONCTIONS DE GESTION DES DONNÉES ---

def load_data_for_analysis():
    """Charge les données brutes nécessaires à l'analyse depuis la vue FactTicketPerformance."""
    if engine is None:
        return pd.DataFrame()

    try:
        # Requête pour charger les données (ajustez si nécessaire)
        query = "SELECT * FROM FactTicketPerformance WHERE AssignedDate IS NOT NULL AND SolvedDate IS NOT NULL"
        data = pd.read_sql(query, engine)
        print(f"Données chargées : {len(data)} lignes.")
        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()


def delete_old_data(conn):
    """Supprime les anciennes données des tables de faits et de dimension."""
    try:
        # IMPORTANT : Supprimer d'abord la table de FAITS
        conn.execute(text("DELETE FROM FactAnomaliesDetail"))
        
        # Ensuite, supprimer la table de DIMENSION
        conn.execute(text("DELETE FROM DimRecurrentProblems"))
        
        conn.commit()
        print("Anciennes données supprimées des tables de faits et de dimension.")
        return True
    except Exception as e:
        conn.rollback()
        print(f"Erreur lors de la suppression des données : {e}")
        return False


def save_analysis_results(df_anomalies: pd.DataFrame, cluster_results: pd.DataFrame):
    """
    Sauvegarde les résultats de l'analyse dans GLPI_DWH.
    La table DimRecurrentProblems (Dimension) DOIT être insérée avant 
    FactAnomaliesDetail (Fait) pour respecter la contrainte de clé étrangère.
    """
    if engine is None:
        print("Erreur: Moteur de base de données non initialisé.")
        return False
        
    try:
        with engine.connect() as conn:
            
            # 1. Suppression des anciennes données
            if not delete_old_data(conn):
                return False

            # 2. Insertion de la table de DIMENSION (DimRecurrentProblems)
            # Ceci DOIT être fait en premier pour que FactAnomaliesDetail puisse y faire référence.
            print("Début de l'insertion dans DimRecurrentProblems (Dimension)...")
            cluster_results.to_sql(
                'DimRecurrentProblems',
                conn, 
                if_exists='append', 
                index=False
            )
            print(f"✅ Insertion réussie dans DimRecurrentProblems: {len(cluster_results)} lignes.")

            # 3. Insertion de la table de FAITS (FactAnomaliesDetail)
            # Ceci DOIT être fait en second.
            print("Début de l'insertion dans FactAnomaliesDetail (Fait)...")
            df_anomalies.to_sql(
                'FactAnomaliesDetail', 
                conn, 
                if_exists='append', 
                index=False
            )
            print(f"✅ Insertion réussie dans FactAnomaliesDetail: {len(df_anomalies)} lignes.")

            conn.commit()
            return True

    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")
        return False