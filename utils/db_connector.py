import pyodbc
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy.exc import DBAPIError

# --- CONFIGURATION DE LA CONNEXION ---
SERVER_NAME = 'CIYGSG9030DK\\SQLEXPRESS' 
DATABASE_NAME = 'GLPI_DWH'
DRIVER = 'ODBC Driver 17 for SQL Server'

def get_db_connection_url():
    """Crée l'URL de connexion pour SQLAlchemy (avec connexion Windows trusted)."""
    return f"mssql+pyodbc://@{SERVER_NAME}/{DATABASE_NAME}?driver={DRIVER}&trusted_connection=yes"

def get_db_connection():
    """Crée une connexion directe pyodbc (pour les opérations unitaires si nécessaire)."""
    try:
        conn_str = f'DRIVER={DRIVER};SERVER={SERVER_NAME};DATABASE={DATABASE_NAME};Trusted_Connection=yes;'
        return pyodbc.connect(conn_str)
    except Exception as e:
        print(f"Erreur de connexion pyodbc: {e}")
        return None

# Initialisation du moteur SQLAlchemy une seule fois
try:
    engine = create_engine(get_db_connection_url(), connect_args={'autocommit': True})
    print("Connexion à la base de données établie")
except Exception as e:
    print(f"Erreur de connexion SQLAlchemy : {e}")
    engine = None

def load_categories_data():
    """
    Charge les catégories depuis DimCategory pour le clustering.
    """
    if engine is None:
        return pd.DataFrame()
    
    try:
        query = text("""
        SELECT 
            CategoryID, 
            CategoryName, 
            ISNULL(CategoryFullName, CategoryName) AS Description,
            ParentCategoryID
        FROM DimCategory
        WHERE CategoryID IS NOT NULL
        ORDER BY CategoryName
        """)
        
        df = pd.read_sql(query, engine)
        print(f"{len(df)} catégories chargées depuis DimCategory")
        return df
        
    except Exception as e:
        print(f"Erreur chargement catégories: {e}")
        return pd.DataFrame()

def load_data_for_analysis():
    """
    Charge les données nécessaires pour l'analyse des anomalies (tickets, descriptions, durée).
    """
    if engine is None:
        print("Erreur: Moteur de base de données non initialisé.")
        return pd.DataFrame()
    
    try:
        query = text("""
        SELECT 
            FTP.FactKey,
            FTP.TicketID,
            FTP.AssigneeEmployeeKey,
            COALESCE(DE.UserFirstname + ' ' + DE.RealName, FTP.AssigneeFullName) AS AssigneeFullName,
            FTP.ProblemDescription,
            FTP.SolutionContent,
            FTP.ResolutionDurationSec,
            DD.FullDate AS DateCreation
        FROM FactTicketPerformance FTP
        JOIN DimDate DD ON FTP.DateCreationKey = DD.DateKey
        LEFT JOIN DimEmployee DE ON FTP.AssigneeEmployeeKey = DE.EmployeeKey
        WHERE FTP.ProblemDescription IS NOT NULL 
          AND FTP.SolutionContent IS NOT NULL
          AND FTP.ResolutionDurationSec IS NOT NULL
          AND FTP.AssigneeFullName IS NOT NULL
        ORDER BY DD.FullDate DESC
        """)
        
        df = pd.read_sql(query, engine)
        
        if not df.empty:
            df['TempsHeures'] = df['ResolutionDurationSec'] / 3600.0
            df['TempsHeures'] = df['TempsHeures'].round(2)
            
        print(f"Données chargées : {len(df)} tickets")
        return df
        
    except DBAPIError as e:
        print(f"Erreur de chargement des données: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Erreur de chargement des données: {e}")
        return pd.DataFrame()

def delete_old_data(conn):
    """
    Supprime les anciennes données des tables de faits et de dimension.
    Utilise TRUNCATE pour réinitialiser les identifiants.
    """
    try:
        # Utiliser TRUNCATE TABLE pour réinitialiser les ID auto-incrémentés
        conn.execute(text("TRUNCATE TABLE FactAnomaliesDetail"))
        conn.execute(text("TRUNCATE TABLE DimRecurrentProblems"))
        
        print("Tables vidées et identifiants réinitialisés")
        return True
    except Exception as e:
        # Si TRUNCATE échoue (à cause des contraintes de clé étrangère), utiliser DELETE
        try:
            conn.execute(text("DELETE FROM FactAnomaliesDetail"))
            conn.execute(text("DELETE FROM DimRecurrentProblems"))
            
            # Réinitialiser les séquences d'identifiants
            conn.execute(text("DBCC CHECKIDENT ('DimRecurrentProblems', RESEED, 0)"))
            conn.execute(text("DBCC CHECKIDENT ('FactAnomaliesDetail', RESEED, 0)"))
            
            print("Anciennes données supprimées et identifiants réinitialisés")
            return True
        except Exception as e2:
            conn.rollback()
            print(f"Erreur lors de la suppression des données : {e2}")
            return False

def save_analysis_results(df_anomalies: pd.DataFrame, cluster_results: pd.DataFrame):
    """
    Sauvegarde les résultats d'analyse dans DimRecurrentProblems puis FactAnomaliesDetail.
    """
    if engine is None:
        print("Erreur: Moteur de base de données non initialisé.")
        return False
        
    try:
        with engine.connect() as conn:
            transaction = conn.begin()
            
            try:
                if not delete_old_data(conn):
                    transaction.rollback()
                    return False
                    
                if cluster_results is not None and not cluster_results.empty:
                    clusters_to_save = cluster_results[[
                        'ProblemNameGroup', 'ClusterID', 'KeywordMatch', 'RecurrenceCount', 'CategoryID'
                    ]].copy()
                    
                    # Réinitialiser les ClusterID pour éviter les doublons
                    clusters_to_save['ClusterID'] = range(1, len(clusters_to_save) + 1)
                    
                    clusters_to_save['CategoryID'] = clusters_to_save['CategoryID'].replace({np.nan: None})
                    
                    clusters_to_save.to_sql(
                        'DimRecurrentProblems', 
                        conn, 
                        if_exists='append', 
                        index=False
                    )
                    print(f"{len(clusters_to_save)} problèmes récurrents sauvegardés dans DimRecurrentProblems")
                    
                    # Mettre à jour les ClusterID dans df_anomalies pour correspondre
                    if not df_anomalies.empty:
                        # Créer un mapping entre les anciens et nouveaux ClusterID
                        cluster_mapping = dict(zip(cluster_results['ClusterID'], clusters_to_save['ClusterID']))
                        df_anomalies['ClusterID'] = df_anomalies['ClusterID'].map(cluster_mapping).fillna(0)
                
                if not df_anomalies.empty:
                    
                    anomalies_to_save = df_anomalies[[
                        'TicketID', 'FactKey', 'AssigneeEmployeeKey', 'AssigneeFullName',
                        'TicketNote', 'EmployeeAvgScore', 'ScoreSemantique', 'NoteSemantique',
                        'ScoreConcordance', 'NoteConcordance', 'TempsHeures', 'NoteTemporelle',
                        'Statut', 'ClusterID'
                    ]].copy()
                    
                    anomalies_to_save = anomalies_to_save.fillna({
                        'TicketNote': 0, 'EmployeeAvgScore': 0, 
                        'ScoreSemantique': 0, 'NoteSemantique': 0,
                        'ScoreConcordance': 0, 'NoteConcordance': 0,
                        'TempsHeures': 0, 'NoteTemporelle': 0,
                        'Statut': 'Non Déterminé', 'ClusterID': 0
                    })
                    
                    anomalies_to_save.to_sql(
                        'FactAnomaliesDetail', 
                        conn, 
                        if_exists='append', 
                        index=False
                    )
                    print(f"{len(anomalies_to_save)} anomalies sauvegardées dans FactAnomaliesDetail")
                
                transaction.commit()
                print("Sauvegarde terminée avec succès")
                return True
                
            except Exception as e:
                transaction.rollback()
                print(f"Erreur lors de la sauvegarde: {e}")
                return False
            
    except Exception as e:
        print(f"Erreur de connexion lors de la sauvegarde: {e}")
        return False