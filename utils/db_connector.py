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
    """
    try:
        # Supprimer d'abord FactAnomaliesDetail (enfant) puis DimRecurrentProblems (parent)
        conn.execute(text("DELETE FROM FactAnomaliesDetail"))
        conn.execute(text("DELETE FROM DimRecurrentProblems"))
        
        print("Anciennes données supprimées")
        return True
    except Exception as e:
        print(f"Erreur lors de la suppression des données : {e}")
        # Continuer même si l'effacement échoue
        return True

def save_analysis_results(df_anomalies: pd.DataFrame, cluster_results: pd.DataFrame):
    """
    Sauvegarde les résultats d'analyse dans DimRecurrentProblems puis FactAnomaliesDetail.
    """
    if engine is None:
        print("Erreur: Moteur de base de données non initialisé.")
        return False
        
    try:
        with engine.connect() as conn:
            
            # Supprimer les anciennes données
            delete_old_data(conn)
                
            if cluster_results is not None and not cluster_results.empty:
                # Préparer les données pour DimRecurrentProblems
                clusters_to_save = cluster_results[[
                    'ProblemNameGroup', 'ClusterID', 'KeywordMatch', 'RecurrenceCount', 'CategoryID'
                ]].copy()
                
                # S'assurer que ClusterID est unique
                clusters_to_save['ClusterID'] = range(1, len(clusters_to_save) + 1)
                
                clusters_to_save['CategoryID'] = clusters_to_save['CategoryID'].replace({np.nan: None})
                
                # Insérer dans DimRecurrentProblems
                clusters_to_save.to_sql(
                    'DimRecurrentProblems', 
                    conn, 
                    if_exists='append', 
                    index=False
                )
                print(f"{len(clusters_to_save)} problèmes récurrents sauvegardés dans DimRecurrentProblems")
                
                # Créer un mapping pour mettre à jour les ClusterID dans df_anomalies
                cluster_mapping = dict(zip(cluster_results['ClusterID'], clusters_to_save['ClusterID']))
                
                # Mettre à jour les ClusterID dans df_anomalies
                if not df_anomalies.empty:
                    df_anomalies['ClusterID'] = df_anomalies['ClusterID'].map(cluster_mapping).fillna(0)
            
            if not df_anomalies.empty:
                # Préparer les données pour FactAnomaliesDetail
                anomalies_to_save = df_anomalies[[
                    'TicketID', 'FactKey', 'AssigneeEmployeeKey', 'AssigneeFullName',
                    'TicketNote', 'EmployeeAvgScore', 'ScoreSemantique', 'NoteSemantique',
                    'ScoreConcordance', 'NoteConcordance', 'TempsHeures', 'NoteTemporelle',
                    'Statut', 'ClusterID', 'CategoryID'  # Ajout de CategoryID
                ]].copy()
                
                # Remplir les valeurs NaN
                anomalies_to_save = anomalies_to_save.fillna({
                    'TicketNote': 0, 'EmployeeAvgScore': 0, 
                    'ScoreSemantique': 0, 'NoteSemantique': 0,
                    'ScoreConcordance': 0, 'NoteConcordance': 0,
                    'TempsHeures': 0, 'NoteTemporelle': 0,
                    'Statut': 'Non Déterminé', 'ClusterID': 0,
                    'CategoryID': 0  # Ajout de CategoryID
                })
                
                # Insérer dans FactAnomaliesDetail
                anomalies_to_save.to_sql(
                    'FactAnomaliesDetail', 
                    conn, 
                    if_exists='append', 
                    index=False
                )
                print(f"{len(anomalies_to_save)} anomalies sauvegardées dans FactAnomaliesDetail")
            
            # Valider la transaction
            conn.commit()
            print("Sauvegarde terminée avec succès")
            return True
            
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")
        try:
            conn.rollback()
        except:
            pass
        return False