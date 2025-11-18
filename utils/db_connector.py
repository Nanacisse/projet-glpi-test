import pyodbc
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime
 
# Configuration de la connexion
SERVER_NAME = 'CIYGSG9030DK\SQLEXPRESS' 
DATABASE_NAME = 'GLPI_DWH'
DRIVER = 'ODBC Driver 17 for SQL Server'
 
def get_db_connection_url():
    """Crée l'URL de connexion pour SQLAlchemy."""
    return f"mssql+pyodbc://@{SERVER_NAME}/{DATABASE_NAME}?driver={DRIVER}&trusted_connection=yes"
 
def get_db_connection():
    """Crée une connexion directe pyodbc."""
    try:
        conn_str = f'DRIVER={DRIVER};SERVER={SERVER_NAME};DATABASE={DATABASE_NAME};Trusted_Connection=yes;'
        return pyodbc.connect(conn_str)
    except Exception as e:
        print(f"Erreur de connexion pyodbc: {e}")
        return None
 
def load_data_for_analysis():
    """Charge les données nécessaires pour l'analyse depuis FactTicketPerformance."""
    try:
        engine = create_engine(get_db_connection_url())
        
        query = text("""
        SELECT 
            FTP.FactKey,
            FTP.TicketID,
            FTP.AssigneeEmployeeKey,
            -- Tente de concaténer Prénom et Nom de la table DimEmployee pour garantir le nom complet
            COALESCE(DE.FirstName + ' ' + DE.LastName, FTP.AssigneeFullName) AS AssigneeFullName,
            FTP.ProblemDescription,
            FTP.SolutionContent,
            FTP.ResolutionDurationSec,
            DD.FullDate AS DateCreation
        FROM FactTicketPerformance FTP
        JOIN DimDate DD ON FTP.DateCreationKey = DD.DateKey
        -- Jointure hypothétique pour le nom complet, à vérifier avec votre schéma
        LEFT JOIN DimEmployee DE ON FTP.AssigneeEmployeeKey = DE.EmployeeKey
        WHERE FTP.ProblemDescription IS NOT NULL 
          AND FTP.SolutionContent IS NOT NULL
          AND FTP.ResolutionDurationSec IS NOT NULL
          AND FTP.AssigneeFullName IS NOT NULL
        ORDER BY DD.FullDate DESC
        """)
        
        df = pd.read_sql(query, engine)
        
        if not df.empty:
            # Conversion de la durée de résolution en heures
            df['TempsHeures'] = df['ResolutionDurationSec'] / 3600.0
            
        return df
        
    except Exception as e:
        print(f"Erreur de chargement des données: {e}")
        return pd.DataFrame()
 
def save_analysis_results(df_anomalies, cluster_results=None):
    """
    Sauvegarde les résultats d'analyse dans FactAnomaliesDetail et DimRecurrentProblems.
    """
    try:
        engine = create_engine(get_db_connection_url())
        
        # VIDER LES TABLES AVANT NOUVELLE ANALYSE
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM FactAnomaliesDetail"))
            conn.execute(text("DELETE FROM DimRecurrentProblems"))
            conn.commit()
            print("Anciennes données supprimées")
        
        # 1. Sauvegarde dans FactAnomaliesDetail
        if not df_anomalies.empty:
            # Vérifier que FactKey existe
            if 'FactKey' not in df_anomalies.columns:
                df_anomalies['FactKey'] = df_anomalies.index
            
            # Préparer les données pour l'insertion
            anomalies_to_save = df_anomalies[[
                'TicketID', 'FactKey', 'AssigneeEmployeeKey', 'AssigneeFullName',
                'TicketNote', 'EmployeeAvgScore', 'ScoreSemantique', 'ScoreConcordance',
                'TempsHeures', 'TempsMoyenHeures', 'EcartTypeHeures', 'ScoreTemporel',
                'AnomalieTemporelle', 'Statut', 'AnomalyDescription'
            ]].copy()
            
            # Nettoyer les données
            anomalies_to_save = anomalies_to_save.fillna({
                'TicketNote': 0,
                'EmployeeAvgScore': 0,
                'ScoreSemantique': 0,
                'ScoreConcordance': 0,
                'TempsHeures': 0,
                'TempsMoyenHeures': 0,
                'EcartTypeHeures': 0,
                'ScoreTemporel': 0,
                'AnomalieTemporelle': 'Non',
                'Statut': 'Non Déterminé',
                'AnomalyDescription': 'Aucune description'
            })
            
            # INSERTION DIRECTE
            anomalies_to_save.to_sql(
                'FactAnomaliesDetail', 
                engine, 
                if_exists='append', 
                index=False
            )
            print(f"{len(anomalies_to_save)} anomalies sauvegardées dans FactAnomaliesDetail")
        
        # 2. Sauvegarde des clusters dans DimRecurrentProblems
        if cluster_results is not None and not cluster_results.empty:
            # Préparer les données des clusters
            clusters_to_save = cluster_results[[
                'ProblemNameGroup', 'ClusterID', 'KeywordMatch', 'RecurrenceCount'
            ]].copy()
            
            # INSERTION DIRECTE
            clusters_to_save.to_sql(
                'DimRecurrentProblems', 
                engine, 
                if_exists='append', 
                index=False
            )
            print(f"{len(clusters_to_save)} problèmes récurrents sauvegardés dans DimRecurrentProblems")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")
        return False