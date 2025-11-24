import pyodbc
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime

# --- CONFIGURATION DE LA CONNEXION ---
# Correction du SyntaxWarning en utilisant double barre oblique inverse '\\'
SERVER_NAME = 'CIYGSG9030DK\\SQLEXPRESS' 
DATABASE_NAME = 'GLPI_DWH'
DRIVER = 'ODBC Driver 17 for SQL Server'

def get_db_connection_url():
    """Crée l'URL de connexion pour SQLAlchemy."""
    # Utilisation du nom de serveur corrigé
    return f"mssql+pyodbc://@{SERVER_NAME}/{DATABASE_NAME}?driver={DRIVER}&trusted_connection=yes"

def get_db_connection():
    """Crée une connexion directe pyodbc."""
    try:
        conn_str = f'DRIVER={DRIVER};SERVER={SERVER_NAME};DATABASE={DATABASE_NAME};Trusted_Connection=yes;'
        return pyodbc.connect(conn_str)
    except Exception as e:
        print(f"Erreur de connexion pyodbc: {e}")
        return None

# Initialisation du moteur SQLAlchemy une seule fois
try:
    engine = create_engine(get_db_connection_url())
except Exception as e:
    print(f"Erreur de connexion SQLAlchemy : {e}")
    engine = None

def load_data_for_analysis():
    """Charge les données nécessaires pour l'analyse en utilisant votre requête détaillée."""
    if engine is None:
        print("Erreur: Moteur de base de données non initialisé.")
        return pd.DataFrame()
    
    try:
        # Votre requête SQL d'origine avec les JOINs
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
            # Conversion de la durée de résolution en heures (TempsHeures)
            df['TempsHeures'] = df['ResolutionDurationSec'] / 3600.0
            
        print(f"Données chargées : {len(df)} lignes.")
        return df
        
    except Exception as e:
        print(f"Erreur de chargement des données: {e}")
        return pd.DataFrame()

def delete_old_data(conn):
    """Supprime les anciennes données des tables de faits et de dimension dans le bon ordre."""
    try:
        # Ordre de suppression : FAIT (dépendant) avant DIMENSION (référencée)
        conn.execute(text("DELETE FROM FactAnomaliesDetail"))
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
    Sauvegarde les résultats d'analyse dans DimRecurrentProblems puis FactAnomaliesDetail.
    """
    if engine is None:
        print("Erreur: Moteur de base de données non initialisé.")
        return False
        
    try:
        with engine.connect() as conn:
            
            # 1. Suppression des anciennes données
            if not delete_old_data(conn):
                return False
                
            # 2. Sauvegarde des clusters dans DimRecurrentProblems - ORDRE CORRECT : DIMENSION EN PREMIER
            if cluster_results is not None and not cluster_results.empty:
                clusters_to_save = cluster_results[[
                    'ProblemNameGroup', 'ClusterID', 'KeywordMatch', 'RecurrenceCount'
                ]].copy()
                
                clusters_to_save.to_sql(
                    'DimRecurrentProblems', 
                    conn, 
                    if_exists='append', 
                    index=False
                )
                print(f"{len(clusters_to_save)} problèmes récurrents sauvegardés dans DimRecurrentProblems")
            
            # 3. Sauvegarde dans FactAnomaliesDetail - ORDRE CORRECT : FAIT EN SECOND
            if not df_anomalies.empty:
                # Si 'FactKey' n'est pas déjà dans le DataFrame (il devrait l'être par load_data_for_analysis, mais sécurité)
                if 'FactKey' not in df_anomalies.columns:
                    df_anomalies['FactKey'] = df_anomalies.index
                
                # CORRECTION : Ajout explicite de la colonne 'ClusterID' pour la clé étrangère
                anomalies_to_save = df_anomalies[[
                    'TicketID', 'FactKey', 'AssigneeEmployeeKey', 'AssigneeFullName',
                    'TicketNote', 'EmployeeAvgScore', 'ScoreSemantique', 'ScoreConcordance',
                    'TempsHeures', 'TempsMoyenHeures', 'EcartTypeHeures', 'ScoreTemporel',
                    'AnomalieTemporelle', 'Statut', 'AnomalyDescription', 
                    'ClusterID' # NOUVELLE COLONNE AJOUTÉE ICI
                ]].copy()
                
                # Remplacement des NaN selon votre logique
                anomalies_to_save = anomalies_to_save.fillna({
                    'TicketNote': 0, 'EmployeeAvgScore': 0, 'ScoreSemantique': 0,
                    'ScoreConcordance': 0, 'TempsHeures': 0, 'TempsMoyenHeures': 0,
                    'EcartTypeHeures': 0, 'ScoreTemporel': 0, 'AnomalieTemporelle': 'Non',
                    'Statut': 'Non Déterminé', 'AnomalyDescription': 'Aucune description',
                    'ClusterID': 0 # S'assurer que les tickets sans cluster ont une valeur par défaut valide (ex: 0)
                })
                
                anomalies_to_save.to_sql(
                    'FactAnomaliesDetail', 
                    conn, 
                    if_exists='append', 
                    index=False
                )
                print(f"{len(anomalies_to_save)} anomalies sauvegardées dans FactAnomaliesDetail")
            
            conn.commit()
            return True
            
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")
        return False