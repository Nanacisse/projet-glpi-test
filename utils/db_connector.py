import pyodbc
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy.exc import DBAPIError
import warnings

# Supprimer les warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION DE LA CONNEXION ---
SERVER_NAME = 'CIYGSG9030DK\\SQLEXPRESS' 
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

# Initialisation du moteur SQLAlchemy
try:
    engine = create_engine(get_db_connection_url(), connect_args={'autocommit': True}, pool_pre_ping=True)
    print("Connexion à la base de données établie")
except Exception as e:
    print(f"Erreur de connexion SQLAlchemy : {e}")
    engine = None

def load_categories_data():
    """Charge les catégories depuis DimCategory."""
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
    """Charge les données nécessaires pour l'analyse."""
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
    """Supprime les anciennes données des tables."""
    try:
        # Supprimer d'abord FactAnomaliesDetail (enfant) puis DimRecurrentProblems (parent)
        conn.execute(text("DELETE FROM FactAnomaliesDetail"))
        conn.execute(text("DELETE FROM DimRecurrentProblems"))
        
        print("Anciennes données supprimées")
        return True
    except Exception as e:
        print(f"⚠ Erreur lors de la suppression des données : {e}")
        # Continuer même si l'effacement échoue
        return True

def save_analysis_results(df_anomalies: pd.DataFrame, cluster_results: pd.DataFrame):
    """Sauvegarde les résultats d'analyse SANS TempsMoyenHeures."""
    if engine is None:
        print("Erreur: Moteur de base de données non initialisé.")
        return False
        
    try:
        with engine.connect() as conn:
            
            # Supprimer les anciennes données
            delete_old_data(conn)
                
            if cluster_results is not None and not cluster_results.empty:
                # ⭐⭐ CORRECTION : NE PAS INCLURE TempsMoyenHeures ⭐⭐
                # Sélectionner uniquement les colonnes qui existent
                available_columns = ['ProblemNameGroup', 'ClusterID', 'KeywordMatch', 'RecurrenceCount', 'CategoryID']
                
                # Vérifier quelles colonnes sont présentes dans cluster_results
                columns_to_use = [col for col in available_columns if col in cluster_results.columns]
                
                if len(columns_to_use) < 3:
                    print("⚠ Pas assez de colonnes dans cluster_results")
                    columns_to_use = ['ProblemNameGroup', 'ClusterID', 'KeywordMatch']
                
                clusters_to_save = cluster_results[columns_to_use].copy()
                
                # S'assurer que ClusterID est unique et commence à 1
                clusters_to_save['ClusterID'] = range(1, len(clusters_to_save) + 1)
                
                # Gérer les valeurs NULL/Nan pour CategoryID si présent
                if 'CategoryID' in clusters_to_save.columns:
                    clusters_to_save['CategoryID'] = clusters_to_save['CategoryID'].replace({np.nan: None}).fillna(0)
                
                print(f"💾 Préparation {len(clusters_to_save)} clusters...")
                print(f"   Colonnes à sauvegarder: {list(clusters_to_save.columns)}")
                
                # ⭐⭐ NE PAS ajouter TempsMoyenHeures
                # La ligne suivante était problématique :
                # clusters_to_save['TempsMoyenHeures'] = cluster_times  # ⬅️ SUPPRIMER
                
                # Insérer dans DimRecurrentProblems
                clusters_to_save.to_sql(
                    'DimRecurrentProblems', 
                    conn, 
                    if_exists='append', 
                    index=False
                )
                print(f"✅ {len(clusters_to_save)} problèmes récurrents sauvegardés dans DimRecurrentProblems")
                
                # Créer un mapping pour mettre à jour les ClusterID dans df_anomalies
                cluster_mapping = dict(zip(range(len(clusters_to_save)), clusters_to_save['ClusterID']))
                
                # Mettre à jour les ClusterID dans df_anomalies
                if not df_anomalies.empty:
                    df_anomalies['ClusterID'] = df_anomalies['ClusterID'].map(cluster_mapping).fillna(0)
            
            if not df_anomalies.empty:
                # Préparer les données pour FactAnomaliesDetail
                anomalies_to_save = df_anomalies[[
                    'TicketID', 'FactKey', 'AssigneeEmployeeKey', 'AssigneeFullName',
                    'TicketNote', 'EmployeeAvgScore', 'ScoreSemantique', 'NoteSemantique',
                    'ScoreConcordance', 'NoteConcordance', 'TempsHeures', 'NoteTemporelle',
                    'Statut', 'ClusterID', 'CategoryID'
                ]].copy()
                
                # Remplir les valeurs NaN
                anomalies_to_save = anomalies_to_save.fillna({
                    'TicketNote': 0, 'EmployeeAvgScore': 0, 
                    'ScoreSemantique': 0, 'NoteSemantique': 0,
                    'ScoreConcordance': 0, 'NoteConcordance': 0,
                    'TempsHeures': 0, 'NoteTemporelle': 0,
                    'Statut': 'Non Déterminé', 'ClusterID': 0,
                    'CategoryID': 0
                })
                
                # S'assurer que les types de données sont corrects
                anomalies_to_save['TempsHeures'] = pd.to_numeric(anomalies_to_save['TempsHeures'], errors='coerce').fillna(0)
                
                print(f"💾 Préparation {len(anomalies_to_save)} anomalies...")
                
                # Insérer dans FactAnomaliesDetail
                anomalies_to_save.to_sql(
                    'FactAnomaliesDetail', 
                    conn, 
                    if_exists='append', 
                    index=False
                )
                print(f"✅ {len(anomalies_to_save)} anomalies sauvegardées dans FactAnomaliesDetail")
            
            # Valider la transaction
            conn.commit()
            print("🎉 Sauvegarde terminée avec succès!")
            return True
            
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")
        import traceback
        traceback.print_exc()
        try:
            conn.rollback()
        except:
            pass
        return False