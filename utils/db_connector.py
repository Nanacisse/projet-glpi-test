import pyodbc
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime

# Configuration de la connexion
SERVER_NAME = 'CIYGSG9030DK\SQLEXPRESS'  # À remplacer
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
        print(f"❌ Erreur de connexion pyodbc: {e}")
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
            FTP.AssigneeFullName,  -- CORRECTION: On prend directement AssigneeFullName de FactTicketPerformance
            FTP.ProblemDescription,
            FTP.SolutionContent,
            FTP.ResolutionDurationSec,
            DD.FullDate AS DateCreation
        FROM FactTicketPerformance FTP
        JOIN DimDate DD ON FTP.DateCreationKey = DD.DateKey
        WHERE FTP.ProblemDescription IS NOT NULL 
          AND FTP.SolutionContent IS NOT NULL
          AND FTP.ResolutionDurationSec IS NOT NULL
          AND FTP.AssigneeFullName IS NOT NULL  -- S'assurer qu'on a un nom
        ORDER BY DD.FullDate DESC
        """)
        
        df = pd.read_sql(query, engine)
        
        if not df.empty:
            # Conversion de la durée de résolution en heures
            df['TempsHeures'] = df['ResolutionDurationSec'] / 3600.0
            print(f"Chargement de {len(df)} tickets avec employés assignés")
            print(f"Colonnes chargées: {df.columns.tolist()}")
            print(f"FactKey présent: {'FactKey' in df.columns}")
            print(f"AssigneeFullName présent: {'AssigneeFullName' in df.columns}")
            print(f"Échantillon des noms COMPLETS: {df['AssigneeFullName'].head(5).tolist()}")
        else:
            print("❌ Aucune donnée chargée depuis la base")
            
        return df
        
    except Exception as e:
        print(f"❌ Erreur de chargement des données: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def save_analysis_results(df_anomalies, cluster_results=None):
    """
    Sauvegarde les résultats d'analyse dans FactAnomaliesDetail et DimRecurrentProblems.
    """
    try:
        engine = create_engine(get_db_connection_url())
        
        # 1. Sauvegarde dans FactAnomaliesDetail
        if not df_anomalies.empty:
            # VÉRIFICATION CRITIQUE : S'assurer que FactKey existe
            if 'FactKey' not in df_anomalies.columns:
                print("❌ ERREUR: Colonne FactKey manquante dans les données d'anomalies")
                # Créer une FactKey temporaire si elle n'existe pas
                df_anomalies['FactKey'] = df_anomalies.index
            
            # DEBUG: Vérifier les noms avant sauvegarde
            print(f"Vérification des noms avant sauvegarde:")
            if 'AssigneeFullName' in df_anomalies.columns:
                print(f"   Échantillon AssigneeFullName: {df_anomalies['AssigneeFullName'].head(3).tolist()}")
            else:
                print("❌ AssigneeFullName manquant dans df_anomalies")
            
            # Préparer les données pour l'insertion
            anomalies_to_save = df_anomalies[[
                'TicketID', 'FactKey', 'AssigneeEmployeeKey', 'AssigneeFullName',
                'TicketNote', 'EmployeeAvgScore', 'ScoreSemantique', 'ScoreConcordance',
                'TempsHeures', 'TempsMoyenHeures', 'EcartTypeHeures', 'ScoreTemporel',
                'AnomalieTemporelle', 'Statut', 'AnomalyDescription'
            ]].copy()
            
            # Nettoyer et valider les données
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
            
            # DEBUG: Afficher les premières lignes pour vérification
            print(f" Données à sauvegarder dans FactAnomaliesDetail:")
            print(anomalies_to_save[['TicketID', 'AssigneeFullName', 'Statut']].head(3))
            
            # Vérifier les doublons avant insertion
            with engine.connect() as conn:
                # Vérifier si la table existe et a des données récentes
                try:
                    existing_count = pd.read_sql(
                        text("SELECT COUNT(*) as count FROM FactAnomaliesDetail"), 
                        conn
                    ).iloc[0]['count']
                    print(f"FactAnomaliesDetail contient déjà {existing_count} enregistrements")
                    
                    # Vérifier les doublons basés sur TicketID
                    existing_tickets = pd.read_sql(
                        text("SELECT DISTINCT TicketID FROM FactAnomaliesDetail"), 
                        conn
                    )['TicketID'].tolist()
                    
                    print(f"Tickets existants: {len(existing_tickets)}")
                    
                    # Filtrer les doublons
                    nouvelles_anomalies = anomalies_to_save[~anomalies_to_save['TicketID'].isin(existing_tickets)]
                    print(f"Nouvelles anomalies à insérer: {len(nouvelles_anomalies)}")
                    
                except Exception as e:
                    print(f"Erreur vérification des doublons: {e}")
                    nouvelles_anomalies = anomalies_to_save
            
            if not nouvelles_anomalies.empty:
                # Insertion par lot
                nouvelles_anomalies.to_sql(
                    'FactAnomaliesDetail', 
                    engine, 
                    if_exists='append', 
                    index=False,
                    chunksize=100
                )
                print(f"{len(nouvelles_anomalies)} anomalies sauvegardées dans FactAnomaliesDetail")
                
                # VÉRIFICATION : Compter les enregistrements après insertion
                with engine.connect() as conn:
                    new_count = pd.read_sql(
                        text("SELECT COUNT(*) as count FROM FactAnomaliesDetail"), 
                        conn
                    ).iloc[0]['count']
                    print(f"Total après insertion: {new_count} enregistrements")
                    
                    # Vérifier les noms sauvegardés
                    recent_names = pd.read_sql(
                        text("SELECT TOP 5 TicketID, AssigneeFullName FROM FactAnomaliesDetail ORDER BY AnomalyDetailKey DESC"), 
                        conn
                    )
                    print(f" Noms récemment sauvegardés:")
                    for _, row in recent_names.iterrows():
                        print(f"   Ticket {row['TicketID']}: {row['AssigneeFullName']}")
            else:
                print("ℹAucune nouvelle anomalie à sauvegarder (doublons évités)")
        
        # 2. Sauvegarde des clusters dans DimRecurrentProblems
        if cluster_results is not None and not cluster_results.empty:
            print(f"Données clusters à sauvegarder: {len(cluster_results)} clusters")
            
            # Préparer les données des clusters
            clusters_to_save = cluster_results[[
                'ProblemNameGroup', 'ClusterID', 'KeywordMatch', 'RecurrenceCount'
            ]].copy()
            
            # DEBUG
            print("Données clusters:")
            print(clusters_to_save)
            
            # Vérifier les doublons pour DimRecurrentProblems
            with engine.connect() as conn:
                try:
                    # Vider la table avant nouvelle insertion (ou gérer les doublons différemment)
                    conn.execute(text("DELETE FROM DimRecurrentProblems"))
                    print("Anciens clusters supprimés de DimRecurrentProblems")
                except Exception as e:
                    print(f"Impossible de vider DimRecurrentProblems: {e}")
            
            if not clusters_to_save.empty:
                clusters_to_save.to_sql(
                    'DimRecurrentProblems', 
                    engine, 
                    if_exists='append', 
                    index=False
                )
                print(f"{len(clusters_to_save)} problèmes récurrents sauvegardés dans DimRecurrentProblems")
                
                # VÉRIFICATION
                with engine.connect() as conn:
                    cluster_count = pd.read_sql(
                        text("SELECT COUNT(*) as count FROM DimRecurrentProblems"), 
                        conn
                    ).iloc[0]['count']
                    print(f"Total clusters après insertion: {cluster_count}")
            else:
                print("ℹAucun nouveau problème récurrent à sauvegarder")
        else:
            print("❌ Aucune donnée de cluster à sauvegarder")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur critique lors de la sauvegarde: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_tables_data():
    """Vérifie le contenu des tables après analyse"""
    try:
        engine = create_engine(get_db_connection_url())
        
        # Vérifier FactAnomaliesDetail
        anomalies_count = pd.read_sql(
            text("SELECT COUNT(*) as count FROM FactAnomaliesDetail"), 
            engine
        ).iloc[0]['count']
        
        # Vérifier DimRecurrentProblems  
        clusters_count = pd.read_sql(
            text("SELECT COUNT(*) as count FROM DimRecurrentProblems"), 
            engine
        ).iloc[0]['count']
        
        print(f"VÉRIFICATION TABLES:")
        print(f"   FactAnomaliesDetail: {anomalies_count} enregistrements")
        print(f"   DimRecurrentProblems: {clusters_count} enregistrements")
        
        # Afficher quelques noms de FactAnomaliesDetail pour vérification
        if anomalies_count > 0:
            sample_names = pd.read_sql(
                text("SELECT TOP 5 AssigneeFullName FROM FactAnomaliesDetail"), 
                engine
            )
            print(f"Échantillon des noms dans FactAnomaliesDetail:")
            for name in sample_names['AssigneeFullName']:
                print(f"   - {name}")
        
        return anomalies_count > 0 and clusters_count > 0
        
    except Exception as e:
        print(f"Erreur vérification: {e}")
        return False