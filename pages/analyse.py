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
            DD.FullDate AS DateCreation  -- AJOUTÉ ICI
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