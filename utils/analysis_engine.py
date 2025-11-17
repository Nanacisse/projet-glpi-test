import pandas as pd
import numpy as np
import spacy
from difflib import SequenceMatcher
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import re

# --- Initialisation des ressources IA/NLP ---
try:
    nlp = spacy.load("fr_core_news_sm")
    st_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
except Exception as e:
    print(f"Erreur de chargement des modèles NLP/IA : {e}")
    nlp = None
    st_model = None

# Définition des constantes d'anomalie
SEMAN_THRESHOLD = 0.60  # 60%
CONC_THRESHOLD = 0.40   # 40%
Z_SCORE_THRESHOLD = 2   # |Z| > 2

def calculate_semantique_score(text):
    """Calcule le score sémantique pour les descriptions de SOLUTIONS (employés)."""
    if pd.isna(text) or text is None:
        return 0.0
    
    try:
        text_str = str(text).strip()
        if len(text_str) == 0:
            return 0.0
        
        # Vérifier si spaCy est disponible
        if not nlp:
            return 50.0
            
        doc = nlp(text_str)
        
        # Métriques de qualité sémantique
        total_tokens = len(doc)
        if total_tokens == 0:
            return 0.0
        
        # 1. Score basé sur la longueur (0-40 points)
        length_score = min(40, total_tokens * 2)
        
        # 2. Score basé sur les phrases (0-30 points)
        sentences = list(doc.sents)
        num_sentences = len(sentences)
        structure_score = min(30, num_sentences * 10)
        
        # 3. Score basé sur les mots non-stop (0-30 points)
        non_stop_words = [token for token in doc if token.is_alpha and not token.is_stop]
        lexical_score = min(30, len(non_stop_words) * 2)
        
        # Score total simple
        total_score = length_score + structure_score + lexical_score
        
        # Ajustement pour rester dans 0-100
        final_score = min(100, total_score)
        
        return round(final_score, 2)
        
    except Exception as e:
        return 50.0

def calculate_concordance_score(problem, solution):
    """Calcule le score de concordance entre problème et solution."""
    if pd.isna(problem) or pd.isna(solution):
        return 0.0
    
    try:
        problem_str = str(problem).lower().strip()
        solution_str = str(solution).lower().strip()
        
        if not problem_str or not solution_str:
            return 0.0
        
        # 1. Similarité basique (SequenceMatcher)
        matcher = SequenceMatcher(None, problem_str, solution_str)
        base_similarity = matcher.ratio() * 40
        
        # 2. Présence de mots-clés de résolution
        resolution_keywords = ['résolu', 'corrigé', 'réparé', 'fixé', 'solution', 
                              'résolution', 'terminé', 'complété', 'réussi']
        
        resolution_found = any(keyword in solution_str for keyword in resolution_keywords)
        resolution_score = 20 if resolution_found else 0
        
        # 3. Longueur relative de la solution
        problem_words = len(problem_str.split())
        solution_words = len(solution_str.split())
        
        if problem_words > 0 and solution_words > 0:
            length_ratio = min(1.0, solution_words / problem_words)
            length_score = length_ratio * 20
        else:
            length_score = 0
        
        # 4. Structure de la solution
        solution_has_steps = any(marker in solution_str for marker in ['premièrement', 'ensuite', 'puis', 'étape', 'step'])
        structure_score = 10 if solution_has_steps else 5
        
        # 5. Présence d'indicateurs de complétion
        completion_indicators = any(marker in solution_str for marker in ['terminé', 'fini', 'complété', 'finalisé'])
        completion_score = 10 if completion_indicators else 0
        
        total_score = base_similarity + resolution_score + length_score + structure_score + completion_score
        return min(100, round(total_score, 2))
        
    except Exception as e:
        return 50.0

def calculate_temporal_score(df):
    """Calcule le Z-score temporel et détecte les anomalies temporelles."""
    if df.empty:
        return df
        
    # Calcul des statistiques temporelles
    mean_h = df['TempsHeures'].mean()
    std_h = df['TempsHeures'].std()
    
    df['TempsMoyenHeures'] = round(mean_h, 2)
    df['EcartTypeHeures'] = round(std_h if std_h > 0 else 1.0, 2)
    
    # Calcul du Z-score
    df['ScoreTemporel'] = (df['TempsHeures'] - mean_h) / df['EcartTypeHeures']
    
    # Détection d'anomalie temporelle
    df['AnomalieTemporelle'] = np.where(np.abs(df['ScoreTemporel']) > Z_SCORE_THRESHOLD, 'Oui', 'Non')
    
    return df

def determine_final_status(row):
    """Détermine le statut final basé sur les 3 scores."""
    sem_ok = row['ScoreSemantique'] >= SEMAN_THRESHOLD * 100
    conc_ok = row['ScoreConcordance'] >= CONC_THRESHOLD * 100
    temp_ok = row['AnomalieTemporelle'] == 'Non'
    
    # Logique des statuts
    if sem_ok and conc_ok and temp_ok:
        return 'OK'
    elif sem_ok and conc_ok and not temp_ok:
        return 'Anomalie de Temps'
    elif sem_ok and not conc_ok and temp_ok:
        return 'Anomalie de Concordance'
    elif not sem_ok and conc_ok and temp_ok:
        return 'Anomalie Sémantique'
    
    # Cas d'anomalies multiples
    num_anomalies = sum([not sem_ok, not conc_ok, not temp_ok])
    
    if num_anomalies >= 2:
        return 'Multiples Anomalies'
    
    return 'Anomalie Indéterminée'

def calculate_ticket_note(row):
    """Calcule la Note de Ticket (Base 10) par pénalité."""
    status = row['Statut']
    
    if status == 'OK':
        return 10.0
    elif status == 'Anomalie de Temps':
        return 7.0
    elif status == 'Multiples Anomalies':
        return 5.0
    elif status == 'Anomalie Sémantique':
        return 8.0
    elif status == 'Anomalie de Concordance':
        return 8.0
    else:
        return 6.0

def generate_anomaly_description(row):
    """Génère une description de l'anomalie basée sur les scores."""
    anomalies = []
    
    if row['ScoreSemantique'] < SEMAN_THRESHOLD * 100:
        anomalies.append("Description de la solution peu claire")
    
    if row['ScoreConcordance'] < CONC_THRESHOLD * 100:
        anomalies.append("Solution peu pertinente par rapport au problème")
    
    if row['AnomalieTemporelle'] == 'Oui':
        anomalies.append("Temps de résolution anormal")
    
    if anomalies:
        return "; ".join(anomalies)
    else:
        return "Aucune anomalie détectée"

def run_full_analysis(df):
    """Exécute l'intégralité du pipeline d'analyse IA."""
    if df.empty:
        return df, None
    
    print(f"🔧 Début de l'analyse sur {len(df)} tickets assignés")
    
    # Vérifier que FactKey existe
    if 'FactKey' not in df.columns:
        df['FactKey'] = df.index
    
    # 1. Analyse Sémantique
    df['ScoreSemantique'] = df['SolutionContent'].apply(calculate_semantique_score)
    
    # 2. Analyse de Concordance
    df['ScoreConcordance'] = df.apply(
        lambda row: calculate_concordance_score(row['ProblemDescription'], row['SolutionContent']),
        axis=1
    )
    
    # 3. Analyse Temporelle
    df = calculate_temporal_score(df.copy())
    
    # 4. Détermination du Statut Final
    df['Statut'] = df.apply(determine_final_status, axis=1)
    
    # 5. Calcul de la Note de Ticket
    df['TicketNote'] = df.apply(calculate_ticket_note, axis=1)
    
    # 6. Calcul de la Moyenne Employé
    if 'AssigneeEmployeeKey' in df.columns:
        employee_avg = df.groupby('AssigneeEmployeeKey')['TicketNote'].mean().round(2)
        df['EmployeeAvgScore'] = df['AssigneeEmployeeKey'].map(employee_avg)
        df['EmployeeAvgScore'] = df['EmployeeAvgScore'].fillna(df['TicketNote'])
    else:
        df['EmployeeAvgScore'] = df['TicketNote']
    
    # 7. Génération de la description d'anomalie
    df['AnomalyDescription'] = df.apply(generate_anomaly_description, axis=1)
    
    # 8. Clustering pour problèmes récurrents
    cluster_results = None
    if st_model is not None and 'ProblemDescription' in df.columns:
        try:
            descriptions = df['ProblemDescription'].astype(str).tolist()
            if descriptions:
                embeddings = st_model.encode(descriptions, show_progress_bar=False)
                n_clusters = min(10, max(2, len(descriptions) // 5))
                clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
                cluster_labels = clustering_model.fit_predict(embeddings)
                
                df['ClusterID'] = cluster_labels
                
                # Préparer les résultats pour DimRecurrentProblems
                cluster_data = []
                for cluster_id in range(n_clusters):
                    cluster_descriptions = df[df['ClusterID'] == cluster_id]['ProblemDescription'].tolist()
                    if cluster_descriptions:
                        sample_description = cluster_descriptions[0]
                        words = sample_description.split()[:5]
                        keywords = " ".join(words)
                        
                        cluster_data.append({
                            'ProblemNameGroup': f"Cluster_{cluster_id}",
                            'ClusterID': cluster_id,
                            'KeywordMatch': keywords,
                            'RecurrenceCount': len(cluster_descriptions)
                        })
                
                cluster_results = pd.DataFrame(cluster_data)
                
        except Exception as e:
            df['ClusterID'] = 0
            cluster_results = None

    print(f"✅ Analyse terminée: {len(df)} tickets analysés")
    return df, cluster_results