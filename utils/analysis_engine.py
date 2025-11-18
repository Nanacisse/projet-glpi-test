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
    """Calcule le score sémantique pour les descriptions de solutions."""
    if not nlp or pd.isna(text):
        return 0.0
    
    try:
        text_str = str(text)
        if len(text_str.strip()) == 0:
            return 0.0
            
        doc = nlp(text_str)
        
        # Métriques de qualité sémantique
        total_tokens = len(doc)
        if total_tokens == 0:
            return 0.0
        
        # 1. Longueur appropriée
        length_score = min(20, total_tokens * 0.5)
        
        # 2. Structure (présence de phrases complètes)
        sentences = list(doc.sents)
        num_sentences = len(sentences)
        structure_score = min(15, num_sentences * 3)
        
        # 3. Richesse lexicale
        non_stop_words = [token for token in doc if token.is_alpha and not token.is_stop]
        lexical_richness = len(non_stop_words) / total_tokens if total_tokens > 0 else 0
        richness_score = lexical_richness * 30
        
        # 4. Cohérence technique
        technical_terms = ['erreur', 'bug', 'problème', 'solution', 'correct', 
                          'réparer', 'installer', 'configurer', 'résoudre', 'dépanner',
                          'incident', 'panne', 'dysfonctionnement', 'technique']
        
        technical_count = sum(1 for token in doc if token.text.lower() in technical_terms)
        technical_score = min(20, technical_count * 2)
        
        # 5. Clarté (faible proportion de mots vides)
        stop_word_ratio = sum(1 for token in doc if token.is_stop) / total_tokens
        clarity_score = (1 - stop_word_ratio) * 15
        
        # Score total
        total_score = length_score + structure_score + richness_score + technical_score + clarity_score
        
        # Pénalités
        penalties = 0
        
        # Pénalité pour mots inconnus
        unknown_words = sum(1 for token in doc if token.is_oov and token.is_alpha)
        penalties += min(10, unknown_words * 2)
        
        # Pénalité pour phrases trop longues
        if sentences:
            avg_sentence_length = total_tokens / len(sentences)
            if avg_sentence_length > 25:
                penalties += 10
        
        final_score = max(0, min(100, total_score - penalties))
        return round(final_score, 2)
        
    except Exception as e:
        print(f"Erreur analyse sémantique: {e}")
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
        print(f"Erreur calcul concordance: {e}")
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
        anomalies.append("Description du problème peu claire")
    
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
    
    # 1. Analyse Sémantique: APPLIQUÉE À LA DESCRIPTION DES SOLUTIONS
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
                        # Extraire les mots-clés communs
                        sample_description = cluster_descriptions[0]
                        
                        cluster_data.append({
                            'ProblemNameGroup': f"Cluster_{cluster_id}",
                            'ClusterID': cluster_id,
                            'KeywordMatch': f"Cluster de {len(cluster_descriptions)} problèmes similaires",
                            'RecurrenceCount': len(cluster_descriptions)
                        })
                
                cluster_results = pd.DataFrame(cluster_data)
                print(f"Clustering terminé: {n_clusters} clusters identifiés")
                
        except Exception as e:
            print(f"Erreur clustering: {e}")
            
    return df, cluster_results