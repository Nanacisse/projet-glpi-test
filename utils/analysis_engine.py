import pandas as pd
import numpy as np
import spacy
from difflib import SequenceMatcher
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import re
from collections import Counter
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.fr.stop_words import STOP_WORDS
 
# --- Configuration et Initialisation ---
 
# Définition des constantes d'anomalie
SEMAN_THRESHOLD = 0.60  # 60%
CONC_THRESHOLD = 0.40   # 40%
Z_SCORE_THRESHOLD = 2   # |Z| > 2 (Écart-type)
 
# Initialisation des ressources IA/NLP
nlp = None
st_model = None
try:
    # Charge le modèle français pour la Reconnaissance d'Entités Nommées (REN) et la Lemmatisation
    nlp = spacy.load("fr_core_news_sm") 
    # Charge le modèle Sentence Transformer pour l'encodage sémantique
    st_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
except Exception as e:
    print(f"Erreur de chargement des modèles NLP/IA (Vérifiez les dépendances spacy et sentence-transformers) : {e}")

# Liste des mots génériques d'exclusion (Stop Words classiques + termes de gestion de ticket)
BASE_EXCLUSION_WORDS = set(list(STOP_WORDS) + [
    # Contractions/Lettres courtes à retirer (en minuscule)
    'l', 'd', 'c', 'j', 's', 'm', 'n', 'y',
    # Termes GLPI/Génériques/Mots à ignorer
    'glpi', 'ticket', 'demande', 'facture', 'code', 'version', 'numéro', 'id', 'lien', 'demander',
    'bonjour', 'merci', 'suite', 'depuis', 'date', 'jour', 'mois', 'année', 'heure', 'minute',
    'quelque', 'plusieurs', 'chaque', 'urgent', 'important', 'nécessaire', 
    'besoin', 'utilisateur', 'personne', 'collaborateur', 
    'problème', 'erreur', 'incident', 'souci', 'bug', 'panne', 'fixé', 'résolu', 'corrigé', 'réparé', 
    'falloir', 'pouvoir', 'vouloir', 'devoir', 'savoir', 'impossible', 'bloqué', 'technique', 'général', 'cas', 'via',
    'mise', 'jour', 'création', 'ajout', 'modification', 'supprimer', 'archive', 'vide', 'archivage',
    'compte', 'installation', 'connexion', 'gestion', 'ouvrir', 'fonctionne', 'base', 'faire', 'avoir', 'etre',
    # Termes métier fréquemment lemmatisés ou techniques
    'pc', 'système', 'application', 'logiciel', 'matériel', 'serveur', 'réseau'
])

# --- Fonctions de Pré-traitement Avancé (Lemmatisation et NER) ---

def preprocess_text(text, lemmatize=True, filter_entities=True):
    """
    Nettoie, lemmatise le texte, et filtre les entités nommées (NER).
    Utilisé pour préparer les données avant l'encodage sémantique et le TF-IDF.
    
    Args:
        text (str): La description du problème à nettoyer.
        lemmatize (bool): Applique la lemmatisation.
        filter_entities (bool): Exclut les noms propres (PERSONNE, LIEU, ORG, PROD).
        
    Returns:
        str: Le texte pré-traité.
    """
    if not nlp or pd.isna(text) or not text.strip():
        return ""
        
    text_str = str(text).lower().strip()
    
    # Remplacement de la ponctuation et des chiffres par des espaces pour isoler les mots
    text_str = re.sub(f'[{re.escape(string.punctuation.replace("-", "").replace("'", "") + string.digits)}]', ' ', text_str)
    
    doc = nlp(text_str)
    tokens = []
    
    entities_to_ignore = set()
    if filter_entities:
        # Reconnaissance d'Entités Nommées (NER)
        entity_labels_to_exclude = ['PER', 'ORG', 'PROD', 'GPE', 'LOC', 'MISC'] 
        for ent in doc.ents:
            if ent.label_ in entity_labels_to_exclude:
                entities_to_ignore.update({ent.text.lower(), ent.lemma_.lower()})
                
    
    for token in doc:
        # Ignorer les jetons qui ne sont pas alphabétiques ou qui sont des stop words
        if not token.is_alpha or token.is_stop or token.text in BASE_EXCLUSION_WORDS:
            continue
            
        # Lemmatisation (forme de base du mot)
        term = token.lemma_.lower() if lemmatize else token.text.lower()
        
        # Filtrage des entités nommées et des mots trop courts
        if term not in entities_to_ignore and len(term) > 2:
            tokens.append(term)
            
    # Rejoint les jetons nettoyés et lemmatisés
    return " ".join(tokens)

# --- Fonctions de Calcul de Scores et d'Anomalies (Non modifiées) ---

def calculate_semantique_score(text):
    """
    Calcule le score sémantique (0-100) pour une description de solution.
    Évalue la longueur, la richesse lexicale et la pertinence.
    """
    if not nlp or pd.isna(text):
        return 0.0
    
    try:
        text_str = str(text)
        if len(text_str.strip()) == 0:
            return 0.0
            
        doc = nlp(text_str)
        total_tokens = len(doc)
        if total_tokens == 0:
            return 0.0
        
        # Longueur et Structure
        length_score = min(20, total_tokens * 0.5)
        sentences = list(doc.sents)
        structure_score = min(15, len(sentences) * 3)
        
        # Richesse lexicale et Technique
        non_stop_words = [token for token in doc if token.is_alpha and not token.is_stop]
        lexical_richness = len(non_stop_words) / total_tokens if total_tokens > 0 else 0
        richness_score = lexical_richness * 30
        
        technical_terms = ['erreur', 'bug', 'problème', 'solution', 'correct', 
                           'résoudre', 'incident', 'technique', 'corriger', 'réparer']
        technical_count = sum(1 for token in doc if token.text.lower() in technical_terms)
        technical_score = min(20, technical_count * 2)
        
        # Clarté (faible proportion de mots vides)
        stop_word_ratio = sum(1 for token in doc if token.is_stop) / total_tokens
        clarity_score = (1 - stop_word_ratio) * 15
        
        total_score = length_score + structure_score + richness_score + technical_score + clarity_score
        
        # Pénalités (mots inconnus, phrases trop longues)
        penalties = 0
        unknown_words = sum(1 for token in doc if token.is_oov and token.is_alpha)
        penalties += min(10, unknown_words * 2)
        
        if sentences and total_tokens > 0:
            avg_sentence_length = total_tokens / len(sentences)
            if avg_sentence_length > 25:
                penalties += 10
        
        final_score = max(0, min(100, total_score - penalties))
        return round(final_score, 2)
        
    except Exception as e:
        # Retourne une valeur neutre en cas d'erreur de traitement
        return 50.0

def calculate_concordance_score(problem, solution):
    """Calcule le score de concordance entre la description du problème et de la solution."""
    if pd.isna(problem) or pd.isna(solution):
        return 0.0
    
    try:
        problem_str = str(problem).lower().strip()
        solution_str = str(solution).lower().strip()
        
        if not problem_str or not solution_str:
            return 0.0
        
        # 1. Similarité textuelle basique
        matcher = SequenceMatcher(None, problem_str, solution_str)
        base_similarity = matcher.ratio() * 40
        
        # 2. Présence de mots-clés de résolution
        resolution_keywords = ['résolu', 'corrigé', 'réparé', 'fixé', 'solution', 'résolution', 'complété']
        resolution_found = any(keyword in solution_str for keyword in resolution_keywords)
        resolution_score = 20 if resolution_found else 0
        
        # 3. Longueur relative de la solution (pénalise les solutions trop courtes)
        problem_words = len(problem_str.split())
        solution_words = len(solution_str.split())
        
        if problem_words > 0 and solution_words > 0:
            length_ratio = min(1.0, solution_words / problem_words)
            length_score = length_ratio * 20
        else:
            length_score = 0
        
        # 4. Indices de structure (étapes, achèvement)
        structure_score = 10 if any(marker in solution_str for marker in ['premièrement', 'étape']) else 5
        completion_score = 10 if any(marker in solution_str for marker in ['terminé', 'fini']) else 0
        
        total_score = base_similarity + resolution_score + length_score + structure_score + completion_score
        return min(100, round(total_score, 2))
        
    except Exception as e:
        print(f"Erreur calcul concordance: {e}")
        return 50.0

def calculate_temporal_score(df):
    """
    Calcule le Z-score temporel (mesure l'écart par rapport à la moyenne)
    et identifie les anomalies temporelles.
    """
    if df.empty or 'TempsHeures' not in df.columns or df['TempsHeures'].isnull().all():
        df['TempsMoyenHeures'] = 0.0
        df['EcartTypeHeures'] = 1.0
        df['ScoreTemporel'] = 0.0
        df['AnomalieTemporelle'] = 'Non'
        return df
        
    mean_h = df['TempsHeures'].mean()
    std_h = df['TempsHeures'].std() 
    
    std_safe = std_h if std_h > 0 else 1.0e-9 
    
    df['TempsMoyenHeures'] = round(mean_h, 2)
    df['EcartTypeHeures'] = round(std_h if std_h > 0 else 1.0, 2)
    
    # Calcul du Z-Score
    df['ScoreTemporel'] = (df['TempsHeures'] - mean_h) / std_safe
    
    # Détection d'anomalie
    df['AnomalieTemporelle'] = np.where(np.abs(df['ScoreTemporel']) > Z_SCORE_THRESHOLD, 'Oui', 'Non')
    
    return df

def determine_final_status(row):
    """Détermine le statut final basé sur les 3 indicateurs d'anomalie."""
    sem_ok = row['ScoreSemantique'] >= SEMAN_THRESHOLD * 100
    conc_ok = row['ScoreConcordance'] >= CONC_THRESHOLD * 100
    temp_ok = row['AnomalieTemporelle'] == 'Non'
    
    if sem_ok and conc_ok and temp_ok:
        return 'OK'
    
    num_anomalies = sum([not sem_ok, not conc_ok, not temp_ok])
    
    if num_anomalies == 1:
        if not temp_ok:
            return 'Anomalie de Temps'
        elif not conc_ok:
            return 'Anomalie de Concordance'
        elif not sem_ok:
            return 'Anomalie Sémantique'
    
    if num_anomalies >= 2:
        return 'Multiples Anomalies'
    
    return 'Anomalie Indéterminée'

def calculate_ticket_note(row):
    """Calcule la Note de Ticket (Base 10) par pénalité selon le Statut."""
    status = row['Statut']
    
    if status == 'OK':
        return 10.0
    elif status == 'Anomalie de Temps':
        return 7.0
    elif status == 'Anomalie Sémantique' or status == 'Anomalie de Concordance':
        return 8.0
    elif status == 'Multiples Anomalies':
        return 5.0
    else:
        return 6.0

def generate_anomaly_description(row):
    """Génère une description détaillée de l'anomalie."""
    anomalies = []
    
    if row['ScoreSemantique'] < SEMAN_THRESHOLD * 100:
        anomalies.append(f"Description du solution peu claire (Score: {row['ScoreSemantique']:.2f}%)")
    
    if row['ScoreConcordance'] < CONC_THRESHOLD * 100:
        anomalies.append(f"Solution peu pertinente par rapport au problème (Score: {row['ScoreConcordance']:.2f}%)")
    
    if row['AnomalieTemporelle'] == 'Oui':
        anomalies.append(f"Temps de résolution anormal ({row['TempsHeures']:.2f}h / Z-Score: {row['ScoreTemporel']:.2f})")
    
    return "; ".join(anomalies) if anomalies else "Aucune anomalie détectée"

# --- Fonctions de Clustering et d'Extraction de Mots-clés (MODIFIÉES) ---

def extract_cluster_info(descriptions):
    """
    Extrait un nom de groupe et des mots-clés significatifs pour un cluster 
    en utilisant une approche dynamique (TF-IDF) sur le texte pré-traité.
    """
    if not descriptions or not nlp:
        return "Sans description", "Aucun contenu"
    
    # 1. Pré-traitement de chaque description du cluster
    # On utilise la lemmatisation et le filtre d'entités pour un TF-IDF très ciblé
    cleaned_descriptions = [preprocess_text(desc, lemmatize=True, filter_entities=True) for desc in descriptions]
    cleaned_descriptions = [d for d in cleaned_descriptions if d.strip()] # Retirer les descriptions vides après nettoyage
    
    if not cleaned_descriptions:
        return "Sans contenu après nettoyage", "Aucun mot-clé pertinent"

    # 2. Vectorisation TF-IDF (Pondération et sélection des mots-clés techniques)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), 
        max_df=0.85, 
        min_df=0.01 
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned_descriptions)
    except ValueError:
        return "Contenu homogène ou générique", "TF-IDF non applicable"

    feature_names = vectorizer.get_feature_names_out()
    
    # 3. Calcul des scores moyens TF-IDF
    avg_tfidf_scores = tfidf_matrix.mean(axis=0).tolist()[0]
    tfidf_scores = pd.Series(avg_tfidf_scores, index=feature_names)
    
    # 4. Extraction des mots-clés les plus importants (Top 5)
    top_keywords_df = tfidf_scores.sort_values(ascending=False).head(5)
    final_keywords = top_keywords_df.index.tolist()
    
    # --- Attribution de catégorie plus adaptative ---

    technical_categories = {
        'Problème Matériel': ['imprimante', 'écran', 'souris', 'clavier', 'casque', 'téléphone', 'fixe', 'matériel', 'scanner', 'scan', 'disque', 'toner', 'cartouche', 'mobile'],
        'Email et Communication': ['mail', 'email', 'messagerie', 'outlook', 'courriel', 'exchange', 'teams', 'zoom'],
        'Logiciel et Application': ['windows', 'office', 'sap', 'programme', 'fiori', 'excel', 'word', 'adobe', 'navision', 'onedrive', 'sharepoint', 'acrobat'],
        'Sécurité et Accès': ['motpasse', 'authentification', 'mdp', 'droits', 'vpn', 'sécurité', 'mfa', 'verrouiller', 'acces', 'connecter'], 
        'Réseau et Infrastructure': ['wifi', 'reseau', 'internet', 'switch', 'routeur', 'câble', 'fibre', 'lan', 'wan', 'ip', 'cloud', 'azure', 'aws', 'serveur'],
        'Installation et Déploiement': ['déploiement', 'configuration', 'migration', 'transfert', 'reinitialisation', 'setup', 'image', 'gabarit', 'préparation']
    }
    
    group_name = "Problème Inconnu ou Spécifique" 
    category_scores = {}
    
    # 5. Scoring de catégorie basé sur les mots-clés TF-IDF trouvés
    for category, keywords in technical_categories.items():
        score = sum(1 for kw in final_keywords if any(term in kw for term in keywords))
        if score > 0:
            category_scores[category] = score
    
    if category_scores:
        group_name = max(category_scores.items(), key=lambda x: x[1])[0]
        
    # Si le score de catégorie est faible, on utilise les mots-clés TF-IDF pour nommer le groupe
    if not category_scores or max(category_scores.values()) < 1:
        if final_keywords:
            group_name = f"Thème: {final_keywords[0].capitalize()}"
            if len(final_keywords) > 1:
                group_name += f" et {final_keywords[1].capitalize()}"
        
    
    keywords_match = ", ".join(final_keywords) if final_keywords else "Analyse sémantique IA (TF-IDF)"
        
    return group_name, keywords_match

# --- Fonction Principale du Pipeline (MODIFIÉE pour le pré-traitement sémantique) ---

def run_full_analysis(df):
    """Exécute l'intégralité du pipeline d'analyse IA."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Début de l'analyse sur {len(df)} tickets assignés")
    
    # 1. Initialisation des colonnes de Clustering
    df['ClusterID'] = 0 
    
    # 2. Calculs de Scores (Basés sur le texte BRUT)
    df['ScoreSemantique'] = df['SolutionContent'].apply(calculate_semantique_score)
    df['ScoreConcordance'] = df.apply(
        lambda row: calculate_concordance_score(row['ProblemDescription'], row['SolutionContent']),
        axis=1
    )
    df = calculate_temporal_score(df.copy()) 
    
    # 3. Détermination du Statut, Note et Description
    df['Statut'] = df.apply(determine_final_status, axis=1)
    df['TicketNote'] = df.apply(calculate_ticket_note, axis=1)
    
    # Calcul de la Moyenne Employé
    if 'AssigneeEmployeeKey' in df.columns:
        employee_avg = df.groupby('AssigneeEmployeeKey')['TicketNote'].mean().round(2)
        df['EmployeeAvgScore'] = df['AssigneeEmployeeKey'].map(employee_avg)
        df['EmployeeAvgScore'] = df['EmployeeAvgScore'].fillna(df['TicketNote'])
    else:
        df['EmployeeAvgScore'] = df['TicketNote']
    
    df['AnomalyDescription'] = df.apply(generate_anomaly_description, axis=1)
    
    # 4. Clustering Sémantique (Regroupement final)
    cluster_results = pd.DataFrame() 
    
    if st_model is not None and 'ProblemDescription' in df.columns:
        try:
            # 4.a. Pré-traitement de TOUTES les descriptions pour l'encodage Sémantique
            df['CleanedDescription'] = df['ProblemDescription'].apply(
                lambda x: preprocess_text(x, lemmatize=True, filter_entities=True)
            )

            # 4.b. Filtration des données valides pour le clustering
            valid_data = df[df['CleanedDescription'].notna() & (df['CleanedDescription'].str.strip() != '')]
            valid_indices = valid_data.index.tolist()
            valid_descriptions = valid_data['CleanedDescription'].tolist()
            
            # Utilisation des descriptions nettoyées pour la Vectorisation Sémantique (Sentence Transformer)
            if len(valid_descriptions) > 1:
                print(f"🔧 Début du clustering sémantique sur {len(valid_descriptions)} descriptions nettoyées...")
                # Note: Cette étape est la plus coûteuse en temps
                embeddings = st_model.encode(valid_descriptions, show_progress_bar=False)
                
                base_tickets = len(valid_descriptions)
                # Calcule le nombre de clusters de manière dynamique (max 60 clusters)
                n_clusters = min(60, max(2, base_tickets // 50 + 1)) 
                
                # Algorithme de Clustering (Agglomerative)
                clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
                clustering_model.fit(embeddings)
                
                cluster_mapping = pd.Series(clustering_model.labels_, index=valid_indices)
                df.loc[valid_indices, 'ClusterID'] = cluster_mapping.values
                
                # 4.c. Extraction des informations de cluster
                cluster_data = []
                for cluster_id in range(n_clusters):
                    cluster_descriptions_original = df[df['ClusterID'] == cluster_id]['ProblemDescription'].tolist()
                    if cluster_descriptions_original:
                        # Utilise la fonction TF-IDF dynamique
                        group_name, keywords = extract_cluster_info(cluster_descriptions_original)
                        cluster_data.append({
                            'ProblemNameGroup': group_name,
                            'ClusterID': cluster_id,
                            'KeywordMatch': keywords,
                            'RecurrenceCount': len(cluster_descriptions_original)
                        })
                
                cluster_results = pd.DataFrame(cluster_data)
                print(f"✅ Clustering terminé: {len(cluster_results)} clusters générés")
                
        except Exception as e:
            print(f"❌ Erreur de Clustering ou de Vectorisation: {e}")
    
    # 5. Nettoyage de la colonne temporaire 'CleanedDescription'
    if 'CleanedDescription' in df.columns:
        df = df.drop(columns=['CleanedDescription'])
        
    df['ClusterID'] = df['ClusterID'].fillna(0).astype(int)
    df_anomalies = df.copy()
    
    # 6. Finalisation des résultats de Clustering
    if not cluster_results.empty:
        tickets_restants = len(df_anomalies[df_anomalies['ClusterID'] == 0])
        if tickets_restants > 0:
            if 0 not in cluster_results['ClusterID'].values:
                 cluster_results.loc[len(cluster_results)] = {
                    'ProblemNameGroup': 'Ticket Non Analysé (Description Manquante)',
                    'ClusterID': 0, 
                    'KeywordMatch': 'Non applicable',
                    'RecurrenceCount': tickets_restants
                }

        cluster_results = cluster_results.sort_values(by='ClusterID').reset_index(drop=True)
        
    else:
        df_anomalies['ClusterID'] = 0 
        cluster_results = pd.DataFrame([{
            'ProblemNameGroup': 'Échec de Classification IA',
            'ClusterID': 0,
            'KeywordMatch': 'Vérifiez les dépendances spacy et sentence-transformers',
            'RecurrenceCount': len(df)
        }])

    # 7. Sélection des colonnes pour la table FactAnomaliesDetail
    df_anomalies = df_anomalies[[
        'TicketID', 'FactKey', 'AssigneeEmployeeKey', 'AssigneeFullName', 
        'TicketNote', 'EmployeeAvgScore', 'ScoreSemantique', 'ScoreConcordance',
        'TempsHeures', 'TempsMoyenHeures', 'EcartTypeHeures', 'ScoreTemporel',
        'AnomalieTemporelle', 'Statut', 'AnomalyDescription', 
        'ClusterID' 
    ]].copy()
    
    return df_anomalies, cluster_results

