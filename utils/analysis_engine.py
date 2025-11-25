import pandas as pd
import numpy as np
import spacy
from difflib import SequenceMatcher
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import re
from collections import Counter
import string
 
# Initialisation des ressources IA/NLP
try:
    # fr_core_news_sm est nécessaire pour la reconnaissance d'entités nommées (REN)
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
 
# --- Fonctions d'analyse de Score (Inchangées) ---

def calculate_semantique_score(text):
    """Calcule le score sémantique pour les descriptions de solutions."""
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
        
        # Longueur appropriée
        length_score = min(20, total_tokens * 0.5)
        
        # Structure (phrases complètes)
        sentences = list(doc.sents)
        num_sentences = len(sentences)
        structure_score = min(15, num_sentences * 3)
        
        # Richesse lexicale
        non_stop_words = [token for token in doc if token.is_alpha and not token.is_stop]
        lexical_richness = len(non_stop_words) / total_tokens if total_tokens > 0 else 0
        richness_score = lexical_richness * 30
        
        # Cohérence technique
        technical_terms = ['erreur', 'bug', 'problème', 'solution', 'correct', 
                           'réparer', 'installer', 'configurer', 'résoudre', 'dépanner',
                           'incident', 'panne', 'dysfonctionnement', 'technique']
        
        technical_count = sum(1 for token in doc if token.text.lower() in technical_terms)
        technical_score = min(20, technical_count * 2)
        
        # Clarté (faible proportion de mots vides)
        stop_word_ratio = sum(1 for token in doc if token.is_stop) / total_tokens
        clarity_score = (1 - stop_word_ratio) * 15
        
        # Score total
        total_score = length_score + structure_score + richness_score + technical_score + clarity_score
        
        # Pénalités
        penalties = 0
        
        unknown_words = sum(1 for token in doc if token.is_oov and token.is_alpha)
        penalties += min(10, unknown_words * 2)
        
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
        
        # Similarité basique
        matcher = SequenceMatcher(None, problem_str, solution_str)
        base_similarity = matcher.ratio() * 40
        
        # Présence de mots-clés de résolution
        resolution_keywords = ['résolu', 'corrigé', 'réparé', 'fixé', 'solution', 
                              'résolution', 'terminé', 'complété', 'réussi']
        
        resolution_found = any(keyword in solution_str for keyword in resolution_keywords)
        resolution_score = 20 if resolution_found else 0
        
        # Longueur relative de la solution
        problem_words = len(problem_str.split())
        solution_words = len(solution_str.split())
        
        if problem_words > 0 and solution_words > 0:
            length_ratio = min(1.0, solution_words / problem_words)
            length_score = length_ratio * 20
        else:
            length_score = 0
        
        # Structure de la solution
        solution_has_steps = any(marker in solution_str for marker in ['premièrement', 'ensuite', 'puis', 'étape', 'step'])
        structure_score = 10 if solution_has_steps else 5
        
        # Présence d'indicateurs de complétion
        completion_indicators = any(marker in solution_str for marker in ['terminé', 'fini', 'complété', 'finalisé'])
        completion_score = 10 if completion_indicators else 0
        
        total_score = base_similarity + resolution_score + length_score + structure_score + completion_score
        return min(100, round(total_score, 2))
        
    except Exception as e:
        print(f"Erreur calcul concordance: {e}")
        return 50.0

def calculate_temporal_score(df):
    """Calcule le Z-score temporel et détecte les anomalies temporelles."""
    if df.empty or 'TempsHeures' not in df.columns:
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
    
    df['ScoreTemporel'] = (df['TempsHeures'] - mean_h) / std_safe
    
    df['AnomalieTemporelle'] = np.where(np.abs(df['ScoreTemporel']) > Z_SCORE_THRESHOLD, 'Oui', 'Non')
    
    return df

def determine_final_status(row):
    """Détermine le statut final basé sur les 3 scores."""
    sem_ok = row['ScoreSemantique'] >= SEMAN_THRESHOLD * 100
    conc_ok = row['ScoreConcordance'] >= CONC_THRESHOLD * 100
    temp_ok = row['AnomalieTemporelle'] == 'Non'
    
    if sem_ok and conc_ok and temp_ok:
        return 'OK'
    elif sem_ok and conc_ok and not temp_ok:
        return 'Anomalie de Temps'
    elif sem_ok and not conc_ok and temp_ok:
        return 'Anomalie de Concordance'
    elif not sem_ok and conc_ok and temp_ok:
        return 'Anomalie Sémantique'
    
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
        anomalies.append("Description du solution peu claire")
    
    if row['ScoreConcordance'] < CONC_THRESHOLD * 100:
        anomalies.append("Solution peu pertinente par rapport au problème")
    
    if row['AnomalieTemporelle'] == 'Oui':
        anomalies.append(f"Temps de résolution anormal ({row['TempsHeures']:.2f}h)")
    
    if anomalies:
        return "; ".join(anomalies)
    else:
        return "Aucune anomalie détectée"

# --- Logique de Clustering (extraction de mots-clés optimisée) ---

def extract_cluster_info(descriptions):
    """
    Extrait un nom de groupe significatif et des mots-clés pour un cluster, 
    en utilisant la REN pour filtrer les noms propres et le nommage automatique.
    """
    if not descriptions or not nlp:
        return "Sans description", "Aucun contenu"
    
    all_text = " ".join(descriptions)
    
    # 1. Traitement spaCy pour la Reconnaissance d'Entités Nommées (REN)
    doc = nlp(all_text)
    entities_to_ignore = set()
    for ent in doc.ents:
        # Ignore les Personnes (PER), Organisations (ORG), Lieux (LOC/GPE)
        # Ajout des produits (PROD) et Événements (EVENT) qui peuvent être des noms propres
        if ent.label_ in ['PER', 'ORG', 'LOC', 'GPE', 'PROD', 'EVENT']: 
            # Ajout du texte de l'entité et de ses mots individuels
            entities_to_ignore.add(ent.text.lower())
            entities_to_ignore.update(ent.text.lower().split())
            
    # 2. Nettoyage et tokenisation standard
    # Conserver les tirets/apostrophes si nécessaires pour les mots (ex: mot-clé, c'est)
    translator = str.maketrans(string.punctuation.replace('-', '').replace("'", ''), ' ' * len(string.punctuation.replace('-', '').replace("'", '')) + ' ' * len(string.digits), string.digits)
    clean_text = all_text.lower().translate(translator)
    words = clean_text.split()
    
    # 3. STOP WORDS ÉTENDUS (Renforcé)
    extended_stop_words = set(list(spacy.lang.fr.stop_words.STOP_WORDS) + [
        # Mots vides courants
        'le', 'la', 'les', 'un', 'une', 'des', 'avec', 'pour', 'qui', 'est', 'être', 'avoir', 'faire',
        'dans', 'sur', 'sous', 'vers', 'avant', 'après', 'chez', 'entre', 'sans', 'comme', 'comment',
        
        # Termes spécifiques à GLPI / la procédure
        'glpi', 'ticket', 'demande', 'facture', 'code', 'version', 'numéro', 'id', 'lien', 'demander',
        
        # Termes de politesse / temps
        'bonjour', 'merci', 'cordialement', 'salut', 'hello', 'suite', 'depuis', 'date', 'jour', 'mois', 'année',
        
        # Termes génériques (maintenant inclus 'problème')
        'quelque', 'plusieurs', 'chaque', 'urgent', 'important', 'nécessaire', 
        'besoin', 'madame', 'monsieur', 'utilisateur', 'personne', 'collaborateur', 
        'problème', 'erreur', 'incident', 'souci', 'bug', 'panne', 'fixé', 'résolu', 
        'corrigé', 'réparé', 'falloir', 'pouvoir', 'vouloir', 'devoir', 'savoir', 
        'impossible', 'bloqué', 'technique', 'techniquement', 'général', 'cas', 'via',
        
        # Termes d'action/statut courant
        'mise', 'jour', 'création', 'ajout', 'modification', 'supprimer', 'archive', 'vide', 'archivage',
        'compte', 'accès', 'installer', 'installation', 'connexion', 'gestion', 'ouvrir', 'fonctionne',
        
        # Noms propres spécifiques trouvés dans les captures à ajouter pour le filtrage
        'flora', 'rimco', 'pedro', 'hotix', 'gabon', 'cameroun', 'bernabe', 'senegal', 'base', 'whatsapp',
        'timesheet', 'astral', 'hotel', 'lepic', 'lhotel', 'riviera', 'bouquet', 'mariage', 'bbci', 'licence', 
    ])
    
    # 4. Filtrage des mots significatifs (Retrait des stop words ET des entités nommées)
    meaningful_words = [
        word for word in words 
        if word not in extended_stop_words 
        and word not in entities_to_ignore 
        and len(word) >= 3
        # Exclusion des termes purement numériques (déjà fait par translate, mais vérification de sûreté)
        and not word.isdigit()
        # Exclusion des singletons d'une seule lettre ou de codes trop courts
        and not (len(word) == 1 and word.isalpha())
    ]
    
    # 5. Catégories techniques prédéfinies (Inchangé)
    technical_categories = {
        'Réseau et Connexion': ['wifi', 'connexion', 'réseau', 'internet', 'vpn', 'serveur', 'switch', 'routeur', 'box', 'partage', 'partagé', 'fichiers', 'dossier'],
        'Email et Communication': ['mail', 'email', 'messagerie', 'outlook', 'courriel', 'exchange', 'teams', 'client', 'envoi'],
        'Matériel': ['imprimante', 'ordinateur', 'écran', 'clavier', 'souris', 'scanner', 'portable', 'pc', 'matériel', 'batterie', 'scan', 'impression'],
        'Logiciel et Application': ['logiciel', 'application', 'windows', 'office', 'sap', 'programme', 'fiori', 'excel', 'word', 'powerpoint', 'adobe', 'suite', 'applicatif'],
        'Sécurité et Accès': ['motdepasse', 'authentification', 'login', 'compte', 'accès', 'mdp', 'mot', 'passe', 'droits', 'autorisation', 'validation', 'session', 'antivirus'],
        'Installation et Déploiement': ['installation', 'déploiement', 'configuration', 'migration', 'transfert', 'miseajour']
    }
    
    group_name = "Domaine Non Classé"
    common_words = []
    
    if meaningful_words:
        word_freq = Counter(meaningful_words)
        
        # Tentative d'attribution d'une catégorie prédéfinie (Priorité)
        category_scores = {}
        for category, keywords in technical_categories.items():
            score = sum(word_freq.get(keyword, 0) for keyword in keywords)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            # Succès: Utiliser la meilleure catégorie
            group_name = max(category_scores.items(), key=lambda x: x[1])[0]
            
        # Extraction des mots-clés les plus fréquents pour KeywordMatch
        # Utiliser un seuil dynamique basé sur le nombre de tickets dans le cluster
        min_occurrences = max(2, len(descriptions) // 10) # Augmenté le seuil pour des mots plus pertinents
        common_words = [
            word for word, count in word_freq.most_common(10) 
            if count >= min_occurrences
        ]

        # Nommage AUTOMATIQUE (si aucune catégorie prédéfinie n'a été trouvée)
        if group_name == "Domaine Non Classé":
            if common_words:
                # Utilise les 2 mots-clés principaux pour nommer le groupe
                name_words = common_words[:2]
                group_name = "Problème sur " + " / ".join(name_words).capitalize()
            else:
                group_name = "Nouveau Problème Non Classé"
        
        # Finalisation des mots-clés
        keywords_match = ", ".join(common_words[:5]) if common_words else "analyse sémantique"
        
        # FILTRAGE FINAL des mots-clés pour éviter la redondance
        # Retirer tout mot-clé qui est déjà très générique ou fait partie du nom du groupe
        final_keywords = []
        group_words = set(group_name.lower().split())
        
        for word in common_words[:5]:
            # Vérifie si le mot n'est pas déjà dans le nom du groupe et n'est pas trop générique
            if word not in group_words and word not in extended_stop_words:
                final_keywords.append(word)
                
        keywords_match = ", ".join(final_keywords) if final_keywords else "analyse sémantique"
        
        return group_name, keywords_match
        
    return group_name, "analyse sémantique insuffisante"


# --- Fonction Principale (Finalisée) ---

def run_full_analysis(df):
    """Exécute l'intégralité du pipeline d'analyse IA."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Début de l'analyse sur {len(df)} tickets assignés")
    
    # 1. Calculs de Scores
    df['ScoreSemantique'] = df['SolutionContent'].apply(calculate_semantique_score)
    df['ScoreConcordance'] = df.apply(
        lambda row: calculate_concordance_score(row['ProblemDescription'], row['SolutionContent']),
        axis=1
    )
    df = calculate_temporal_score(df.copy()) 
    
    # 2. Détermination du Statut et de la Note
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
    
    # 3. Clustering pour problèmes récurrents
    cluster_results = pd.DataFrame()
    df['ClusterID'] = -1 
    
    if st_model is not None and 'ProblemDescription' in df.columns:
        try:
            valid_data = df[df['ProblemDescription'].notna() & (df['ProblemDescription'].str.strip() != '')]
            valid_indices = valid_data.index.tolist()
            valid_descriptions = valid_data['ProblemDescription'].tolist()

            if len(valid_descriptions) > 1:
                print(f"🔧 Début du clustering sur {len(valid_descriptions)} tickets...")
                embeddings = st_model.encode(valid_descriptions, show_progress_bar=False)
                
                base_tickets = len(valid_descriptions)
                n_clusters = min(60, max(3, base_tickets // 50 + 1)) 
                
                # Utilisation d'AgglomerativeClustering (Hierarchical) qui est souvent plus stable
                clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
                clustering_model.fit(embeddings)
                
                cluster_mapping = pd.Series(clustering_model.labels_, index=valid_indices)
                df.loc[valid_indices, 'ClusterID'] = cluster_mapping.values
                
                # Préparation de DimRecurrentProblems
                cluster_data = []
                for cluster_id in range(n_clusters):
                    cluster_descriptions = df[df['ClusterID'] == cluster_id]['ProblemDescription'].tolist()
                    if cluster_descriptions:
                        group_name, keywords = extract_cluster_info(cluster_descriptions)
                        cluster_data.append({
                            'ProblemNameGroup': group_name,
                            'ClusterID': cluster_id,
                            'KeywordMatch': keywords,
                            'RecurrenceCount': len(cluster_descriptions)
                        })
                
                cluster_results = pd.DataFrame(cluster_data)
                print(f"✅ Clustering terminé: {len(cluster_results)} clusters générés")
                
        except Exception as e:
            print(f"❌ Erreur clustering: {e}")
            df['ClusterID'] = -1
    
    # 4. Sélection des colonnes pour FactAnomaliesDetail
    df_anomalies = df[[
        'TicketID', 'FactKey', 'AssigneeEmployeeKey', 'AssigneeFullName', 
        'TicketNote', 'EmployeeAvgScore', 'ScoreSemantique', 'ScoreConcordance',
        'TempsHeures', 'TempsMoyenHeures', 'EcartTypeHeures', 'ScoreTemporel',
        'AnomalieTemporelle', 'Statut', 'AnomalyDescription', 
        'ClusterID' 
    ]].copy()
    
    # S'assurer que le ClusterID est un entier pour la sauvegarde SQL
    df_anomalies['ClusterID'] = df_anomalies['ClusterID'].fillna(-1).astype(int)

    # Incrémentation du ClusterID de +1 pour la base de données (PK ne doit pas commencer à 0)
    df_anomalies['ClusterID'] = df_anomalies['ClusterID'] + 1
    if not cluster_results.empty:
        cluster_results['ClusterID'] = cluster_results['ClusterID'] + 1
        
    # Ajout du ClusterID 0 (pour les tickets non clusterisés) dans les résultats des clusters si nécessaire
    if 0 not in cluster_results['ClusterID'].values:
        cluster_results.loc[-1] = {
            'ProblemNameGroup': 'Ticket Non Classé / Anomalie',
            'ClusterID': 0,
            'KeywordMatch': 'Non applicable',
            'RecurrenceCount': 0
        }
        cluster_results.index = cluster_results.index + 1
        cluster_results = cluster_results.sort_values(by='ClusterID').reset_index(drop=True)

    return df_anomalies, cluster_results



//////////////////////////////////////
 import pandas as pd
import numpy as np
import spacy
from difflib import SequenceMatcher
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import re
from collections import Counter
import string
 
# Initialisation des ressources IA/NLP
try:
    # fr_core_news_sm est nécessaire pour la reconnaissance d'entités nommées (REN)
    nlp = spacy.load("fr_core_news_sm") 
    st_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
except Exception as e:
    print(f"Erreur de chargement des modèles NLP/IA : {e}")
    nlp = None
    st_model = None
 
# Définition des constantes d'anomalie
SEMAN_THRESHOLD = 0.60
CONC_THRESHOLD = 0.40
Z_SCORE_THRESHOLD = 2
 
# --- Fonctions d'analyse de Score ---

def calculate_semantique_score(text):
    """Calcule le score sémantique pour les descriptions de solutions."""
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
        
        # ... (Logique de score sémantique inchangée) ...
        length_score = min(20, total_tokens * 0.5)
        sentences = list(doc.sents)
        num_sentences = len(sentences)
        structure_score = min(15, num_sentences * 3)
        
        non_stop_words = [token for token in doc if token.is_alpha and not token.is_stop]
        lexical_richness = len(non_stop_words) / total_tokens if total_tokens > 0 else 0
        richness_score = lexical_richness * 30
        
        technical_terms = ['erreur', 'bug', 'problème', 'solution', 'correct', 
                           'réparer', 'installer', 'configurer', 'résoudre', 'dépanner',
                           'incident', 'panne', 'dysfonctionnement', 'technique']
        
        technical_count = sum(1 for token in doc if token.text.lower() in technical_terms)
        technical_score = min(20, technical_count * 2)
        
        stop_word_ratio = sum(1 for token in doc if token.is_stop) / total_tokens
        clarity_score = (1 - stop_word_ratio) * 15
        
        total_score = length_score + structure_score + richness_score + technical_score + clarity_score
        
        penalties = 0
        unknown_words = sum(1 for token in doc if token.is_oov and token.is_alpha)
        penalties += min(10, unknown_words * 2)
        
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
        
        # ... (Logique de score de concordance inchangée) ...
        matcher = SequenceMatcher(None, problem_str, solution_str)
        base_similarity = matcher.ratio() * 40
        
        resolution_keywords = ['résolu', 'corrigé', 'réparé', 'fixé', 'solution', 
                              'résolution', 'terminé', 'complété', 'réussi']
        
        resolution_found = any(keyword in solution_str for keyword in resolution_keywords)
        resolution_score = 20 if resolution_found else 0
        
        problem_words = len(problem_str.split())
        solution_words = len(solution_str.split())
        
        if problem_words > 0 and solution_words > 0:
            length_ratio = min(1.0, solution_words / problem_words)
            length_score = length_ratio * 20
        else:
            length_score = 0
        
        solution_has_steps = any(marker in solution_str for marker in ['premièrement', 'ensuite', 'puis', 'étape', 'step'])
        structure_score = 10 if solution_has_steps else 5
        
        completion_indicators = any(marker in solution_str for marker in ['terminé', 'fini', 'complété', 'finalisé'])
        completion_score = 10 if completion_indicators else 0
        
        total_score = base_similarity + resolution_score + length_score + structure_score + completion_score
        return min(100, round(total_score, 2))
        
    except Exception as e:
        print(f"Erreur calcul concordance: {e}")
        return 50.0

def calculate_temporal_score(df):
    """Calcule le Z-score temporel et détecte les anomalies temporelles."""
    if df.empty or 'TempsHeures' not in df.columns:
        df['TempsMoyenHeures'] = 0.0
        df['EcartTypeHeures'] = 1.0
        df['ScoreTemporel'] = 0.0
        df['AnomalieTemporelle'] = 'Non'
        return df
        
    mean_h = df['TempsHeures'].mean()
    std_h = df['TemtsHeures'].std()
    
    std_safe = std_h if std_h > 0 else 1.0e-9 
    
    df['TempsMoyenHeures'] = round(mean_h, 2)
    df['EcartTypeHeures'] = round(std_h if std_h > 0 else 1.0, 2)
    
    df['ScoreTemporel'] = (df['TempsHeures'] - mean_h) / std_safe
    
    df['AnomalieTemporelle'] = np.where(np.abs(df['ScoreTemporel']) > Z_SCORE_THRESHOLD, 'Oui', 'Non')
    
    return df

def determine_final_status(row):
    """Détermine le statut final basé sur les 3 scores."""
    sem_ok = row['ScoreSemantique'] >= SEMAN_THRESHOLD * 100
    conc_ok = row['ScoreConcordance'] >= CONC_THRESHOLD * 100
    temp_ok = row['AnomalieTemporelle'] == 'Non'
    
    if sem_ok and conc_ok and temp_ok:
        return 'OK'
    elif sem_ok and conc_ok and not temp_ok:
        return 'Anomalie de Temps'
    elif sem_ok and not conc_ok and temp_ok:
        return 'Anomalie de Concordance'
    elif not sem_ok and conc_ok and temp_ok:
        return 'Anomalie Sémantique'
    
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
        anomalies.append("Description du solution peu claire")
    
    if row['ScoreConcordance'] < CONC_THRESHOLD * 100:
        anomalies.append("Solution peu pertinente par rapport au problème")
    
    if row['AnomalieTemporelle'] == 'Oui':
        anomalies.append(f"Temps de résolution anormal ({row['TempsHeures']:.2f}h)")
    
    if anomalies:
        return "; ".join(anomalies)
    else:
        return "Aucune anomalie détectée"

# --- Logique de Clustering (extraction de mots-clés optimisée) ---

def extract_cluster_info(descriptions):
    """
    Extrait un nom de groupe significatif et des mots-clés pour un cluster, 
    avec correction maketrans.
    """
    if not descriptions or not nlp:
        return "Sans description", "Aucun contenu"
    
    all_text = " ".join(descriptions)
    
    # 1. Traitement spaCy pour la Reconnaissance d'Entités Nommées (REN)
    doc = nlp(all_text)
    entities_to_ignore = set()
    for ent in doc.ents:
        if ent.label_ in ['PER', 'ORG', 'LOC', 'GPE', 'PROD', 'EVENT']: 
            entities_to_ignore.add(ent.text.lower())
            entities_to_ignore.update(ent.text.lower().split())
            
    # 2. Nettoyage et tokenisation standard
    
    # --- CORRECTION DE L'ERREUR MAKETRANS ---
    to_replace_punc = string.punctuation.replace('-', '').replace("'", '')
    replace_with_punc = ' ' * len(to_replace_punc)
    
    to_replace_digits = string.digits
    replace_with_digits = ' ' * len(string.digits)
    
    to_replace = to_replace_punc + to_replace_digits
    replace_with = replace_with_punc + replace_with_digits
    
    translator = str.maketrans(to_replace, replace_with)
    
    clean_text = all_text.lower().translate(translator)
    words = clean_text.split()
    
    # 3. STOP WORDS ÉTENDUS (Renforcé)
    extended_stop_words = set(list(spacy.lang.fr.stop_words.STOP_WORDS) + [
        'le', 'la', 'les', 'un', 'une', 'des', 'avec', 'pour', 'qui', 'est', 'être', 'avoir', 'faire',
        'dans', 'sur', 'sous', 'vers', 'avant', 'après', 'chez', 'entre', 'sans', 'comme', 'comment',
        'glpi', 'ticket', 'demande', 'facture', 'code', 'version', 'numéro', 'id', 'lien', 'demander',
        'bonjour', 'merci', 'cordialement', 'salut', 'hello', 'suite', 'depuis', 'date', 'jour', 'mois', 'année',
        'quelque', 'plusieurs', 'chaque', 'urgent', 'important', 'nécessaire', 
        'besoin', 'madame', 'monsieur', 'utilisateur', 'personne', 'collaborateur', 
        'problème', 'erreur', 'incident', 'souci', 'bug', 'panne', 'fixé', 'résolu', 
        'corrigé', 'réparé', 'falloir', 'pouvoir', 'vouloir', 'devoir', 'savoir', 
        'impossible', 'bloqué', 'technique', 'techniquement', 'général', 'cas', 'via',
        'mise', 'jour', 'création', 'ajout', 'modification', 'supprimer', 'archive', 'vide', 'archivage',
        'compte', 'accès', 'installer', 'installation', 'connexion', 'gestion', 'ouvrir', 'fonctionne',
        'flora', 'rimco', 'pedro', 'hotix', 'gabon', 'cameroun', 'bernabe', 'senegal', 'base', 'whatsapp',
        'timesheet', 'astral', 'hotel', 'lepic', 'lhotel', 'riviera', 'bouquet', 'mariage', 'bbci', 'licence', 
    ])
    
    # 4. Filtrage des mots significatifs
    meaningful_words = [
        word for word in words 
        if word not in extended_stop_words 
        and word not in entities_to_ignore 
        and len(word) >= 3
        and not word.isdigit()
        and not (len(word) == 1 and word.isalpha())
    ]
    
    # 5. Catégories techniques prédéfinies
    technical_categories = {
        'Réseau et Connexion': ['wifi', 'connexion', 'réseau', 'internet', 'vpn', 'serveur', 'switch', 'routeur', 'box', 'partage', 'partagé', 'fichiers', 'dossier'],
        'Email et Communication': ['mail', 'email', 'messagerie', 'outlook', 'courriel', 'exchange', 'teams', 'client', 'envoi'],
        'Poste de Travail / Matériel': ['imprimante', 'ordinateur', 'écran', 'clavier', 'souris', 'scanner', 'portable', 'pc', 'matériel', 'batterie', 'scan', 'impression'],
        'Logiciel et Application': ['logiciel', 'application', 'windows', 'office', 'sap', 'programme', 'fiori', 'excel', 'word', 'powerpoint', 'adobe', 'suite', 'applicatif'],
        'Sécurité et Accès': ['motdepasse', 'authentification', 'login', 'compte', 'accès', 'mdp', 'mot', 'passe', 'droits', 'autorisation', 'validation', 'session', 'antivirus'],
        'Installation et Déploiement': ['installation', 'déploiement', 'configuration', 'migration', 'transfert', 'miseajour']
    }
    
    group_name = "Domaine Non Classé"
    common_words = []
    
    if meaningful_words:
        word_freq = Counter(meaningful_words)
        
        category_scores = {}
        for category, keywords in technical_categories.items():
            score = sum(word_freq.get(keyword, 0) for keyword in keywords)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            group_name = max(category_scores.items(), key=lambda x: x[1])[0]
            
        min_occurrences = max(2, len(descriptions) // 10)
        common_words = [
            word for word, count in word_freq.most_common(10) 
            if count >= min_occurrences
        ]

        if group_name == "Domaine Non Classé":
            if common_words:
                name_words = common_words[:2]
                group_name = "Problème sur " + " / ".join(name_words).capitalize()
            else:
                group_name = "Nouveau Problème Non Classé"
        
        final_keywords = []
        group_words = set(group_name.lower().split())
        final_ignore_words = extended_stop_words.union(group_words)
        
        for word in common_words[:5]:
            if word not in final_ignore_words:
                final_keywords.append(word)
                
        keywords_match = ", ".join(final_keywords) if final_keywords else "analyse sémantique"
        
        return group_name, keywords_match
        
    return group_name, "analyse sémantique insuffisante"


# --- Fonction Principale (Finalisée avec correction KeyError) ---

def run_full_analysis(df):
    """Exécute l'intégralité du pipeline d'analyse IA."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Début de l'analyse sur {len(df)} tickets assignés")
    
    # 1. Calculs de Scores
    df['ScoreSemantique'] = df['SolutionContent'].apply(calculate_semantique_score)
    df['ScoreConcordance'] = df.apply(
        lambda row: calculate_concordance_score(row['ProblemDescription'], row['SolutionContent']),
        axis=1
    )
    df = calculate_temporal_score(df.copy()) 
    
    # 2. Détermination du Statut et de la Note
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
    
    # 3. Clustering pour problèmes récurrents
    cluster_results = pd.DataFrame() # Initialisation à vide
    df['ClusterID'] = 0 # Initialisation GARANTIE de la colonne
    
    if st_model is not None and 'ProblemDescription' in df.columns:
        try:
            valid_data = df[df['ProblemDescription'].notna() & (df['ProblemDescription'].str.strip() != '')]
            valid_indices = valid_data.index.tolist()
            valid_descriptions = valid_data['ProblemDescription'].tolist()

            if len(valid_descriptions) > 1:
                print(f"🔧 Début du clustering sur {len(valid_descriptions)} tickets...")
                embeddings = st_model.encode(valid_descriptions, show_progress_bar=False)
                
                base_tickets = len(valid_descriptions)
                n_clusters = min(60, max(3, base_tickets // 50 + 1)) 
                
                clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
                clustering_model.fit(embeddings)
                
                cluster_mapping = pd.Series(clustering_model.labels_, index=valid_indices)
                df.loc[valid_indices, 'ClusterID'] = cluster_mapping.values
                
                cluster_data = []
                for cluster_id in range(n_clusters):
                    cluster_descriptions = df[df['ClusterID'] == cluster_id]['ProblemDescription'].tolist()
                    if cluster_descriptions:
                        group_name, keywords = extract_cluster_info(cluster_descriptions)
                        cluster_data.append({
                            'ProblemNameGroup': group_name,
                            'ClusterID': cluster_id,
                            'KeywordMatch': keywords,
                            'RecurrenceCount': len(cluster_descriptions)
                        })
                
                cluster_results = pd.DataFrame(cluster_data)
                print(f"✅ Clustering terminé: {len(cluster_results)} clusters générés")
                
        except Exception as e:
            print(f"❌ Erreur clustering (maketrans ou autre): {e}")
            # Si échec, ClusterID reste à 0 (valeur initialisée)
            df.loc[df['ClusterID'] != 0, 'ClusterID'] = 0 

    # S'assurer que le ClusterID est un entier (résout le problème potentiel de type)
    df['ClusterID'] = df['ClusterID'].fillna(0).astype(int)

    # Incrémentation du ClusterID de +1 pour la base de données (PK ne doit pas commencer à 0)
    df_anomalies = df.copy()
    
    # Gestion de la sortie ClusterID (résout le KeyError dans la suite du pipeline)
    if not cluster_results.empty:
        # Incrémentation de l'ID pour les deux DataFrames (pour que le cluster 0 commence à 1)
        df_anomalies['ClusterID'] = df_anomalies['ClusterID'] + 1
        cluster_results['ClusterID'] = cluster_results['ClusterID'] + 1
        
        # Ajout du ClusterID 0 (pour les tickets non classés/anomalie)
        # On utilise l'ID 0 pour la catégorie spéciale après incrémentation.
        cluster_results.loc[len(cluster_results)] = {
            'ProblemNameGroup': 'Ticket Non Classé / Anomalie',
            'ClusterID': 0, # ID 0 utilisé pour les exceptions
            'KeywordMatch': 'Non applicable',
            'RecurrenceCount': len(df_anomalies[df_anomalies['ClusterID'] == 0]) 
        }
        cluster_results = cluster_results.sort_values(by='ClusterID').reset_index(drop=True)
    else:
        # Si le clustering a échoué ou n'a pas pu s'exécuter:
        df_anomalies['ClusterID'] = 0 # Tout est assigné à l'ID 0
        cluster_results = pd.DataFrame([{
            'ProblemNameGroup': 'Erreur de Classification',
            'ClusterID': 0,
            'KeywordMatch': 'Échec du moteur IA',
            'RecurrenceCount': len(df)
        }])

    # 4. Sélection des colonnes pour FactAnomaliesDetail
    df_anomalies = df_anomalies[[
        'TicketID', 'FactKey', 'AssigneeEmployeeKey', 'AssigneeFullName', 
        'TicketNote', 'EmployeeAvgScore', 'ScoreSemantique', 'ScoreConcordance',
        'TempsHeures', 'TempsMoyenHeures', 'EcartTypeHeures', 'ScoreTemporel',
        'AnomalieTemporelle', 'Statut', 'AnomalyDescription', 
        'ClusterID' 
    ]].copy()
    
    return df_anomalies, cluster_results