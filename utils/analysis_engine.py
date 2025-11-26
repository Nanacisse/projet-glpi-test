import pandas as pd
import numpy as np
import spacy
from difflib import SequenceMatcher
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import re
from collections import Counter
import string
 
# --- Configuration et Initialisation ---
 
# Définition des constantes d'anomalie
SEMAN_THRESHOLD = 0.60  # 60%
CONC_THRESHOLD = 0.40   # 40%
Z_SCORE_THRESHOLD = 2   # |Z| > 2 (Écart-type)
 
# Initialisation des ressources IA/NLP
nlp = None
st_model = None
try:
    # Charge le modèle français pour la Reconnaissance d'Entités Nommées (REN)
    nlp = spacy.load("fr_core_news_sm") 
    # Charge le modèle Sentence Transformer pour l'encodage sémantique
    st_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
except Exception as e:
    print(f"Erreur de chargement des modèles NLP/IA (Vérifiez les dépendances spacy et sentence-transformers) : {e}")
    # Les variables restent None, et le clustering sera ignoré si st_model est None

 
# --- Fonctions de Calcul de Scores et d'Anomalies ---

def calculate_semantique_score(text):
    """
    Calcule le score sémantique (0-100) pour une description de solution.
    Évalue la longueur, la richesse lexicale et la pertinence technique.
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
        # print(f"Erreur analyse sémantique pour le texte '{text_str[:50]}...': {e}")
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
    # CORRECTION ICI: Utiliser 'TempsHeures' et non 'TemtsHeures'
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
    
    return 'Anomalie Indéterminée' # Cas par défaut, devrait rarement arriver

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
        return 6.0 # Anomalie Indéterminée

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


# --- Fonctions de Clustering et d'Extraction de Mots-clés ---

def extract_cluster_info(descriptions):
    """
    Extrait un nom de groupe significatif et des mots-clés pour un cluster.
    Mise à jour pour les 6 catégories demandées + 'Problème Inconnu'.
    """
    if not descriptions or not nlp:
        return "Sans description", "Aucun contenu"
    
    all_text = " ".join(descriptions)
    
    # 1. Reconnaissance d'Entités Nommées (REN) pour ignorer les noms propres
    doc = nlp(all_text)
    entities_to_ignore = set()
    for ent in doc.ents:
        if ent.label_ in ['PER', 'ORG', 'LOC', 'GPE', 'PROD', 'EVENT']: 
            entities_to_ignore.add(ent.text.lower())
            entities_to_ignore.update(ent.text.lower().split())
            
    # 2. Nettoyage et tokenisation standard avec correction maketrans
    to_replace_punc = string.punctuation.replace('-', '').replace("'", '')
    replace_with_punc = ' ' * len(to_replace_punc)
    to_replace_digits = string.digits
    replace_with_digits = ' ' * len(string.digits)
    
    to_replace = to_replace_punc + to_replace_digits
    replace_with = replace_with_punc + replace_with_digits
    
    # Sécurité maketrans
    if len(to_replace) != len(replace_with):
        clean_text = re.sub(f'[{re.escape(string.punctuation.replace("-", "").replace("'", "") + string.digits)}]', ' ', all_text.lower())
    else:
        translator = str.maketrans(to_replace, replace_with)
        clean_text = all_text.lower().translate(translator)
        
    words = clean_text.split()
    
    # 3. STOP WORDS ÉTENDUS (Mots vides + Mots génériques à exclure des KW Match)
    extended_stop_words = set(list(spacy.lang.fr.stop_words.STOP_WORDS) + [
        'le', 'la', 'les', 'un', 'une', 'des', 'avec', 'pour', 'qui', 'est', 'être', 'avoir', 'faire',
        'dans', 'sur', 'sous', 'vers', 'avant', 'après', 'chez', 'entre', 'sans', 'comme', 'comment',
        # Termes GLPI/Génériques
        'glpi', 'ticket', 'demande', 'facture', 'code', 'version', 'numéro', 'id', 'lien', 'demander',
        'bonjour', 'merci', 'suite', 'depuis', 'date', 'jour', 'mois', 'année', 'heure', 'minute',
        'quelque', 'plusieurs', 'chaque', 'urgent', 'important', 'nécessaire', 
        'besoin', 'utilisateur', 'personne', 'collaborateur', 
        'problème', 'erreur', 'incident', 'souci', 'bug', 'panne', 'fixé', 'résolu', 'corrigé', 'réparé', 
        'falloir', 'pouvoir', 'vouloir', 'devoir', 'savoir', 'impossible', 'bloqué', 'technique', 'général', 'cas', 'via',
        'mise', 'jour', 'création', 'ajout', 'modification', 'supprimer', 'archive', 'vide', 'archivage',
        'compte', 'installation', 'connexion', 'gestion', 'ouvrir', 'fonctionne',
        'base', 'timesheet', 'astral', 'hotel', 'lepic', 'lhotel', 'riviera', 'bouquet', 
        'mariage', 'bbci', 'licence', 'client', 'service', 'site', 'serveur', 
        'salle', 'magasin', 'magasins', 'siège', 'sièges', 'hadi', 'san', 'pedro', 'rimco', 'hotix', 'digitalix',
        'gabon', 'cameroun', 'bernabe', 'senegal', 'dga', 'envoi', 'transfert', 'barres', 'carte', 'cadeau',
        'stock', 'dos', 'stickers', 
        'vente', 'flash', 'pegas', 'wingle', 'diesel', 'support', 'dpp','gwmdspd','décembre','dashbord',
        'navision', 'correction', 'hassan', 'semaine', 'ticket', 'hotel','ouattara', 'rosine','pneu',
        'power', 'bbc''css', 'CSS' 'css', 'dpos', 'l', 'j''d', 's' 'c', 'm','ç' 'c', 'n',
        # Les mots-clés techniques sont maintenus pour le SCORING
    ])
    
    # 4. Filtrage des mots significatifs
    meaningful_words = [
        word for word in words 
        if word not in extended_stop_words 
        and word not in entities_to_ignore 
        and len(word) >= 3
        and not word.isdigit()
    ]
    
    # 5. Catégories techniques (Mots-clés enrichis)
    technical_categories = {
        'Problème Matériel': ['imprimante', 'écran', 'souris', 'clavier', 'casque', 'téléphone', 'batterie', 'pc', 'portable', 'fixe', 'matériel', 'scanner', 'scan', 'disque', 'dur', 'ram', 'carte', 'graphique', 'cpu', 'ecran', 'projecteur', 'peripherique', 'changement', 'reparation', 'impression', 'ssd', 'mémoire', 'toner', 'cartouche', 'mfp', 'tablette', 'mobile'],
        'Email et Communication': ['mail', 'email', 'messagerie', 'outlook', 'courriel', 'exchange', 'teams', 'whatsapp', 'téléconférence', 'visio', 'conférence', 'boite', 'archive', 'signature', 'calendrier', 'envoi', 'réception', 'spam', 'quota', 'bloqué', 'délai', 'zoom', 'skype', 'meeting', 'contact', 'groupe'],
        'Logiciel et Application': ['logiciel', 'application', 'windows', 'office', 'sap', 'programme', 'fiori', 'excel', 'word', 'powerpoint', 'adobe', 'suite', 'applicatif', 'licence', 'version', 'miseajour', 'ciel', 'gestion', 'comptabilité', 'hrm', 'outil', 'base', 'donnée', 'bug', 'plantage', 'navision', 'onedrive', 'sharepoint', 'développement', 'script', 'macro', 'vba', 'acrobat'],
        'Sécurité et Accès': ['motdepasse', 'authentification', 'login', 'compte', 'accès', 'mdp', 'droits', 'autorisation', 'validation', 'session', 'antivirus', 'firewall', 'vpn', 'sécurité', 'crypto', 'certificat', 'chiffrement', 'mfa', 'double', 'factor', 'verrouillé', 'piratage', 'acces', 'privilège', 'restriction', 'phishing', 'biométrie', 'utilisateur'],
        'Réseau et Communication': ['wifi', 'connexion', 'réseau', 'internet', 'switch', 'routeur', 'câble', 'fibre', 'box', 'lan', 'wan', 'ip', 'partage', 'dossier', 'fichier', 'ftp', 'impossibilité', 'lent', 'debit', 'coupure', 'infrastructure', 'partagé', 'téléchargement', 'dhcp', 'dns', 'proxy', 'cloud', 'azure', 'aws', 'sauvegarde', 'backup', 'vitesse', 'ralentissement'],
        'Installation et Déploiement': ['installation', 'déploiement', 'configuration', 'migration', 'transfert', 'miseajour', 'parametrage', 'création', 'nouveau', 'poste', 'initiation', 'remplacement', 'transfert', 'déménagement', 'changement', 'formatage', 'reinitialisation', 'setup', 'image', 'gabarit', 'préparation', 'installer']
    }
    
    group_name = "Problème Inconnu" # La nouvelle catégorie par défaut
    common_words = []
    
    if meaningful_words:
        word_freq = Counter(meaningful_words)
        
        # 5a. Attribution d'une catégorie prédéfinie (Priorité)
        category_scores = {}
        for category, keywords in technical_categories.items():
            # Compte les occurrences des mots-clés techniques
            score = sum(word_freq.get(keyword, 0) for keyword in keywords)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            # On prend la catégorie avec le plus grand score
            group_name = max(category_scores.items(), key=lambda x: x[1])[0]
            
        # 5b. Extraction des mots-clés les plus fréquents (pour toutes les catégories)
        min_occurrences = max(2, len(descriptions) // 10) # Seuil dynamique
        common_words = [
            word for word, count in word_freq.most_common(10) 
            if count >= min_occurrences
        ]

        # 5c. Filtrage final des mots-clés
        final_keywords = []
        group_words = set(group_name.lower().split())
        final_ignore_words = extended_stop_words.union(group_words)
        
        # Filtre les mots-clés pour éviter de donner le nom de la catégorie comme mot-clé principal
        for word in common_words[:5]:
            if word not in final_ignore_words:
                final_keywords.append(word)
                
        # 5d. Détermination du KeywordMatch final
        if group_name == "Problème Inconnu":
            keywords_match = "non définis" # Si Inconnu, les mots-clés ne sont pas pertinents
        else:
            keywords_match = ", ".join(final_keywords) if final_keywords else "analyse sémantique"
        
        return group_name, keywords_match
        
    return group_name, "non définis"


# --- Fonction Principale du Pipeline ---

def run_full_analysis(df):
    """Exécute l'intégralité du pipeline d'analyse IA."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Début de l'analyse sur {len(df)} tickets assignés")
    
    # 1. Initialisation des colonnes de Clustering (Importante si le clustering échoue)
    df['ClusterID'] = 0 
    
    # 2. Calculs de Scores
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
    
    # 4. Clustering (seulement si le modèle est chargé)
    cluster_results = pd.DataFrame() 
    
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
                
                # Le ClusterID commence à 0
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
            print(f"❌ Erreur de Clustering ou de maketrans: {e}")
            # Si échec, ClusterID reste à 0 pour tous les tickets.
    
    # S'assurer que le ClusterID est un entier
    df['ClusterID'] = df['ClusterID'].fillna(0).astype(int)
    df_anomalies = df.copy()
    
    # 5. Finalisation des résultats de Clustering (SANS GESTION SPÉCIALE DU CLUSTER ID 0)
    if not cluster_results.empty:
        # Pas d'incrémentation : le ClusterID 0 correspond au premier cluster
        
        # Ajout d'une entrée pour les tickets non analysés (ceux restés à ID 0 car description manquante)
        tickets_restants = len(df_anomalies[df_anomalies['ClusterID'] == 0])
        if tickets_restants > 0:
            # S'assurer que le ClusterID 0 correspond bien aux tickets restants ou au premier cluster normal
            # Si le cluster 0 n'est pas dans les résultats, cela signifie qu'il ne contient que les tickets exclus.
            if 0 not in cluster_results['ClusterID'].values:
                 cluster_results.loc[len(cluster_results)] = {
                    'ProblemNameGroup': 'Ticket Non Analysé (Description Manquante)',
                    'ClusterID': 0, 
                    'KeywordMatch': 'Non applicable',
                    'RecurrenceCount': tickets_restants
                }

        cluster_results = cluster_results.sort_values(by='ClusterID').reset_index(drop=True)
        
    else:
        # Si le clustering a échoué: tout est assigné à l'ID 0
        df_anomalies['ClusterID'] = 0 
        cluster_results = pd.DataFrame([{
            'ProblemNameGroup': 'Échec de Classification IA',
            'ClusterID': 0,
            'KeywordMatch': 'Vérifiez les dépendances spacy et sentence-transformers',
            'RecurrenceCount': len(df)
        }])

    # 6. Sélection des colonnes pour la table FactAnomaliesDetail
    df_anomalies = df_anomalies[[
        'TicketID', 'FactKey', 'AssigneeEmployeeKey', 'AssigneeFullName', 
        'TicketNote', 'EmployeeAvgScore', 'ScoreSemantique', 'ScoreConcordance',
        'TempsHeures', 'TempsMoyenHeures', 'EcartTypeHeures', 'ScoreTemporel',
        'AnomalieTemporelle', 'Statut', 'AnomalyDescription', 
        'ClusterID' 
    ]].copy()
    
    return df_anomalies, cluster_results