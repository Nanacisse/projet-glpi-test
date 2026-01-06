import pandas as pd
import numpy as np
import spacy
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import re
from collections import Counter
import string
import language_tool_python
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import warnings
import time
import subprocess

# Supprimer les warnings
warnings.filterwarnings('ignore')

# --- Configuration et Initialisation ---

# Définition des constantes d'anomalie
SEMAN_THRESHOLD = 0.80  # 80% pour la sémantique
CONC_THRESHOLD = 0.20   # 20% pour la concordance
SLA_THRESHOLD = 4.0     # 4 heures pour le SLA

# --- CONSTANTES DE CLUSTERING INTELLIGENTES ---
MAX_TOTAL_CLUSTERS = 60            # Maximum ABSOLU (jamais dépasser)
IDEAL_TICKETS_PER_CLUSTER = 55     # Cible: ~55 tickets par cluster (3,500/60 ≈ 58)
MIN_TICKETS_PER_CLUSTER = 40       # Minimum pour un cluster significatif
MAX_TICKETS_PER_CLUSTER = 80       # Maximum avant de diviser
MIN_CLUSTER_SIZE = 3               # Minimum tickets pour créer un cluster
MAX_CATEGORIES_TO_USE = 25         # Maximum catégories DimCategory à utiliser

# Initialisation des ressources
nlp = None
st_model = None
tool = None

print("Chargement des modèles NLP...")
try:
    nlp = spacy.load("fr_core_news_sm")
    st_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    # Vérifier si Java est disponible pour language_tool
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            tool = language_tool_python.LanguageTool('fr', timeout=30)
            print("✓ Vérification grammaticale activée avec Java")
        else:
            print("⚠ Java non disponible - Vérification grammaticale désactivée")
            tool = None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as java_error:
        print(f"⚠ Java non détecté, désactivation de la vérification grammaticale: {java_error}")
        tool = None
    
    print("✓ Modèles NLP chargés avec succès")
    
except Exception as e:
    print(f"⚠ Erreur de chargement des modèles: {e}")
    print("Continuer avec les fonctionnalités de base...")
    nlp = None
    st_model = None
    tool = None

# --- Nouvelle analyse sémantique ---

def detect_vague_words_automatically(text: str, doc) -> List[str]:
    """
    Détecte automatiquement les mots vagues dans un texte.
    Utilise la grammaire et le contexte pour identifier les mots vagues.
    """
    vague_words = []
    
    try:
        modal_verbs = ['pouvoir', 'devoir', 'falloir', 'vouloir', 'sembler']
        
        uncertainty_adverbs = ['peut-être', 'probablement', 'éventuellement', 
                              'possiblement', 'apparemment', 'normalement',
                              'habituellement', 'généralement', 'souvent']
        
        generic_verbs = ['faire', 'mettre', 'prendre', 'voir', 'dire', 
                        'donner', 'rendre', 'laisser', 'passer']
        
        for token in doc:
            if token.lemma_ in modal_verbs and len(list(token.children)) < 2:
                vague_words.append(token.text)
            elif token.text.lower() in uncertainty_adverbs:
                vague_words.append(token.text)
            elif (token.pos_ == 'VERB' and token.lemma_ in generic_verbs and
                  not any(child.dep_ == 'obj' for child in token.children)):
                vague_words.append(token.text)
        
        sentences = list(doc.sents)
        for sent in sentences:
            words = [token.text.lower() for token in sent if token.is_alpha]
            unique_words = set(words)
            
            if len(words) < 8 and len(unique_words) < 6:
                vague_words.append("phrase_generique")
        
        return list(set(vague_words))
        
    except Exception as e:
        print(f"Erreur détection mots vagues: {e}")
        return []

def detect_structural_elements_automatically(doc) -> int:
    """
    Détecte automatiquement les éléments de structure dans un texte.
    Retourne le nombre d'étapes identifiées.
    """
    try:
        etapes_count = 0
        sentences = list(doc.sents)
        
        if not sentences:
            return 0
        
        numerical_markers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                            'premier', 'deuxième', 'troisième', 'quatrième',
                            'premièrement', 'deuxièmement', 'troisièmement']
        
        temporal_markers = ['ensuite', 'puis', 'après', 'alors', 'maintenant',
                           'finalement', 'enfin', 'ensuite']
        
        logical_markers = ['d\'abord', 'première étape', 'deuxième étape',
                          'étape suivante', 'dernière étape', 'étape finale']
        
        conditional_patterns = [r'si .* alors', r'lorsque .* donc',
                               r'après avoir .* ensuite']
        
        for sent in sentences:
            sent_text = sent.text.lower()
            
            for marker in numerical_markers + temporal_markers + logical_markers:
                if marker in sent_text:
                    etapes_count += 1
                    break
            
            for pattern in conditional_patterns:
                if re.search(pattern, sent_text):
                    etapes_count += 1
                    break
        
        return min(4, etapes_count)
        
    except Exception as e:
        print(f"Erreur détection structure: {e}")
        return 0

def calculate_semantique_score(text):
    """
    Calcule le score sémantique selon les 4 critères:
    1. Longueur des phrases (30 points)
    2. Structure logique (20 points) - DÉTECTION AUTOMATIQUE
    3. Qualité grammaticale (30 points)
    4. Détection des mots vagues (20 points) - DÉTECTION AUTOMATIQUE
    Total: 100 points convertis en pourcentage
    """
    if pd.isna(text) or not isinstance(text, str):
        return 0.0
    
    text_str = str(text).strip()
    if not text_str:
        return 0.0
    
    try:
        doc = nlp(text_str)
        sentences = list(doc.sents)
        
        if not sentences:
            return 0.0
        
        longueur_score = 30
        for sent in sentences:
            word_count = len([token for token in sent if not token.is_punct])
            if word_count > 25:
                longueur_score -= 5
                break
        
        etapes_trouvees = detect_structural_elements_automatically(doc)
        structure_score = etapes_trouvees * 5
        structure_score = min(20, structure_score)
        
        grammaire_score = 30
        if tool:
            try:
                matches = tool.check(text_str)
                nb_fautes = len(matches)
                nb_mots = len([token for token in doc if token.is_alpha])
                if nb_mots > 0:
                    taux_fautes = nb_fautes / nb_mots
                    grammaire_score = 30 * (1 - min(taux_fautes, 1))
            except Exception as e:
                print(f"⚠ Vérification grammaticale échouée: {e}")
                grammaire_score = 25
        else:
            grammaire_score = 25
        
        mots_vagues_trouves = detect_vague_words_automatically(text_str, doc)
        vague_score = 20 - (len(mots_vagues_trouves) * 4)
        vague_score = max(0, vague_score)
        
        total_points = longueur_score + structure_score + grammaire_score + vague_score
        pourcentage = (total_points / 100) * 100
        
        return min(100, round(pourcentage, 2))
        
    except Exception as e:
        print(f"Erreur analyse sémantique: {e}")
        return 50.0

def calculate_note_semantique(score_semantique):
    """Convertit le score sémantique (%) en note sur 10."""
    return round((score_semantique / 100) * 10, 2)

# --- Nouvelle analyse de concordance ---

def detect_resolution_keywords_automatically(solution_text: str, doc) -> bool:
    """
    Détecte automatiquement si la solution contient des mots-clés de résolution.
    Retourne True si au moins un mot-clé de résolution est détecté.
    """
    try:
        solution_lower = solution_text.lower()
        
        resolution_patterns = [
            r'problème (?:est|a été) (?:résolu|corrigé|réparé|fixé)',
            r'(?:j\'ai|nous avons) (?:résolu|corrigé|réparé)',
            r'solution (?:est|a été) (?:trouvée|appliquée|mise en œuvre)',
            r'ticket (?:est|a été) (?:clôturé|fermé|terminé)',
            r'incident (?:est|a été) (?:traité|réglé)'
        ]
        
        for pattern in resolution_patterns:
            if re.search(pattern, solution_lower):
                return True
        
        resolution_verbs = ['résoudre', 'corriger', 'réparer', 'fixer',
                           'terminer', 'clôturer', 'traiter', 'régler']
        
        for token in doc:
            if token.lemma_ in resolution_verbs and token.pos_ == 'VERB':
                children = list(token.children)
                if not any(child.dep_ == 'neg' for child in children):
                    return True
        
        resolution_nouns = ['solution', 'résolution', 'correction', 'réparation']
        
        for token in doc:
            if token.lemma_ in resolution_nouns and token.pos_ == 'NOUN':
                return True
        
        return False
        
    except Exception as e:
        print(f"Erreur détection mots-clés résolution: {e}")
        return False

def detect_completion_indicators_automatically(solution_text: str, doc) -> bool:
    """
    Détecte automatiquement si la solution contient des indicateurs de complétion.
    Retourne True si au moins un indicateur est détecté.
    """
    try:
        solution_lower = solution_text.lower()
        
        completion_patterns = [
            r'(?:a été|est) (?:validé|vérifié|testé|confirmé)',
            r'(?:j\'ai|nous avons) (?:vérifié|testé|validé)',
            r'(?:fonctionne|opérationnel|en marche) (?:correctement|normalement)',
            r'(?:mise en œuvre|implémentation) (?:terminée|achevée|complète)',
            r'(?:installation|configuration) (?:finalisée|achevée)'
        ]
        
        for pattern in completion_patterns:
            if re.search(pattern, solution_lower):
                return True
        
        completion_verbs = ['valider', 'vérifier', 'tester', 'confirmer',
                           'exécuter', 'appliquer', 'implémenter', 'installer']
        
        for token in doc:
            if token.lemma_ in completion_verbs and token.pos_ == 'VERB':
                if token.morph.get('Tense') in ['Past', 'Pres']:
                    return True
        
        time_indicators = ['maintenant', 'actuellement', 'désormais', 'à présent']
        
        for token in doc:
            if token.text.lower() in time_indicators:
                return True
        
        return False
        
    except Exception as e:
        print(f"Erreur détection indicateurs complétion: {e}")
        return False

def calculate_concordance_score(problem, solution):
    """
    Calcule le score de concordance selon les 3 critères:
    1. Similarité sémantique (20 points)
    2. Mots-clés de résolution (40 points) - DÉTECTION AUTOMATIQUE
    3. Indicateurs de complétion (40 points) - DÉTECTION AUTOMATIQUE
    Total: 100 points convertis en pourcentage
    """
    if pd.isna(problem) or pd.isna(solution):
        return 0.0
    
    problem_str = str(problem).strip()
    solution_str = str(solution).strip()
    
    if not problem_str or not solution_str:
        return 0.0
    
    try:
        solution_doc = nlp(solution_str) if nlp else None
        
        similarite_score = 0
        if st_model and problem_str and solution_str:
            try:
                embeddings = st_model.encode([problem_str, solution_str])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                
                if similarity >= 0.65:
                    similarite_score = 20
                elif 0.50 <= similarity < 0.65:
                    similarite_score = 15
                elif 0.30 <= similarity < 0.50:
                    similarite_score = 10
                else:
                    similarite_score = 5
            except:
                similarite_score = 5
        
        resolution_detected = detect_resolution_keywords_automatically(solution_str, solution_doc)
        resolution_score = 40 if resolution_detected else 0
        
        completion_detected = detect_completion_indicators_automatically(solution_str, solution_doc)
        completion_score = 40 if completion_detected else 0
        
        total_points = similarite_score + resolution_score + completion_score
        pourcentage = (total_points / 100) * 100
        
        return min(100, round(pourcentage, 2))
        
    except Exception as e:
        print(f"Erreur calcul concordance: {e}")
        return 50.0

def calculate_note_concordance(score_concordance):
    """Convertit le score de concordance (%) en note sur 10."""
    return round((score_concordance / 100) * 10, 2)

# --- Nouvelle analyse temporelle ---

def calculate_temporal_note(temps_heures):
    """
    Calcule la note temporelle sur 10 selon le SLA:
    ≤ 4h : 10.0
    4-8h : 5.0
    8-24h : 3.0
    >24h : 2.0
    """
    if pd.isna(temps_heures):
        return 0.0
    
    if temps_heures <= 4.0:
        return 10.0
    elif 4.0 < temps_heures <= 8.0:
        return 5.0
    elif 8.0 < temps_heures <= 24.0:
        return 3.0
    else:
        return 2.0

# --- Calcul de la note finale sur 10 ---

def calculate_final_note(row):
    """
    Calcule la note finale sur 10 = Note Temporelle (50%) + Note Sémantique (40%) + Note Concordance (10%)
    """
    note_temporelle = row.get('NoteTemporelle', 0)
    note_semantique = row.get('NoteSemantique', 0)
    note_concordance = row.get('NoteConcordance', 0)
    
    note_finale = (note_temporelle * 0.50) + (note_semantique * 0.40) + (note_concordance * 0.10)
    return round(note_finale, 2)

# --- Calcul de la moyenne employé ---

def calculate_employee_average(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la moyenne des notes pour chaque employé selon la méthode demandée:
    1. Regroupement par employé
    2. Somme des notes pour chaque groupe
    3. Division par le nombre de tickets
    
    Formule: Moyenne Employé = Somme des notes / nombre de tickets
    """
    if 'AssigneeEmployeeKey' not in df.columns or 'TicketNote' not in df.columns:
        return df
    
    df_copy = df.copy()
    
    employee_stats = df_copy.groupby('AssigneeEmployeeKey').agg({
        'TicketNote': ['sum', 'count']
    }).reset_index()
    
    employee_stats.columns = ['AssigneeEmployeeKey', 'TotalNotes', 'TicketCount']
    
    employee_stats['EmployeeAvgScore'] = employee_stats.apply(
        lambda row: round(row['TotalNotes'] / row['TicketCount'], 2) if row['TicketCount'] > 0 else 0,
        axis=1
    )
    
    df_copy = df_copy.merge(
        employee_stats[['AssigneeEmployeeKey', 'EmployeeAvgScore']],
        on='AssigneeEmployeeKey',
        how='left'
    )
    
    return df_copy

# --- Détermination du statut ---

def determine_final_status(row):
    """Détermine le statut final basé sur les 3 indicateurs d'anomalie."""
    sem_ok = row['ScoreSemantique'] >= SEMAN_THRESHOLD * 100
    conc_ok = row['ScoreConcordance'] >= CONC_THRESHOLD * 100
    temp_ok = row['TempsHeures'] <= SLA_THRESHOLD
    
    if sem_ok and conc_ok and temp_ok:
        return 'OK'
    
    num_anomalies = sum([not sem_ok, not conc_ok, not temp_ok])
    
    if num_anomalies >= 2:
        return 'Multiples Anomalies'
    elif not temp_ok:
        return 'Anomalie de Temps'
    elif not conc_ok:
        return 'Anomalie de Concordance'
    elif not sem_ok:
        return 'Anomalie Sémantique'
    
    return 'Anomalie Indéterminée'

# --- Clustering pour problèmes récurrents ---

def extract_keywords_automatically(descriptions: List[str]) -> str:
    """Extrait automatiquement les mots-clés les plus pertinents d'une liste de descriptions."""
    if not descriptions:
        return "Aucun mot-clé"
    
    try:
        all_text = ' '.join(descriptions)
        doc = nlp(all_text.lower())
        
        relevant_words = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
                not token.is_stop and 
                len(token.text) > 3 and
                token.text.isalpha()):
                relevant_words.append(token.lemma_)
        
        if relevant_words:
            word_counts = Counter(relevant_words)
            top_words = [word for word, count in word_counts.most_common(5)]
            return ', '.join(top_words)
        
        return "Aucun mot-clé significatif"
        
    except:
        return "Extraction échouée"

def generate_group_name_from_keywords(keywords: str) -> str:
    """Génère un nom de groupe basé sur les mots-clés."""
    if not keywords or keywords == "Aucun mot-clé":
        return "Problème Divers"
    
    first_keyword = keywords.split(',')[0].strip()
    
    if len(first_keyword) > 3:
        return f"Problème de {first_keyword.capitalize()}"
    else:
        return "Problème Technique"

def perform_advanced_clustering(df: pd.DataFrame, categories_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Effectue le clustering avancé pour les problèmes récurrents avec maximum 60 clusters RÉELS.
    """
    
    cluster_results = []
    df_with_clusters = df.copy()
    df_with_clusters['ClusterID'] = -1
    df_with_clusters['CategoryID'] = 0
    
    total_tickets = len(df)
    print(f"📊 Début clustering sur {total_tickets} tickets")
    
    try:
        # === ÉTAPE 1: Clusters par catégories DimCategory (UNIQUEMENT si significatifs) ===
        categories_used = 0
        if not categories_data.empty:
            print(f"🔍 Recherche correspondance avec {len(categories_data)} catégories DimCategory...")
            
            # Trier les catégories par pertinence potentielle
            category_matches = []
            
            for idx, category_row in categories_data.iterrows():
                if categories_used >= MAX_CATEGORIES_TO_USE:
                    break
                    
                category_id = category_row['CategoryID']
                category_name = category_row['CategoryName']
                category_desc = str(category_row.get('Description', ''))
                
                matching_indices = []
                for ticket_idx, row in df.iterrows():
                    problem_desc = str(row.get('ProblemDescription', '')).lower()
                    solution_desc = str(row.get('SolutionContent', '')).lower()
                    
                    # Recherche dans problème ET solution
                    if (category_name.lower() in problem_desc or 
                        category_name.lower() in solution_desc or
                        (category_desc and category_desc.lower() in problem_desc) or
                        (category_desc and category_desc.lower() in solution_desc)):
                        matching_indices.append(ticket_idx)
                
                if matching_indices:
                    category_matches.append({
                        'category_id': category_id,
                        'category_name': category_name,
                        'indices': matching_indices,
                        'count': len(matching_indices)
                    })
            
            # Trier par nombre de matches (décroissant)
            category_matches.sort(key=lambda x: x['count'], reverse=True)
            
            # Prendre les meilleures catégories (celles avec le plus de tickets)
            for match in category_matches:
                if categories_used >= MAX_CATEGORIES_TO_USE:
                    break
                    
                if match['count'] >= MIN_CLUSTER_SIZE:  # Au moins 3 tickets
                    cluster_id = len(cluster_results)
                    
                    df_with_clusters.loc[match['indices'], 'ClusterID'] = cluster_id
                    df_with_clusters.loc[match['indices'], 'CategoryID'] = match['category_id']
                    
                    # Extraire les descriptions pour mots-clés
                    descriptions = []
                    for idx in match['indices']:
                        if pd.notna(df.loc[idx, 'ProblemDescription']):
                            descriptions.append(str(df.loc[idx, 'ProblemDescription']))
                        if pd.notna(df.loc[idx, 'SolutionContent']):
                            descriptions.append(str(df.loc[idx, 'SolutionContent']))
                    
                    keywords = extract_keywords_automatically(descriptions)
                    
                    cluster_results.append({
                        'ProblemNameGroup': match['category_name'],
                        'ClusterID': cluster_id,
                        'KeywordMatch': keywords if keywords else match['category_name'],
                        'RecurrenceCount': match['count'],
                        'CategoryID': match['category_id']
                    })
                    
                    categories_used += 1
                    print(f"  ✓ Catégorie '{match['category_name']}': {match['count']} tickets")
        
        print(f"✅ {categories_used} clusters catégories créés (min {MIN_CLUSTER_SIZE} tickets)")
        
        # === ÉTAPE 2: Calcul intelligent du nombre de clusters nécessaires ===
        remaining_indices = df_with_clusters[df_with_clusters['ClusterID'] == -1].index.tolist()
        remaining_tickets = len(remaining_indices)
        
        print(f"📦 Tickets restants à clusteriser: {remaining_tickets}")
        
        if remaining_tickets > 0:
            # Calcul du nombre optimal de clusters
            slots_available = MAX_TOTAL_CLUSTERS - len(cluster_results)
            
            # Calcul basé sur ratio idéal
            clusters_by_ratio = remaining_tickets // IDEAL_TICKETS_PER_CLUSTER
            
            # Ajustement: prendre le minimum entre ratio et slots disponibles
            clusters_needed = min(clusters_by_ratio, slots_available)
            
            # Minimum de clusters si assez de tickets
            if remaining_tickets > 100 and clusters_needed < 5:
                clusters_needed = min(5, slots_available)
            
            # Maximum pour éviter les clusters trop petits
            max_by_min_size = remaining_tickets // MIN_TICKETS_PER_CLUSTER
            clusters_needed = min(clusters_needed, max_by_min_size)
            
            print(f"🎯 Calcul clusters nécessaires:")
            print(f"   - Par ratio ({IDEAL_TICKETS_PER_CLUSTER} tickets/cluster): {clusters_by_ratio}")
            print(f"   - Slots disponibles: {slots_available}")
            print(f"   - Clusters décidés: {clusters_needed}")
            print(f"   - Ratio final: {remaining_tickets/clusters_needed:.1f} tickets/cluster")
            
            # === ÉTAPE 3: Clustering hiérarchique ===
            if clusters_needed >= 2 and st_model and remaining_tickets >= 10:
                print(f"🔗 Clustering hiérarchique pour {remaining_tickets} tickets...")
                
                # Limite pratique pour performance
                MAX_TICKETS_FOR_CLUSTERING = 2500
                if remaining_tickets > MAX_TICKETS_FOR_CLUSTERING:
                    print(f"  ⚠ Échantillonnage à {MAX_TICKETS_FOR_CLUSTERING} tickets")
                    # Prendre un échantillon représentatif
                    sample_indices = np.random.choice(
                        remaining_indices, 
                        size=MAX_TICKETS_FOR_CLUSTERING, 
                        replace=False
                    )
                    remaining_indices = sample_indices.tolist()
                    remaining_tickets = len(remaining_indices)
                
                # Préparer les descriptions
                descriptions_to_cluster = []
                valid_indices = []
                
                for idx in remaining_indices:
                    problem_desc = str(df.loc[idx, 'ProblemDescription'])
                    if problem_desc and len(problem_desc.strip()) > 10:
                        descriptions_to_cluster.append(problem_desc)
                        valid_indices.append(idx)
                
                if len(descriptions_to_cluster) >= clusters_needed:
                    print(f"  📝 Encodage de {len(descriptions_to_cluster)} descriptions...")
                    embeddings = st_model.encode(descriptions_to_cluster, show_progress_bar=False)
                    
                    print(f"  🎯 Création de {clusters_needed} clusters...")
                    
                    # Utiliser MiniBatchKMeans pour performance
                    try:
                        from sklearn.cluster import MiniBatchKMeans
                        clustering = MiniBatchKMeans(
                            n_clusters=clusters_needed,
                            random_state=42,
                            batch_size=1000,
                            n_init=3,
                            max_iter=100
                        )
                        cluster_labels = clustering.fit_predict(embeddings)
                        print(f"  ✅ MiniBatchKMeans terminé")
                    except Exception as km_error:
                        print(f"  ⚠ MiniBatchKMeans échoué, fallback à AgglomerativeClustering")
                        clustering = AgglomerativeClustering(
                            n_clusters=clusters_needed,
                            metric='cosine',
                            linkage='average'
                        )
                        cluster_labels = clustering.fit_predict(embeddings)
                    
                    # Organiser les résultats par cluster
                    cluster_groups = {}
                    for idx, cluster_label in zip(valid_indices, cluster_labels):
                        if cluster_label not in cluster_groups:
                            cluster_groups[cluster_label] = []
                        cluster_groups[cluster_label].append(idx)
                    
                    # Créer les clusters RÉELS (uniquement si assez de tickets)
                    clusters_created = 0
                    for cluster_label, indices in cluster_groups.items():
                        if len(indices) >= MIN_CLUSTER_SIZE:  # Au moins 3 tickets
                            cluster_id = len(cluster_results)
                            
                            # Vérifier limite absolue
                            if cluster_id >= MAX_TOTAL_CLUSTERS:
                                print(f"  ⚠ Limite de {MAX_TOTAL_CLUSTERS} clusters atteinte")
                                break
                            
                            df_with_clusters.loc[indices, 'ClusterID'] = cluster_id
                            
                            # Extraire descriptions pour nom et mots-clés
                            cluster_descriptions = []
                            for idx in indices:
                                if pd.notna(df.loc[idx, 'ProblemDescription']):
                                    cluster_descriptions.append(str(df.loc[idx, 'ProblemDescription']))
                            
                            keywords = extract_keywords_automatically(cluster_descriptions)
                            group_name = generate_group_name_from_keywords(keywords)
                            
                            # Vérifier association avec catégorie existante
                            cluster_category_id = 0
                            if categories_data is not None:
                                cluster_text = ' '.join(cluster_descriptions).lower()
                                for _, cat_row in categories_data.iterrows():
                                    cat_name = str(cat_row['CategoryName']).lower()
                                    if cat_name in cluster_text:
                                        cluster_category_id = cat_row['CategoryID']
                                        group_name = f"{cat_row['CategoryName']} ({group_name})"
                                        break
                            
                            cluster_results.append({
                                'ProblemNameGroup': group_name,
                                'ClusterID': cluster_id,
                                'KeywordMatch': keywords,
                                'RecurrenceCount': len(indices),
                                'CategoryID': cluster_category_id
                            })
                            
                            df_with_clusters.loc[indices, 'CategoryID'] = cluster_category_id
                            clusters_created += 1
                    
                    print(f"  ✅ {clusters_created} clusters hiérarchiques créés")
                else:
                    print(f"  ⚠ Pas assez de descriptions valides pour clustering")
        
        # === ÉTAPE 4: Gestion des tickets non clusterisés ===
        non_clustered = df_with_clusters[df_with_clusters['ClusterID'] == -1]
        if not non_clustered.empty:
            print(f"📌 {len(non_clustered)} tickets non clusterisés")
            
            # Si peu de tickets, les ajouter au cluster le plus proche
            if len(non_clustered) < 10 and len(cluster_results) > 0:
                # Trouver le cluster avec le plus de tickets
                largest_cluster_id = max(cluster_results, key=lambda x: x['RecurrenceCount'])['ClusterID']
                df_with_clusters.loc[non_clustered.index, 'ClusterID'] = largest_cluster_id
                print(f"  ➕ Ajoutés au cluster #{largest_cluster_id}")
            elif len(non_clustered) >= MIN_CLUSTER_SIZE and len(cluster_results) < MAX_TOTAL_CLUSTERS:
                # Créer un cluster "Divers"
                cluster_id = len(cluster_results)
                df_with_clusters.loc[non_clustered.index, 'ClusterID'] = cluster_id
                
                cluster_results.append({
                    'ProblemNameGroup': 'Problèmes Divers',
                    'ClusterID': cluster_id,
                    'KeywordMatch': 'Non classifié',
                    'RecurrenceCount': len(non_clustered),
                    'CategoryID': -1
                })
                print(f"  ✅ Cluster 'Divers' créé avec {len(non_clustered)} tickets")
            else:
                # Distribuer parmi les clusters existants
                for idx in non_clustered.index:
                    # Trouver le cluster avec le moins de tickets
                    if cluster_results:
                        smallest_cluster = min(cluster_results, key=lambda x: x['RecurrenceCount'])
                        df_with_clusters.loc[idx, 'ClusterID'] = smallest_cluster['ClusterID']
                        smallest_cluster['RecurrenceCount'] += 1
        
        # === ÉTAPE 5: Finalisation ===
        # Convertir en DataFrame
        cluster_results_df = pd.DataFrame(cluster_results)
        
        if not cluster_results_df.empty:
            # Trier par nombre d'occurrences
            cluster_results_df = cluster_results_df.sort_values('RecurrenceCount', ascending=False)
            
            # Réassigner les IDs de 0 à N-1
            cluster_results_df = cluster_results_df.reset_index(drop=True)
            cluster_results_df['ClusterID'] = range(len(cluster_results_df))
            
            # Mettre à jour les IDs dans df_with_clusters
            id_mapping = {}
            for new_id, row in cluster_results_df.iterrows():
                old_id = row['ClusterID']
                id_mapping[old_id] = new_id
            
            df_with_clusters['ClusterID'] = df_with_clusters['ClusterID'].map(id_mapping)
        
        # Statistiques finales
        clustered_tickets = len(df_with_clusters[df_with_clusters['ClusterID'] != -1])
        final_cluster_count = len(cluster_results_df)
        
        print(f"\n📊 RÉSULTATS FINAUX DU CLUSTERING:")
        print(f"   ✅ Clusters totaux: {final_cluster_count}")
        print(f"   ✅ Tickets clusterisés: {clustered_tickets}/{total_tickets} ({clustered_tickets/total_tickets*100:.1f}%)")
        print(f"   ✅ Ratio moyen: {clustered_tickets/max(1, final_cluster_count):.1f} tickets/cluster")
        
        if final_cluster_count > 0:
            avg_size = cluster_results_df['RecurrenceCount'].mean()
            min_size = cluster_results_df['RecurrenceCount'].min()
            max_size = cluster_results_df['RecurrenceCount'].max()
            print(f"   📈 Taille clusters: min={min_size}, max={max_size}, avg={avg_size:.1f}")
        
        # Vérifier limite
        if final_cluster_count > MAX_TOTAL_CLUSTERS:
            print(f"⚠ ATTENTION: {final_cluster_count} clusters > limite {MAX_TOTAL_CLUSTERS}")
            print(f"   Troncature à {MAX_TOTAL_CLUSTERS} clusters...")
            cluster_results_df = cluster_results_df.head(MAX_TOTAL_CLUSTERS)
        
        print(f"🎯 OBJECTIF ATTEINT: {len(cluster_results_df)} clusters (max {MAX_TOTAL_CLUSTERS})")
        
        return cluster_results_df, df_with_clusters
        
    except Exception as e:
        print(f"❌ Erreur clustering avancé: {e}")
        import traceback
        traceback.print_exc()
        
        # Solution de repli: un seul cluster
        df_with_clusters['ClusterID'] = 0
        df_with_clusters['CategoryID'] = 0
        
        default_cluster = pd.DataFrame([{
            'ProblemNameGroup': 'Tous les Problèmes',
            'ClusterID': 0,
            'KeywordMatch': 'Erreur de clustering',
            'RecurrenceCount': len(df),
            'CategoryID': 0
        }])
        
        return default_cluster, df_with_clusters

# --- Fonction principale du pipeline ---

def run_full_analysis(df):
    """Exécute l'intégralité du pipeline d'analyse IA avec optimisation."""
    if df.empty:
        print("DataFrame vide reçu")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"🚀 Début de l'analyse sur {len(df)} tickets assignés")
    start_time = time.time()
    
    df['ClusterID'] = 0
    df['CategoryID'] = 0
    
    print("📝 Calcul des scores sémantiques...")
    # Traitement optimisé par lots
    batch_size = 500
    scores_semantiques = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_scores = batch['SolutionContent'].apply(calculate_semantique_score)
        scores_semantiques.extend(batch_scores)
        if i % 1500 == 0 and i > 0:
            print(f"  Progression: {i}/{len(df)} tickets")
    
    df['ScoreSemantique'] = scores_semantiques
    df['NoteSemantique'] = df['ScoreSemantique'].apply(calculate_note_semantique)
    
    print("📝 Calcul des scores de concordance...")
    scores_concordance = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_scores = batch.apply(
            lambda row: calculate_concordance_score(row['ProblemDescription'], row['SolutionContent']),
            axis=1
        )
        scores_concordance.extend(batch_scores)
        if i % 1500 == 0 and i > 0:
            print(f"  Progression: {i}/{len(df)} tickets")
    
    df['ScoreConcordance'] = scores_concordance
    df['NoteConcordance'] = df['ScoreConcordance'].apply(calculate_note_concordance)
    
    print("⏱️ Calcul des notes temporelles...")
    df['NoteTemporelle'] = df['TempsHeures'].apply(calculate_temporal_note)
    
    print("🧮 Calcul des notes finales...")
    df['TicketNote'] = df.apply(calculate_final_note, axis=1)
    
    print("🏷️ Détermination des statuts...")
    df['Statut'] = df.apply(determine_final_status, axis=1)
    
    print("👥 Calcul des moyennes employé...")
    df = calculate_employee_average(df)
    
    print("🔗 Clustering avancé en cours...")
    cluster_results = pd.DataFrame()
    try:
        from utils.db_connector import load_categories_data
        categories_data = load_categories_data()
        cluster_results, df_with_clusters = perform_advanced_clustering(df, categories_data)
        
        df['ClusterID'] = df_with_clusters['ClusterID']
        df['CategoryID'] = df_with_clusters['CategoryID']
        
    except Exception as e:
        print(f"⚠ Erreur clustering: {e}")
        cluster_results = pd.DataFrame()
    
    print("📊 Préparation des résultats...")
    df_anomalies = df[[
        'TicketID', 'FactKey', 'AssigneeEmployeeKey', 'AssigneeFullName',
        'TicketNote', 'EmployeeAvgScore', 
        'ScoreSemantique', 'NoteSemantique',
        'ScoreConcordance', 'NoteConcordance',
        'TempsHeures', 'NoteTemporelle',
        'Statut', 'ClusterID', 'CategoryID'
    ]].copy()
    
    # Nettoyage des données numériques
    numeric_cols = ['TicketNote', 'EmployeeAvgScore', 'NoteSemantique', 'NoteConcordance', 'NoteTemporelle']
    for col in numeric_cols:
        if col in df_anomalies.columns:
            df_anomalies[col] = pd.to_numeric(df_anomalies[col], errors='coerce').fillna(0).round(2)
    
    # Calcul du temps d'exécution
    end_time = time.time()
    execution_time = end_time - start_time
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)
    
    print(f"\n✅ ✅ ✅ ANALYSE TERMINÉE AVEC SUCCÈS!")
    print(f"   ⏱️ Temps total: {minutes}m {seconds}s")
    print(f"   📊 Tickets analysés: {len(df_anomalies)}")
    print(f"   🔗 Clusters créés: {len(cluster_results)} (max {MAX_TOTAL_CLUSTERS})")
    
    return df_anomalies, cluster_results