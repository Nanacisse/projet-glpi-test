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
import threading
from queue import Queue

# Supprimer les warnings
warnings.filterwarnings('ignore')

# --- Configuration et Initialisation ---

# Définition des constantes d'anomalie
SEMAN_THRESHOLD = 0.80  # 80% pour la sémantique
CONC_THRESHOLD = 0.20   # 20% pour la concordance
SLA_THRESHOLD = 4.0     # 4 heures pour le SLA

# --- CONSTANTES D'OPTIMISATION ---
MAX_TOTAL_CLUSTERS = 60            # Maximum ABSOLU
IDEAL_TICKETS_PER_CLUSTER = 55     # Cible: ~55 tickets par cluster
MIN_TICKETS_PER_CLUSTER = 40       # Minimum pour cluster significatif
MAX_TICKETS_PER_CLUSTER = 80       # Maximum avant division
MIN_CLUSTER_SIZE = 3               # Minimum tickets pour créer cluster
MAX_CATEGORIES_TO_USE = 25         # Maximum catégories DimCategory
MAX_TICKETS_FOR_CLUSTERING = 1500  # LIMITE pour performance
GRAMMAR_CHECK_TIMEOUT = 2          # Timeout vérification grammaticale
MAX_TEXT_LENGTH_FOR_GRAMMAR = 500  # Limite longueur texte

# Initialisation des ressources
nlp = None
st_model = None
tool = None

def initialize_nlp_models():
    """Initialise les modèles NLP de manière différée."""
    global nlp, st_model, tool
    
    if nlp is None:
        try:
            print("Chargement des modèles NLP...")
            # CORRECTION : Ne pas désactiver 'parser' pour avoir les phrases
            nlp = spacy.load("fr_core_news_sm", disable=['ner', 'textcat'])
            print("spaCy charge avec detection de phrases")
            
            # Charger SentenceTransformer (modèle plus rapide)
            st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("SentenceTransformer charge")
            
            # Initialiser language_tool_python avec timeout
            try:
                # Vérifier si Java est disponible
                print("Verification de Java...")
                result = subprocess.run(['java', '-version'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print(f"Java detecte: {result.stderr.splitlines()[0]}")
                    tool = language_tool_python.LanguageTool('fr')
                    print("Verification grammaticale activee avec Java")
                else:
                    print("Java non disponible - Verification grammaticale desactivee")
                    tool = None
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as java_error:
                print(f"Erreur Java: {java_error}")
                print("Verification grammaticale desactivee")
                tool = None
            
            print("Modeles NLP charges avec succes")
            
        except Exception as e:
            print(f"Erreur chargement modeles: {e}")
            print("Continuer avec fonctionnalites de base...")
            nlp = None
            st_model = None
            tool = None

# Appel initial
initialize_nlp_models()

# --- Fonction de vérification grammaticale avec TIMEOUT ---
def check_grammar_with_timeout(text: str, timeout: int = GRAMMAR_CHECK_TIMEOUT) -> int:
    """Vérifie la grammaire avec timeout pour éviter les blocages."""
    if not tool or not text or len(text.strip()) < 20:
        return 0
    
    # Limiter la longueur du texte pour performance
    text_to_check = text[:MAX_TEXT_LENGTH_FOR_GRAMMAR]
    
    try:
        result_queue = Queue()
        
        def grammar_check():
            try:
                matches = tool.check(text_to_check)
                result_queue.put(len(matches))
            except Exception as e:
                print(f"Erreur verification grammaire: {e}")
                result_queue.put(0)
        
        # Lancer dans un thread séparé
        check_thread = threading.Thread(target=grammar_check)
        check_thread.daemon = True
        check_thread.start()
        
        # Attendre avec timeout
        check_thread.join(timeout=timeout)
        
        if check_thread.is_alive():
            print(f"Timeout verification grammaire apres {timeout}s")
            return 0
        else:
            # Récupérer le résultat
            return result_queue.get() if not result_queue.empty() else 0
            
    except Exception as e:
        print(f"Erreur dans check_grammar_with_timeout: {e}")
        return 0

# --- Nouvelle analyse sémantique CORRIGÉE ---

def detect_vague_words_automatically(text: str, doc) -> List[str]:
    """Détecte automatiquement les mots vagues dans un texte."""
    vague_words = []
    
    try:
        modal_verbs = ['pouvoir', 'devoir', 'falloir', 'vouloir', 'sembler']
        uncertainty_adverbs = ['peut-être', 'probablement', 'éventuellement', 'possiblement']
        generic_verbs = ['faire', 'mettre', 'prendre', 'voir', 'dire']
        
        for token in doc:
            if token.lemma_ in modal_verbs and len(list(token.children)) < 2:
                vague_words.append(token.text)
            elif token.text.lower() in uncertainty_adverbs:
                vague_words.append(token.text)
            elif (token.pos_ == 'VERB' and token.lemma_ in generic_verbs and
                  not any(child.dep_ == 'obj' for child in token.children)):
                vague_words.append(token.text)
        
        # Détection phrases génériques
        try:
            sentences = list(doc.sents)
            for sent in sentences:
                words = [token.text.lower() for token in sent if token.is_alpha]
                if len(words) < 8 and len(set(words)) < 6:
                    vague_words.append("phrase_generique")
        except:
            pass
        
        return list(set(vague_words))
        
    except Exception as e:
        print(f"Erreur detection mots vagues: {e}")
        return []

def detect_structural_elements_automatically(doc) -> int:
    """Détecte automatiquement les éléments de structure dans un texte."""
    try:
        etapes_count = 0
        try:
            sentences = list(doc.sents)
        except:
            # Si pas de détection de phrases, utiliser une méthode simple
            text = doc.text.lower()
            markers = ['premier', 'deuxième', 'troisième', 'ensuite', 'puis', 
                      'après', 'finalement', 'enfin', 'd\'abord']
            
            for marker in markers:
                if marker in text:
                    etapes_count += 1
            
            return min(4, etapes_count)
        
        if not sentences:
            return 0
        
        markers = ['premier', 'deuxième', 'troisième', 'ensuite', 'puis', 
                  'après', 'finalement', 'enfin', 'd\'abord']
        
        for sent in sentences:
            sent_text = sent.text.lower()
            if any(marker in sent_text for marker in markers):
                etapes_count += 1
        
        return min(4, etapes_count)
        
    except Exception as e:
        print(f"Erreur detection structure: {e}")
        return 0

def calculate_semantique_score(text):
    """Calcule le score sémantique CORRIGÉ."""
    if pd.isna(text) or not isinstance(text, str):
        return 50.0
    
    text_str = str(text).strip()
    if not text_str or len(text_str) < 10:
        return 50.0
    
    # CACHE pour éviter les recalculs
    if hasattr(calculate_semantique_score, '_cache'):
        cache = calculate_semantique_score._cache
        text_hash = hash(text_str)
        if text_hash in cache:
            return cache[text_hash]
    else:
        calculate_semantique_score._cache = {}
    
    try:
        # Initialiser NLP si nécessaire
        if nlp is None:
            initialize_nlp_models()
        
        # Textes courts -> analyse simplifiée
        if len(text_str) < 30:
            # Score basé sur la longueur pour texte court
            base_score = 50.0
            if len(text_str) < 15:
                base_score = 40.0
            
            calculate_semantique_score._cache[hash(text_str)] = base_score
            return base_score
        
        # CORRECTION : Utiliser nlp avec meilleure gestion des erreurs
        doc = nlp(text_str)
        
        # 1. Longueur des phrases (30 points)
        longueur_score = 30
        try:
            sentences = list(doc.sents)
            if sentences:
                for sent in sentences:
                    word_count = len([token for token in sent if not token.is_punct])
                    if word_count > 25:
                        longueur_score -= 5
                        break
            else:
                # Si pas de phrases détectées, pénaliser un peu
                longueur_score = 20
        except:
            longueur_score = 20
        
        # 2. Structure logique (20 points)
        etapes_trouvees = detect_structural_elements_automatically(doc)
        structure_score = min(20, etapes_trouvees * 5)
        
        # 3. Qualité grammaticale (30 points) - OPTIMISÉ avec timeout
        grammaire_score = 30
        if tool:
            try:
                nb_fautes = check_grammar_with_timeout(text_str, GRAMMAR_CHECK_TIMEOUT)
                nb_mots = len([token for token in doc if token.is_alpha])
                if nb_mots > 0:
                    taux_fautes = nb_fautes / nb_mots
                    grammaire_score = 30 * (1 - min(taux_fautes, 1))
            except Exception as e:
                print(f"Erreur calcul grammaire: {e}")
                grammaire_score = 25
        else:
            grammaire_score = 25
        
        # 4. Mots vagues (20 points)
        mots_vagues = detect_vague_words_automatically(text_str, doc)
        vague_score = max(0, 20 - (len(mots_vagues) * 4))
        
        # Calcul final
        total_points = longueur_score + structure_score + grammaire_score + vague_score
        score = min(100, max(0, round(total_points, 2)))  # Entre 0 et 100
        
        # Debug: Afficher le détail pour quelques textes
        if calculate_semantique_score.debug_count < 3:
            print(f"  [DEBUG] Score sémantique pour texte (premiers 100 chars): '{text_str[:100]}...'")
            print(f"    Longueur: {longueur_score}/30")
            print(f"    Structure: {structure_score}/20")
            print(f"    Grammaire: {grammaire_score}/30")
            print(f"    Mots vagues: {vague_score}/20 (mots: {mots_vagues})")
            print(f"    TOTAL: {score}/100")
            calculate_semantique_score.debug_count += 1
        
        # Mettre en cache
        calculate_semantique_score._cache[hash(text_str)] = score
        return score
        
    except Exception as e:
        print(f"Erreur calcul score sémantique: {e}")
        # En cas d'erreur, retourner un score variable selon la longueur
        if len(text_str) > 100:
            score = 60.0
        elif len(text_str) > 50:
            score = 55.0
        else:
            score = 50.0
            
        calculate_semantique_score._cache[hash(text_str)] = score
        return score

# Initialiser le compteur de debug
calculate_semantique_score.debug_count = 0
calculate_semantique_score._cache = {}

def calculate_note_semantique(score_semantique):
    """Convertit le score sémantique (%) en note sur 10."""
    return round((score_semantique / 100) * 10, 2)

# --- Nouvelle analyse de concordance OPTIMISÉE ---

def detect_resolution_keywords_automatically(solution_text: str, doc) -> bool:
    """Détecte automatiquement les mots-clés de résolution."""
    try:
        solution_lower = solution_text.lower()
        
        # Recherche directe plus rapide que regex
        resolution_words = ['résolu', 'corrigé', 'réparé', 'fixé', 'terminé', 'clôturé']
        for word in resolution_words:
            if word in solution_lower:
                return True
        
        # Vérification avec spaCy si disponible
        if doc:
            resolution_verbs = ['résoudre', 'corriger', 'réparer', 'fixer']
            for token in doc:
                if token.lemma_ in resolution_verbs and token.pos_ == 'VERB':
                    return True
        
        return False
        
    except Exception:
        return False

def detect_completion_indicators_automatically(solution_text: str, doc) -> bool:
    """Détecte automatiquement les indicateurs de complétion."""
    try:
        solution_lower = solution_text.lower()
        
        completion_words = ['validé', 'vérifié', 'testé', 'confirmé', 'fonctionne']
        for word in completion_words:
            if word in solution_lower:
                return True
        
        return False
        
    except Exception:
        return False

def calculate_concordance_score(problem, solution):
    """Calcule le score de concordance OPTIMISÉ."""
    if pd.isna(problem) or pd.isna(solution):
        return 50.0
    
    problem_str = str(problem).strip()
    solution_str = str(solution).strip()
    
    if not problem_str or not solution_str:
        return 50.0
    
    # CACHE
    cache_key = f"{hash(problem_str[:100])}_{hash(solution_str[:100])}"
    if hasattr(calculate_concordance_score, '_cache'):
        cache = calculate_concordance_score._cache
        if cache_key in cache:
            return cache[cache_key]
    else:
        calculate_concordance_score._cache = {}
    
    try:
        # Initialiser NLP si nécessaire
        if nlp is None:
            initialize_nlp_models()
            
        solution_doc = nlp(solution_str) if nlp else None
        
        # 1. Similarité sémantique (20 points)
        similarite_score = 5  # Valeur de base
        
        if st_model and problem_str and solution_str:
            try:
                # Limiter la longueur pour performance
                prob_short = problem_str[:200]
                sol_short = solution_str[:200]
                
                embeddings = st_model.encode([prob_short, sol_short], 
                                           show_progress_bar=False,
                                           batch_size=2)
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                
                if similarity >= 0.65:
                    similarite_score = 20
                elif similarity >= 0.50:
                    similarite_score = 15
                elif similarity >= 0.30:
                    similarite_score = 10
            except Exception as e:
                print(f"Erreur similarite semantique: {e}")
                similarite_score = 5
        
        # 2. Mots-clés de résolution (40 points)
        resolution_detected = detect_resolution_keywords_automatically(solution_str, solution_doc)
        resolution_score = 40 if resolution_detected else 0
        
        # 3. Indicateurs de complétion (40 points)
        completion_detected = detect_completion_indicators_automatically(solution_str, solution_doc)
        completion_score = 40 if completion_detected else 0
        
        total = similarite_score + resolution_score + completion_score
        score = min(100, max(0, round(total, 2)))  # Entre 0 et 100
        
        # Debug: Afficher le détail pour quelques paires
        if calculate_concordance_score.debug_count < 3:
            print(f"  [DEBUG] Score concordance:")
            print(f"    Probleme: '{problem_str[:50]}...'")
            print(f"    Solution: '{solution_str[:50]}...'")
            print(f"    Similarite: {similarite_score}/20")
            print(f"    Resolution: {resolution_score}/40 (detecte: {resolution_detected})")
            print(f"    Completion: {completion_score}/40 (detecte: {completion_detected})")
            print(f"    TOTAL: {score}/100")
            calculate_concordance_score.debug_count += 1
        
        # Mettre en cache
        calculate_concordance_score._cache[cache_key] = score
        return score
        
    except Exception as e:
        print(f"Erreur calcul score concordance: {e}")
        score = 50.0
        calculate_concordance_score._cache[cache_key] = score
        return score

# Initialiser le compteur de debug
calculate_concordance_score.debug_count = 0
calculate_concordance_score._cache = {}

def calculate_note_concordance(score_concordance):
    """Convertit le score de concordance (%) en note sur 10."""
    return round((score_concordance / 100) * 10, 2)

# --- Nouvelle analyse temporelle ---

def calculate_temporal_note(temps_heures):
    """Calcule la note temporelle sur 10 selon le SLA."""
    if pd.isna(temps_heures):
        return 0.0
    
    if temps_heures <= 4.0:
        return 10.0
    elif temps_heures <= 8.0:
        return 5.0
    elif temps_heures <= 24.0:
        return 3.0
    else:
        return 2.0

# --- Calcul de la note finale sur 10 ---

def calculate_final_note(row):
    """Calcule la note finale sur 10."""
    note_temporelle = row.get('NoteTemporelle', 0)
    note_semantique = row.get('NoteSemantique', 0)
    note_concordance = row.get('NoteConcordance', 0)
    
    note_finale = (note_temporelle * 0.50) + (note_semantique * 0.40) + (note_concordance * 0.10)
    return round(note_finale, 2)

# --- Calcul de la moyenne employé ---

def calculate_employee_average(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule la moyenne des notes pour chaque employé."""
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

# --- Clustering pour problèmes récurrents OPTIMISÉ ---

def extract_keywords_automatically(descriptions: List[str]) -> str:
    """Extrait automatiquement les mots-clés les plus pertinents."""
    if not descriptions:
        return "Aucun mot-clé"
    
    try:
        all_text = ' '.join(descriptions[:20])  # Limiter pour performance
        doc = nlp(all_text.lower())
        
        relevant_words = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'VERB'] and 
                not token.is_stop and 
                len(token.text) > 3):
                relevant_words.append(token.lemma_)
        
        if relevant_words:
            word_counts = Counter(relevant_words)
            top_words = [word for word, _ in word_counts.most_common(3)]
            return ', '.join(top_words)
        
        return "Aucun mot-clé significatif"
        
    except Exception as e:
        print(f"Erreur extraction mots-cles: {e}")
        return "Extraction echouee"

def generate_group_name_from_keywords(keywords: str) -> str:
    """Génère un nom de groupe basé sur les mots-clés."""
    if not keywords or keywords == "Aucun mot-clé":
        return "Probleme Divers"
    
    first_keyword = keywords.split(',')[0].strip()
    return f"Probleme: {first_keyword.capitalize()}"

def perform_advanced_clustering(df: pd.DataFrame, categories_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Effectue le clustering avancé OPTIMISÉ."""
    
    print("Debut clustering optimise...")
    start_time = time.time()
    
    cluster_results = []
    df_with_clusters = df.copy()
    df_with_clusters['ClusterID'] = -1
    df_with_clusters['CategoryID'] = 0
    
    total_tickets = len(df)
    
    try:
        # === ÉTAPE 1: Correspondance rapide avec catégories ===
        categories_used = 0
        if not categories_data.empty and len(categories_data) > 0:
            print(f"Recherche dans {min(len(categories_data), 20)} categories...")
            
            # Prendre seulement les premières catégories pour performance
            for idx, category_row in categories_data.head(20).iterrows():
                if categories_used >= 15:  # Limite raisonnable
                    break
                    
                category_id = category_row['CategoryID']
                category_name = str(category_row['CategoryName']).lower()
                
                if not category_name or len(category_name) < 3:
                    continue
                
                # Recherche rapide avec str.contains
                mask = (df['ProblemDescription'].astype(str).str.lower().str.contains(category_name, na=False) |
                       df['SolutionContent'].astype(str).str.lower().str.contains(category_name, na=False))
                
                matching_indices = df[mask].index.tolist()
                
                if len(matching_indices) >= MIN_CLUSTER_SIZE:
                    cluster_id = len(cluster_results)
                    df_with_clusters.loc[matching_indices, 'ClusterID'] = cluster_id
                    df_with_clusters.loc[matching_indices, 'CategoryID'] = category_id
                    
                    cluster_results.append({
                        'ProblemNameGroup': category_row['CategoryName'],
                        'ClusterID': cluster_id,
                        'KeywordMatch': category_name,
                        'RecurrenceCount': len(matching_indices),
                        'CategoryID': category_id
                    })
                    
                    categories_used += 1
                    print(f"  {category_row['CategoryName']}: {len(matching_indices)} tickets")
        
        print(f"{categories_used} clusters categories")
        
        # === ÉTAPE 2: ÉCHANTILLONNAGE INTELLIGENT ===
        remaining_indices = df_with_clusters[df_with_clusters['ClusterID'] == -1].index.tolist()
        remaining_count = len(remaining_indices)
        
        print(f"Tickets restants: {remaining_count}")
        
        if remaining_count > MAX_TICKETS_FOR_CLUSTERING:
            print(f"Echantillonnage a {MAX_TICKETS_FOR_CLUSTERING} tickets pour performance")
            # Échantillonnage aléatoire stratifié
            remaining_indices = np.random.choice(
                remaining_indices, 
                size=MAX_TICKETS_FOR_CLUSTERING, 
                replace=False
            ).tolist()
            remaining_count = len(remaining_indices)
        
        # === ÉTAPE 3: Clustering hiérarchique sur échantillon ===
        if remaining_count >= 50 and st_model:  # Minimum 50 tickets
            print(f"Clustering sur {remaining_count} tickets...")
            
            # Préparer descriptions
            descriptions = []
            valid_indices = []
            
            for idx in remaining_indices:
                desc = str(df.loc[idx, 'ProblemDescription'])
                if desc and len(desc.strip()) > 20:
                    descriptions.append(desc[:300])  # Limiter longueur
                    valid_indices.append(idx)
            
            if len(descriptions) >= 20:
                print(f"  Encodage {len(descriptions)} descriptions...")
                
                try:
                    # Encodage optimisé
                    embeddings = st_model.encode(
                        descriptions,
                        show_progress_bar=False,
                        batch_size=32,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    
                    # Calcul nombre de clusters
                    slots_available = MAX_TOTAL_CLUSTERS - len(cluster_results)
                    clusters_needed = min(
                        slots_available,
                        len(descriptions) // 40  # ~40 tickets/cluster
                    )
                    
                    if clusters_needed >= 2:
                        print(f"  Creation {clusters_needed} clusters...")
                        
                        try:
                            from sklearn.cluster import MiniBatchKMeans
                            kmeans = MiniBatchKMeans(
                                n_clusters=clusters_needed,
                                random_state=42,
                                batch_size=256,  # Batch plus grand
                                max_iter=50,     # Moins d'itérations
                                n_init=2         # Moins d'initialisations
                            )
                            labels = kmeans.fit_predict(embeddings)
                            print("  Clustering termine")
                        except:
                            # Fallback à clustering hiérarchique
                            from sklearn.cluster import AgglomerativeClustering
                            clustering = AgglomerativeClustering(
                                n_clusters=clusters_needed,
                                linkage='ward',
                                metric='euclidean'
                            )
                            labels = clustering.fit_predict(embeddings)
                        
                        # Créer clusters
                        for cluster_num in range(clusters_needed):
                            cluster_mask = labels == cluster_num
                            cluster_indices = [valid_indices[i] for i, mask in enumerate(cluster_mask) if mask]
                            
                            if cluster_indices and len(cluster_indices) >= MIN_CLUSTER_SIZE:
                                cluster_id = len(cluster_results)
                                
                                if cluster_id >= MAX_TOTAL_CLUSTERS:
                                    break
                                
                                df_with_clusters.loc[cluster_indices, 'ClusterID'] = cluster_id
                                
                                # Nom du cluster
                                cluster_descriptions = [descriptions[i] for i, mask in enumerate(cluster_mask) if mask]
                                keywords = extract_keywords_automatically(cluster_descriptions)
                                group_name = generate_group_name_from_keywords(keywords)
                                
                                cluster_results.append({
                                    'ProblemNameGroup': group_name,
                                    'ClusterID': cluster_id,
                                    'KeywordMatch': keywords,
                                    'RecurrenceCount': len(cluster_indices),
                                    'CategoryID': 0
                                })
                except Exception as e:
                    print(f"  Erreur clustering: {e}")
        
        # === ÉTAPE 4: Gestion tickets non clusterisés ===
        non_clustered = df_with_clusters[df_with_clusters['ClusterID'] == -1]
        if not non_clustered.empty and cluster_results:
            # Ajouter au plus grand cluster
            if cluster_results:
                largest_cluster = max(cluster_results, key=lambda x: x['RecurrenceCount'])
                df_with_clusters.loc[non_clustered.index, 'ClusterID'] = largest_cluster['ClusterID']
                largest_cluster['RecurrenceCount'] += len(non_clustered)
        
        # === ÉTAPE 5: Finalisation ===
        # Convertir en DataFrame
        cluster_df = pd.DataFrame(cluster_results) if cluster_results else pd.DataFrame()
        
        if not cluster_df.empty:
            # Trier et réindexer
            cluster_df = cluster_df.sort_values('RecurrenceCount', ascending=False)
            cluster_df = cluster_df.reset_index(drop=True)
            cluster_df['ClusterID'] = range(len(cluster_df))
            
            # Mettre à jour les IDs
            id_mapping = {old_id: new_id for new_id, old_id in enumerate(cluster_df['ClusterID'])}
            df_with_clusters['ClusterID'] = df_with_clusters['ClusterID'].map(id_mapping).fillna(0).astype(int)
        
        # Statistiques
        clustered_time = time.time() - start_time
        final_count = len(cluster_df)
        clustered_tickets = len(df_with_clusters[df_with_clusters['ClusterID'] != -1])
        
        print(f"\nRESULTATS CLUSTERING:")
        print(f"   Clusters crees: {final_count}")
        print(f"   Tickets clusterises: {clustered_tickets}/{total_tickets}")
        print(f"   Temps: {clustered_time:.1f}s")
        
        if final_count > 0:
            avg_size = cluster_df['RecurrenceCount'].mean()
            print(f"   Taille moyenne: {avg_size:.1f} tickets/cluster")
        
        return cluster_df, df_with_clusters
        
    except Exception as e:
        print(f"Erreur clustering: {str(e)[:100]}")
        
        # Solution de repli
        df_with_clusters['ClusterID'] = 0
        df_with_clusters['CategoryID'] = 0
        
        return pd.DataFrame([{
            'ProblemNameGroup': 'Tous les tickets',
            'ClusterID': 0,
            'KeywordMatch': 'Clustering echoue',
            'RecurrenceCount': len(df),
            'CategoryID': 0
        }]), df_with_clusters

# --- Fonction principale CORRIGÉE ---

def run_full_analysis(df):
    """Exécute le pipeline d'analyse COMPLET corrigé."""
    if df.empty:
        print("DataFrame vide")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"DEMARRAGE ANALYSE SUR {len(df)} TICKETS")
    total_start = time.time()
    step_start = time.time()
    
    # Initialiser les modèles NLP si nécessaire
    if nlp is None or st_model is None:
        initialize_nlp_models()
    
    # Nettoyer les caches au début
    if hasattr(calculate_semantique_score, '_cache'):
        calculate_semantique_score._cache.clear()
    if hasattr(calculate_concordance_score, '_cache'):
        calculate_concordance_score._cache.clear()
    
    # Réinitialiser les compteurs de debug
    calculate_semantique_score.debug_count = 0
    calculate_concordance_score.debug_count = 0
    
    df['ClusterID'] = 0
    df['CategoryID'] = 0
    
    # === ÉTAPE 1: Scores sémantiques CORRIGÉE ===
    print(f"\n[1/6] Calcul scores sémantiques...")
    step_start = time.time()
    
    batch_size = 500
    semantic_scores = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_scores = batch['SolutionContent'].apply(calculate_semantique_score)
        semantic_scores.extend(batch_scores)
        
        if i > 0 and i % 1000 == 0:
            elapsed = time.time() - step_start
            print(f"  Progression: {i}/{len(df)} tickets ({elapsed:.1f}s)")
    
    df['ScoreSemantique'] = semantic_scores
    df['NoteSemantique'] = df['ScoreSemantique'].apply(calculate_note_semantique)
    
    step_time = time.time() - step_start
    print(f"  Termine en {step_time:.1f}s")
    
    # Afficher quelques statistiques sur les scores sémantiques
    if not df['ScoreSemantique'].empty:
        scores = df['ScoreSemantique']
        print(f"  Scores semantiques: Moyenne={scores.mean():.1f}%, "
              f"Min={scores.min():.1f}%, Max={scores.max():.1f}%, "
              f"Ecart-type={scores.std():.1f}%")
        
        # Distribution des scores
        bins = [0, 30, 50, 70, 90, 100]
        labels = ['0-30%', '31-50%', '51-70%', '71-90%', '91-100%']
        distribution = pd.cut(scores, bins=bins, labels=labels, right=False)
        dist_counts = distribution.value_counts().sort_index()
        print(f"  Distribution scores:")
        for label, count in dist_counts.items():
            percentage = (count / len(scores)) * 100
            print(f"    {label}: {count} tickets ({percentage:.1f}%)")
    
    # === ÉTAPE 2: Scores concordance ===
    print(f"\n[2/6] Calcul scores concordance...")
    step_start = time.time()
    
    concordance_scores = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_scores = batch.apply(
            lambda row: calculate_concordance_score(row['ProblemDescription'], row['SolutionContent']),
            axis=1
        )
        concordance_scores.extend(batch_scores)
    
    df['ScoreConcordance'] = concordance_scores
    df['NoteConcordance'] = df['ScoreConcordance'].apply(calculate_note_concordance)
    
    step_time = time.time() - step_start
    print(f"  Termine en {step_time:.1f}s")
    
    # Afficher quelques statistiques sur les scores de concordance
    if not df['ScoreConcordance'].empty:
        scores = df['ScoreConcordance']
        print(f"  Scores concordance: Moyenne={scores.mean():.1f}%, "
              f"Min={scores.min():.1f}%, Max={scores.max():.1f}%, "
              f"Ecart-type={scores.std():.1f}%")
    
    # === ÉTAPE 3: Notes temporelles ===
    print(f"\n[3/6] Calcul notes temporelles...")
    df['NoteTemporelle'] = df['TempsHeures'].apply(calculate_temporal_note)
    
    # === ÉTAPE 4: Notes finales ===
    print(f"\n[4/6] Calcul notes finales...")
    df['TicketNote'] = df.apply(calculate_final_note, axis=1)
    
    # === ÉTAPE 5: Statuts ===
    print(f"\n[5/6] Determination statuts...")
    df['Statut'] = df.apply(determine_final_status, axis=1)
    
    # === ÉTAPE 6: Moyennes employés ===
    print(f"\n[6/6] Calcul moyennes employés...")
    df = calculate_employee_average(df)
    
    # === CLUSTERING ===
    print(f"\nLancement clustering...")
    cluster_start = time.time()
    
    cluster_results = pd.DataFrame()
    try:
        # CORRECTION IMPORTANTE: Adaptez selon votre structure
        try:
            from utils.db_connector import load_categories_data
        except ImportError:
            from db_connector import load_categories_data
            
        categories_data = load_categories_data()
        cluster_results, df_with_clusters = perform_advanced_clustering(df, categories_data)
        
        df['ClusterID'] = df_with_clusters['ClusterID']
        df['CategoryID'] = df_with_clusters['CategoryID']
        
        cluster_time = time.time() - cluster_start
        print(f"  Clustering termine en {cluster_time:.1f}s")
    except Exception as e:
        print(f"  Erreur clustering: {e}")
        cluster_results = pd.DataFrame()
    
    # === PRÉPARATION RÉSULTATS ===
    print(f"\nPreparation resultats...")
    
    df_anomalies = df[[
        'TicketID', 'FactKey', 'AssigneeEmployeeKey', 'AssigneeFullName',
        'TicketNote', 'EmployeeAvgScore', 
        'ScoreSemantique', 'NoteSemantique',
        'ScoreConcordance', 'NoteConcordance',
        'TempsHeures', 'NoteTemporelle',
        'Statut', 'ClusterID', 'CategoryID'
    ]].copy()
    
    # Nettoyage valeurs numériques
    for col in ['TicketNote', 'EmployeeAvgScore', 'NoteSemantique', 'NoteConcordance', 'NoteTemporelle']:
        if col in df_anomalies.columns:
            df_anomalies[col] = pd.to_numeric(df_anomalies[col], errors='coerce').fillna(0).round(2)
    
    # === STATISTIQUES FINALES ===
    total_time = time.time() - total_start
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print(f"\n{'='*50}")
    print(f"ANALYSE TERMINEE AVEC SUCCES!")
    print(f"{'='*50}")
    print(f"STATISTIQUES FINALES:")
    print(f"   Temps total: {minutes}m {seconds}s")
    print(f"   Tickets analyses: {len(df_anomalies)}")
    print(f"   Clusters crees: {len(cluster_results)}")
    print(f"   Performance: {len(df)/total_time:.1f} tickets/seconde")
    
    if not df_anomalies.empty:
        avg_note = df_anomalies['TicketNote'].mean()
        avg_sem = df_anomalies['ScoreSemantique'].mean()
        avg_conc = df_anomalies['ScoreConcordance'].mean()
        ok_count = len(df_anomalies[df_anomalies['Statut'] == 'OK'])
        
        print(f"\n   Scores moyens:")
        print(f"      Note ticket: {avg_note:.2f}/10")
        print(f"      Score semantique: {avg_sem:.1f}%")
        print(f"      Score concordance: {avg_conc:.1f}%")
        print(f"   Tickets OK: {ok_count} ({ok_count/len(df)*100:.1f}%)")
        
        # Distribution des statuts
        status_counts = df_anomalies['Statut'].value_counts()
        print(f"\n   Distribution des statuts:")
        for status, count in status_counts.items():
            percentage = (count / len(df_anomalies)) * 100
            print(f"      {status}: {count} tickets ({percentage:.1f}%)")
    
    print(f"{'='*50}")
    
    return df_anomalies, cluster_results