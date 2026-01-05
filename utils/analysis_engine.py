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

# --- Configuration et Initialisation ---

# Définition des constantes d'anomalie
SEMAN_THRESHOLD = 0.80  # 80% pour la sémantique
CONC_THRESHOLD = 0.20   # 20% pour la concordance
SLA_THRESHOLD = 4.0     # 4 heures pour le SLA

# Initialisation des ressources
nlp = None
st_model = None
tool = None

try:
    nlp = spacy.load("fr_core_news_sm")
    st_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    # Essayer de charger language_tool, mais continuer si échec
    try:
        tool = language_tool_python.LanguageTool('fr')
        print("Modèles NLP chargés avec succès (avec vérification grammaticale)")
    except Exception as java_error:
        print(f"Java non détecté, désactivation de la vérification grammaticale: {java_error}")
        tool = None
    
except Exception as e:
    print(f"Erreur de chargement des modèles: {e}")
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
            except:
                grammaire_score = 25  # Valeur moyenne si échec
        else:
            grammaire_score = 25  # Valeur fixe sans vérification
        
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

def calculate_optimal_clusters(total_tickets: int) -> int:
    """
    Calcule le nombre optimal de clusters de manière intelligente.
    Règle: 1 cluster pour 35 tickets en moyenne, avec ajustements selon le volume.
    """
    # Règle de base : 1 cluster pour 35 tickets
    base_clusters = total_tickets // 35
    
    # Ajustements intelligents selon le volume
    if total_tickets < 100:
        # Très petit volume : clusters plus petits
        optimal_clusters = max(10, min(30, base_clusters))
    elif 100 <= total_tickets < 500:
        # Petit volume : équilibre
        optimal_clusters = max(15, min(50, base_clusters))
    elif 500 <= total_tickets < 2000:
        # Volume moyen : ratio standard
        optimal_clusters = max(20, min(80, base_clusters))
    elif 2000 <= total_tickets < 10000:
        # Grand volume : légèrement plus gros clusters
        # Pour éviter trop de clusters avec beaucoup de tickets
        adjusted_ratio = 45  # Au lieu de 35
        base_clusters = total_tickets // adjusted_ratio
        optimal_clusters = max(30, min(120, base_clusters))
    else:
        # Très grand volume : clusters plus gros
        adjusted_ratio = 60  # Clusters plus gros pour performance
        base_clusters = total_tickets // adjusted_ratio
        optimal_clusters = max(40, min(150, base_clusters))
    
    return optimal_clusters

def perform_advanced_clustering(df: pd.DataFrame, categories_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Effectue le clustering avancé pour les problèmes récurrents avec la nouvelle règle optimale.
    """
    
    cluster_results = []
    df_with_clusters = df.copy()
    df_with_clusters['ClusterID'] = -1
    df_with_clusters['CategoryID'] = 0  # Initialiser CategoryID
    
    try:
        if not categories_data.empty:
            print(f"Recherche de correspondance avec {len(categories_data)} catégories DimCategory...")
            
            for idx, category_row in categories_data.iterrows():
                category_id = category_row['CategoryID']
                category_name = category_row['CategoryName']
                category_desc = str(category_row.get('Description', ''))
                
                if not category_desc:
                    continue
                
                matching_indices = []
                for ticket_idx, row in df.iterrows():
                    problem_desc = str(row.get('ProblemDescription', '')).lower()
                    solution_desc = str(row.get('SolutionContent', '')).lower()
                    
                    if (category_name.lower() in problem_desc or 
                        category_name.lower() in solution_desc or
                        category_desc.lower() in problem_desc or 
                        category_desc.lower() in solution_desc):
                        matching_indices.append(ticket_idx)
                
                if matching_indices:
                    cluster_id = len(cluster_results)
                    
                    df_with_clusters.loc[matching_indices, 'ClusterID'] = cluster_id
                    df_with_clusters.loc[matching_indices, 'CategoryID'] = category_id
                    
                    all_descriptions = []
                    for idx in matching_indices:
                        if pd.notna(df.loc[idx, 'ProblemDescription']):
                            all_descriptions.append(str(df.loc[idx, 'ProblemDescription']))
                        if pd.notna(df.loc[idx, 'SolutionContent']):
                            all_descriptions.append(str(df.loc[idx, 'SolutionContent']))
                    
                    specific_keywords = extract_keywords_automatically(all_descriptions)
                    
                    cluster_results.append({
                        'ProblemNameGroup': category_name,
                        'ClusterID': cluster_id,
                        'KeywordMatch': specific_keywords if specific_keywords else category_name,
                        'RecurrenceCount': len(matching_indices),
                        'CategoryID': category_id
                    })
        
        remaining_indices = df_with_clusters[df_with_clusters['ClusterID'] == -1].index.tolist()
        
        if remaining_indices and len(remaining_indices) > 1 and st_model:
            print(f"Clustering hiérarchique pour {len(remaining_indices)} tickets restants...")
            
            # OPTIMISATION: Limiter à 2500 tickets max pour le clustering
            MAX_TICKETS_FOR_CLUSTERING = 2500
            if len(remaining_indices) > MAX_TICKETS_FOR_CLUSTERING:
                print(f"  Échantillonnage à {MAX_TICKETS_FOR_CLUSTERING} tickets")
                remaining_indices = remaining_indices[:MAX_TICKETS_FOR_CLUSTERING]
            
            descriptions_to_cluster = []
            valid_indices = []
            
            for idx in remaining_indices:
                problem_desc = str(df.loc[idx, 'ProblemDescription'])
                if problem_desc and len(problem_desc.strip()) > 10:
                    descriptions_to_cluster.append(problem_desc)
                    valid_indices.append(idx)
            
            if len(descriptions_to_cluster) >= 2:
                print(f"  Encodage de {len(descriptions_to_cluster)} descriptions...")
                embeddings = st_model.encode(descriptions_to_cluster, show_progress_bar=False)
                
                # ⭐⭐ NOUVELLE RÈGLE OPTIMISÉE ⭐⭐
                total_tickets_for_clustering = len(descriptions_to_cluster)
                optimal_n_clusters = calculate_optimal_clusters(total_tickets_for_clustering)
                
                print(f"  Création de {optimal_n_clusters} clusters pour {total_tickets_for_clustering} tickets")
                print(f"  (Ratio: {total_tickets_for_clustering/optimal_n_clusters:.1f} tickets/cluster)")
                
                # Utiliser MiniBatchKMeans pour la performance (plus rapide que AgglomerativeClustering)
                try:
                    from sklearn.cluster import MiniBatchKMeans
                    clustering = MiniBatchKMeans(
                        n_clusters=optimal_n_clusters,
                        random_state=42,
                        batch_size=1000,
                        n_init=3,
                        max_iter=100
                    )
                    cluster_labels = clustering.fit_predict(embeddings)
                    print(f"  MiniBatchKMeans terminé avec {optimal_n_clusters} clusters")
                except:
                    # Fallback à AgglomerativeClustering si MiniBatchKMeans échoue
                    print("  MiniBatchKMeans échoué, utilisation d'AgglomerativeClustering")
                    clustering = AgglomerativeClustering(
                        n_clusters=optimal_n_clusters,
                        metric='cosine',
                        linkage='average'
                    )
                    cluster_labels = clustering.fit_predict(embeddings)
                
                for idx, cluster_label in zip(valid_indices, cluster_labels):
                    cluster_id = len(cluster_results) + cluster_label
                    df_with_clusters.loc[idx, 'ClusterID'] = cluster_id
                    
                    if cluster_id >= len(cluster_results):
                        cluster_descriptions = []
                        cluster_indices = []
                        
                        for j, (desc_idx, desc_label) in enumerate(zip(valid_indices, cluster_labels)):
                            if desc_label == cluster_label:
                                cluster_descriptions.append(descriptions_to_cluster[j])
                                cluster_indices.append(desc_idx)
                        
                        if cluster_descriptions:
                            keywords = extract_keywords_automatically(cluster_descriptions)
                            group_name = generate_group_name_from_keywords(keywords)
                            
                            # Essayer d'associer à une catégorie existante
                            cluster_category_id = 0
                            if categories_data is not None and not categories_data.empty:
                                cluster_text = ' '.join(cluster_descriptions).lower()
                                for _, cat_row in categories_data.iterrows():
                                    cat_name = str(cat_row['CategoryName']).lower()
                                    if cat_name in cluster_text:
                                        cluster_category_id = cat_row['CategoryID']
                                        group_name = f"{cat_row['CategoryName']} - {group_name}"
                                        break
                            
                            cluster_results.append({
                                'ProblemNameGroup': group_name,
                                'ClusterID': cluster_id,
                                'KeywordMatch': keywords,
                                'RecurrenceCount': len(cluster_indices),
                                'CategoryID': cluster_category_id
                            })
                        
                        # Mettre à jour le CategoryID dans df_with_clusters
                        df_with_clusters.loc[cluster_indices, 'CategoryID'] = cluster_category_id
                    else:
                        cluster_results[cluster_id]['RecurrenceCount'] += 1
        
        # Gérer les tickets non clusterisés
        non_clustered = df_with_clusters[df_with_clusters['ClusterID'] == -1]
        if not non_clustered.empty:
            cluster_id = len(cluster_results)
            df_with_clusters.loc[non_clustered.index, 'ClusterID'] = cluster_id
            
            cluster_results.append({
                'ProblemNameGroup': 'Non Classifié',
                'ClusterID': cluster_id,
                'KeywordMatch': 'Description insuffisante',
                'RecurrenceCount': len(non_clustered),
                'CategoryID': -1
            })
        
        print(f"✅ Clustering terminé: {len(cluster_results)} clusters créés")
        print(f"   - Tickets clusterisés: {len(df_with_clusters) - len(non_clustered)}")
        print(f"   - Tickets non classifiés: {len(non_clustered)}")
        
        return pd.DataFrame(cluster_results), df_with_clusters
        
    except Exception as e:
        print(f"Erreur clustering avancé: {e}")
        import traceback
        traceback.print_exc()
        df_with_clusters['ClusterID'] = 0
        df_with_clusters['CategoryID'] = 0
        return pd.DataFrame([{
            'ProblemNameGroup': 'Échec de Classification',
            'ClusterID': 0,
            'KeywordMatch': 'Erreur technique',
            'RecurrenceCount': len(df),
            'CategoryID': -1
        }]), df_with_clusters

# --- Fonction principale du pipeline ---

def run_full_analysis(df):
    """Exécute l'intégralité du pipeline d'analyse IA avec les nouvelles règles."""
    if df.empty:
        print("DataFrame vide reçu")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Début de l'analyse sur {len(df)} tickets assignés")
    
    df['ClusterID'] = 0
    df['CategoryID'] = 0  # Initialiser CategoryID
    
    print("Calcul des scores sémantiques...")
    df['ScoreSemantique'] = df['SolutionContent'].apply(calculate_semantique_score)
    df['NoteSemantique'] = df['ScoreSemantique'].apply(calculate_note_semantique)
    
    print("Calcul des scores de concordance...")
    df['ScoreConcordance'] = df.apply(
        lambda row: calculate_concordance_score(row['ProblemDescription'], row['SolutionContent']),
        axis=1
    )
    df['NoteConcordance'] = df['ScoreConcordance'].apply(calculate_note_concordance)
    
    print("Calcul des notes temporelles...")
    df['NoteTemporelle'] = df['TempsHeures'].apply(calculate_temporal_note)
    
    print("Calcul des notes finales...")
    df['TicketNote'] = df.apply(calculate_final_note, axis=1)
    
    print("Détermination des statuts...")
    df['Statut'] = df.apply(determine_final_status, axis=1)
    
    print("Calcul des moyennes employé...")
    df = calculate_employee_average(df)
    
    print("Clustering avancé en cours...")
    cluster_results = pd.DataFrame()
    try:
        from utils.db_connector import load_categories_data
        categories_data = load_categories_data()
        cluster_results, df_with_clusters = perform_advanced_clustering(df, categories_data)
        
        # Mettre à jour ClusterID et CategoryID depuis le clustering
        df['ClusterID'] = df_with_clusters['ClusterID']
        df['CategoryID'] = df_with_clusters['CategoryID']
        
        print(f"Clustering terminé: {len(cluster_results)} clusters créés")
    except Exception as e:
        print(f"Erreur clustering: {e}")
        cluster_results = pd.DataFrame()
    
    print("Préparation des résultats...")
    df_anomalies = df[[
        'TicketID', 'FactKey', 'AssigneeEmployeeKey', 'AssigneeFullName',
        'TicketNote', 'EmployeeAvgScore', 
        'ScoreSemantique', 'NoteSemantique',
        'ScoreConcordance', 'NoteConcordance',
        'TempsHeures', 'NoteTemporelle',
        'Statut', 'ClusterID', 'CategoryID'  # Ajout de CategoryID
    ]].copy()
    
    for col in ['TicketNote', 'EmployeeAvgScore', 'NoteSemantique', 'NoteConcordance', 'NoteTemporelle']:
        if col in df_anomalies.columns:
            df_anomalies[col] = pd.to_numeric(df_anomalies[col], errors='coerce').fillna(0)
    
    print("Analyse terminée avec succès!")
    return df_anomalies, cluster_results