import streamlit as st
from utils.db_connector import load_data_for_analysis, save_analysis_results
from utils.analysis_engine import run_full_analysis
import base64

def get_base64_encoded_image(image_path):
    """Encode une image en base64 pour l'affichage HTML."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""

def start_analysis():
    """Fonction pour lancer l'analyse complète avec sauvegarde en base."""
    st.session_state['analysis_started'] = True
    
    with st.spinner('Analyse en cours...'):
        # 1. Chargement des données
        data_to_analyze = load_data_for_analysis()
        
        if data_to_analyze is None or data_to_analyze.empty:
            st.error("❌ Erreur de chargement des données. Vérifiez la connexion à la base de donnée.")
            st.session_state['analysis_started'] = False
            return
            
        # 2. Exécution du moteur d'analyse
        df_anomalies, cluster_results = run_full_analysis(data_to_analyze)
        
        # 3. Sauvegarde dans la base de données
        save_success = save_analysis_results(df_anomalies, cluster_results)
        
        if not save_success:
            st.warning("⚠️ L'analyse est terminée mais la sauvegarde a échoué.")
        
        # 4. Stockage du résultat dans l'état de la session
        st.session_state['anomaly_data'] = df_anomalies
        st.session_state['current_page'] = 2
        st.session_state['pagination_offset'] = 0

    st.rerun()

def render():
    """Affiche le contenu de la Page 1."""
    
    # Vérification et initialisation des variables de session
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 1
    if 'analysis_started' not in st.session_state:
        st.session_state['analysis_started'] = False
    if 'pagination_offset' not in st.session_state:
        st.session_state['pagination_offset'] = 0
    
    # Si on est déjà sur la page 2, on laisse app.py gérer l'affichage
    if st.session_state.get('current_page') == 2:
        st.rerun()  # <-- PROBLÈME POTENTIEL : redéclenche tout le script app.py
        return

    # Charger les images en base64
    ai_icon = get_base64_encoded_image("styles/icones/ai_icon.png")
    analyze_icon = get_base64_encoded_image("styles/icones/analyze.png")

    # Appliquer le style CSS et injecter l'ID de cadre
    st.markdown(f"""
    <style>
    /* ----------------------------------------------------------------------------------- */
    /* NOUVELLE RÈGLE pour cibler le conteneur Streamlit et appliquer le style analyze-card */
    /* Le style exact est dans main.css, ici on prépare le terrain pour l'ID analyze-frame */
    #analyze-frame {{
        /* Ceci garantit que l'ID est reconnu dans cette page */
        width: 100%;
    }}
    /* ----------------------------------------------------------------------------------- */

    /* Style pour le bouton Streamlit */
    .stButton > button {{
        background: linear-gradient(45deg, #4CAF50, #45a049) !important;
        color: white !important;
        padding: 20px 40px !important;
        font-size: 1.4rem !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        border: none !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 12px rgba(76, 175, 80, 0.3) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 15px !important;
        margin: 0 auto !important;
        min-height: 70px !important;
        width: 100% !important;
        max-width: 400px !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 16px rgba(76, 175, 80, 0.4) !important;
        background: linear-gradient(45deg, #45a049, #4CAF50) !important;
        color: white !important;
    }}
    
    /* Ajouter l'icône via pseudo-élément */
    .stButton > button::before {{
        content: "";
        display: inline-block;
        width: 32px;
        height: 32px;
        background-image: url('data:image/png;base64,{analyze_icon}');
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
    }}
    
    .analyze-container {{
        text-align: center;
        padding: 30px 20px;
    }}
    
    .system-title {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        margin-bottom: 40px;
    }}
    
    .system-title img {{
        width: 60px;
        height: 60px;
    }}
    
    /* MODIFICATION ICI : Cible le nouveau span pour conserver le style du H1 sans ancre */
    .system-title-text {{
        color: #2e2a80;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        font-family: "Segoe UI", Arial, sans-serif;
    }}
    
    .analyze-card {{
        /* Ces styles seront déplacés vers #analyze-frame > div dans main.css, mais on les garde ici au cas où ils sont utilisés ailleurs */
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        border-radius: 20px;
        padding: 50px 40px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 2px solid #e9ecef;
        margin: 20px auto;
        max-width: 600px;
    }}
    
    .analyze-instruction {{
        color: #555;
        margin-bottom: 40px;
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 1.3rem;
        line-height: 1.6;
        font-weight: 500;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Conteneur principal centré
    with st.container():
        # Espacement en haut
        st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
        
        # En-tête avec le nom du système et l'icône AI
        # MODIFICATION ICI : Remplacement de <h1> par <span class='system-title-text'>
        st.markdown(f"""
        <div class='system-title'>
            <img src="data:image/png;base64,{ai_icon}">
            <span class='system-title-text'>YeschControl AI</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Colonnes pour centrer le contenu
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # DÉBUT DE LA MODIFICATION POUR ENCADRER LE TEXTE ET LE BOUTON
            
            #Utiliser st.container pour regrouper les éléments
            with st.container():
                #Injecter l'ID pour que le CSS puisse cibler ce conteneur
                st.markdown('<div id="analyze-frame"></div>', unsafe_allow_html=True)

                # Texte d'instruction (maintenant à l'intérieur du cadre)
                st.markdown("""
                    <div class='analyze-instruction'>
                        Cliquez sur le bouton ci-dessous pour lancer l'analyse
                    </div>
                """, unsafe_allow_html=True)
                
                # Bouton Streamlit unique et stylisé (juste après le texte)
                if st.button("Analyser maintenant", key="analyze_trigger", use_container_width=True):
                    start_analysis()

            # FIN DE LA MODIFICATION POUR ENCADRER LE TEXTE ET LE BOUTON
            
            # Informations sur l'état de l'analyse (EN DEHORS du cadre)
            if st.session_state.get('analysis_started'):
                st.info(" Analyse en cours...")
                
if __name__ == "__main__":
    render()

