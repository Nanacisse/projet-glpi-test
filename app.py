import streamlit as st
# Masquer les éléments de développement
hide_elements_style = """
<style>
/* Masque le pied de page 'Made with Streamlit' */
footer {
    visibility: hidden;
}
/* Masque l'indicateur de statut/rechargement (le 'bonhomme qui clignote') */
div[data-testid="stStatusWidget"] {
    display: none !important; 
}
</style>
"""
st.markdown(hide_elements_style, unsafe_allow_html=True)

#Configuration de base de l'application
st.set_page_config(
    page_title="YeshControl IA",
    page_icon="styles/icones/ai_icon.png", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

#Chargement du style CSS
try:
    with open("styles/main.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Fichier CSS non trouvé")

#Gestion de l'état de la session
if 'analysis_started' not in st.session_state:
    st.session_state['analysis_started'] = False
if 'anomaly_data' not in st.session_state:
    st.session_state['anomaly_data'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 1
if 'pagination_offset' not in st.session_state:
    st.session_state['pagination_offset'] = 0

#Navigation principale
if st.session_state['current_page'] == 1:
    #Import et affichage de la page analyse
    from pages.analyse import render as render_analyse
    render_analyse()
    
elif st.session_state['current_page'] == 2:
    #Import et affichage de la page résultats
    from pages.resultats import render as render_resultats
    render_resultats()

