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

# Configuration de base de l'application
st.set_page_config(
    page_title="YeshControl IA",
    page_icon="styles/icones/ai_icon.png", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Chargement du style CSS
try:
    with open("styles/main.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Fichier CSS non trouvé")

# Gestion de l'état de la session
def initialize_session_state():
    """Initialise toutes les variables de session nécessaires."""
    session_vars = {
        'analysis_started': False,
        'anomaly_data': None,
        'cluster_data': None,  # Pour stocker les résultats du clustering
        'current_page': 1,
        'pagination_offset': 0
    }
    
    for key, default_value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialisation des variables de session
initialize_session_state()

# Navigation principale
def main():
    """Gère la navigation principale de l'application."""
    current_page = st.session_state['current_page']
    
    if current_page == 1:
        # Import et affichage de la page analyse
        from pages.analyse import render as render_analyse
        render_analyse()
        
    elif current_page == 2:
        # Import et affichage de la page résultats
        from pages.resultats import render as render_resultats
        render_resultats()
    
    else:
        st.error(f"Page inconnue: {current_page}")
        # Retour à la page par défaut en cas d'erreur
        st.session_state['current_page'] = 1
        st.rerun()

if __name__ == "__main__":
    main()