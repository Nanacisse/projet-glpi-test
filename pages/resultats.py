import streamlit as st
import pandas as pd
import io
import base64
from datetime import datetime
from utils.analysis_engine import SEMAN_THRESHOLD, CONC_THRESHOLD

#Import des bibliothèques d'export
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

def export_to_pdf(df):
    """Génère un PDF à partir du DataFrame."""
    if not HAS_FPDF:
        st.error("fpdf2 n'est pas installé. Utilisez: pip install fpdf2")
        return None
        
    try:
        pdf = FPDF(orientation='L')
        pdf.add_page()
        pdf.set_font("Arial", size=6)
        
        #En-tête du tableau
        headers = df.columns.tolist()
        col_width = 280 / len(headers)
        
        for header in headers:
            pdf.cell(col_width, 10, str(header), 1, 0, 'C')
        pdf.ln()

        #Corps du tableau
        for index, row in df.iterrows():
            for item in row:
                # Tronquer les textes trop longs
                text = str(item)[:30] + "..." if len(str(item)) > 30 else str(item)
                pdf.cell(col_width, 10, text, 1, 0, 'C')
            pdf.ln()
            
        return pdf.output(dest='S').encode('latin-1')
    except Exception as e:
        st.error(f"Erreur génération PDF: {e}")
        return None

def get_base64_encoded_image(image_path):
    """Encode une image en base64 pour l'affichage HTML."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""

def get_status_with_icon(status):
    """Retourne le statut avec l'icône correspondante."""
    icon_mapping = {
        'OK': 'status_ok.png',
        'Anomalie de Temps': 'danger.png',
        'Anomalie Sémantique': 'danger.png',
        'Anomalie de Concordance': 'danger.png',
        'Multiples Anomalies': 'red.png',
        'Anomalie Indéterminée': 'danger.png'
    }
    
    icon_file = icon_mapping.get(status, 'danger.png')
    try:
        icon_base64 = get_base64_encoded_image(f"styles/icones/{icon_file}")
        icon_html = f'<img src="data:image/png;base64,{icon_base64}" width="16" height="16" style="vertical-align: middle; margin-left: 5px;">'
        return f"{status} {icon_html}"
    except:
        return status

def render():
    """Affiche le contenu de la Page 2."""
    
    #Bouton de retour en haut à gauche
    col_retour, col_titre = st.columns([0.1, 0.9])
    with col_retour:
        if st.button("←", key="retour_analyse", help="Retour à l'analyse"):
            st.session_state['current_page'] = 1
            st.session_state['analysis_started'] = False
            st.rerun()
    
    df_data = st.session_state.get('anomaly_data')
    if df_data is None or (isinstance(df_data, pd.DataFrame) and df_data.empty):
        st.warning("⚠️ Aucune donnée d'anomalie disponible. Veuillez lancer l'analyse d'abord.")
        return
    
    #Préparer les données pour l'affichage
    df_display = df_data.copy()
    
    #Formater les colonnes pour l'affichage exact
    display_columns = {}
    
    #ID Ticket
    if 'TicketID' in df_display.columns:
        display_columns['ID Ticket'] = df_display['TicketID']
    
    #Nom Employé
    if 'AssigneeFullName' in df_display.columns:
        display_columns['Nom Employé'] = df_display['AssigneeFullName']
    elif 'RealName' in df_display.columns:
        display_columns['Nom Employé'] = df_display['RealName']
    else:
        display_columns['Nom Employé'] = "Non spécifié"
    
    #Date Création Ticket
    if 'DateCreation' in df_display.columns:
        display_columns['Date Création Ticket'] = pd.to_datetime(df_display['DateCreation']).dt.strftime('%d/%m/%Y')
    
    #Temps de résolution (h)
    if 'TempsHeures' in df_display.columns:
        display_columns['Temps de résolution (h)'] = df_display['TempsHeures'].apply(lambda x: f"{x:.2f}")
    
    #Temps Moyen (h)
    if 'TempsMoyenHeures' in df_display.columns:
        display_columns['Temps Moyen (h)'] = df_display['TempsMoyenHeures'].apply(lambda x: f"{x:.2f}")
    
    #Écart Type (h)
    if 'EcartTypeHeures' in df_display.columns:
        display_columns['Écart Type (h)'] = df_display['EcartTypeHeures'].apply(lambda x: f"{x:.2f}")
    
    #Score Temporel (Z-score)
    if 'ScoreTemporel' in df_display.columns:
        display_columns['Score Temporel'] = df_display['ScoreTemporel'].apply(lambda x: f"{x:.2f}")
    
    #Anomalie Temporelle
    if 'AnomalieTemporelle' in df_display.columns:
        display_columns['Anomalie Temporelle'] = df_display['AnomalieTemporelle']
    
    #Score Sémantique (%)
    if 'ScoreSemantique' in df_display.columns:
        display_columns['Score Sémantique (%)'] = df_display['ScoreSemantique'].apply(lambda x: f"{x:.2f}%")
    
    #Score Concordance (%)
    if 'ScoreConcordance' in df_display.columns:
        display_columns['Score Concordance (%)'] = df_display['ScoreConcordance'].apply(lambda x: f"{x:.2f}%")
    
    #Note Ticket (Base 10)
    if 'TicketNote' in df_display.columns:
        display_columns['Note Ticket (Base 10)'] = df_display['TicketNote'].apply(lambda x: f"{x:.2f}")
    
    #Moyenne Employé (/10)
    if 'EmployeeAvgScore' in df_display.columns:
        display_columns['Moyenne Employé (/10)'] = df_display['EmployeeAvgScore'].apply(lambda x: f"{x:.2f}")
    
    #Statut avec icônes
    if 'Statut' in df_display.columns:
        display_columns['Statut'] = df_display['Statut'].apply(get_status_with_icon)

    #Créer le DataFrame d'affichage final
    df_display_final = pd.DataFrame(display_columns)
        
    #Titre centré
    st.markdown("""
    <div style='text-align: center; font-family: "Segoe UI", Arial, sans-serif; font-weight: 700; color: #2e2a80; margin-bottom: 20px; font-size: 2rem;'>
    TABLEAU DES ANOMALIES
    </div>
    """, unsafe_allow_html=True)

    #Filtre et lee bouton export sur la même ligne
    col1, col2, col3, col4 = st.columns([0.23, 0.23, 0.23, 0.31])
    
    with col1:
        #Filtre par nom d'employé
        if 'Nom Employé' in df_display_final.columns:
            employes = ['Tous'] + sorted(df_display_final['Nom Employé'].unique().tolist())
            employe_filtre = st.selectbox("Nom employé", employes)
        else:
            employe_filtre = 'Tous'
    
    with col2:
        #Filtre par type d'anomalie
        types_anomalie = ['Tous'] + sorted(df_data['Statut'].unique().tolist())
        type_filtre = st.selectbox("Type d'anomalie", types_anomalie)
    
    with col3:
        #Filtre par date
        if 'DateCreation' in df_data.columns:
            dates = pd.to_datetime(df_data['DateCreation']).dt.date
            min_date = dates.min()
            max_date = dates.max()
            date_filtre = st.date_input(
                "Date création",
                value=None,
                min_value=min_date,
                max_value=max_date
            )
        else:
            date_filtre = None

    #Application des filtres avant l'export
    df_filtre = df_data.copy()
    
    #Filtre employé
    if employe_filtre != 'Tous' and 'AssigneeFullName' in df_filtre.columns:
        df_filtre = df_filtre[df_filtre['AssigneeFullName'] == employe_filtre]
    
    #Filtre type anomalie
    if type_filtre != 'Tous':
        df_filtre = df_filtre[df_filtre['Statut'] == type_filtre]
    
    #Filtre date
    if date_filtre and 'DateCreation' in df_filtre.columns:
        df_filtre = df_filtre[pd.to_datetime(df_filtre['DateCreation']).dt.date == date_filtre]
    
    #Mise à jour de df_display avec les filtres
    df_display_filtre = df_display_final.loc[df_filtre.index] if not df_filtre.empty else df_display_final

    with col4:
        #Sous-colonnes pour selectbox et bouton d'export
        export_col1, export_col2 = st.columns([0.6, 0.4])
        
        with export_col1:
            #Menu déroulant pour l'export
            export_format = st.selectbox(
                'Format',
                ['CSV', 'Excel', 'PDF'] if HAS_FPDF and HAS_OPENPYXL else ['CSV'],
                index=0,
                key='export_selector'
            )
        
        with export_col2:
            #Bouton d'export
            download_data = None
            mime_type = 'application/octet-stream'
            file_name = f"anomalies_glpi_{datetime.now().strftime('%Y%m%d_%H%M')}.{export_format.lower()}"
            
            try:
                if export_format == 'CSV':
                    download_data = df_display_filtre.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    mime_type = 'text/csv; charset=utf-8-sig'
                
                elif export_format == 'Excel' and HAS_OPENPYXL:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_display_filtre.to_excel(writer, index=False, sheet_name='Anomalies')
                    download_data = output.getvalue()
                    mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                
                elif export_format == 'PDF' and HAS_FPDF:
                    download_data = export_to_pdf(df_display_filtre)
                    mime_type = 'application/pdf'
                
                if download_data is not None:
                    st.download_button(
                        label="Exporter",
                        data=download_data,
                        file_name=file_name,
                        mime=mime_type,
                        use_container_width=True,
                        key=f"download_{export_format}"
                    )
                    
            except Exception as e:
                st.error(f"Erreur export {export_format}: {str(e)}")

    #Affichage du tableau
    if not df_display_filtre.empty:
        #Gestion de la pagination
        LINES_PER_PAGE = 50
        total_lines = len(df_display_filtre)
        total_pages = max(1, (total_lines + LINES_PER_PAGE - 1) // LINES_PER_PAGE)
        current_page = st.session_state.get('pagination_offset', 0)
        
        #Calcul des indices pour la page courante
        start_index = current_page * LINES_PER_PAGE
        end_index = min(start_index + LINES_PER_PAGE, total_lines)
        df_page = df_display_filtre.iloc[start_index:end_index]
        
        #Appliquer le style CSS pour centrer l'en-tête du tableau
        st.markdown("""
        <style>
        table th {
            text-align: center !important;
            vertical-align: middle !important;
            font-weight: 600;
            background-color: #CECECE;
        }
        table td {
            text-align: center !important;
            vertical-align: middle !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        #Afficher le tableau avec le style HTML pour les icônes
        st.markdown(df_page.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        #Navigation simplifiée avec seulement les flèches
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

        with nav_col1:
            if st.button("◀️", disabled=(current_page == 0), use_container_width=True, key="prev_button"):
                st.session_state['pagination_offset'] = current_page - 1
                st.rerun()

        with nav_col2:
            st.markdown(
                f"<div style='text-align: center; font-family: Arial, sans-serif; padding: 10px;'>"
                f"Page {current_page + 1} sur {total_pages}"
                f"</div>", 
                unsafe_allow_html=True
            )

        with nav_col3:
            if st.button("▶️", disabled=(end_index >= total_lines), use_container_width=True, key="next_button"):
                st.session_state['pagination_offset'] = current_page + 1
                st.rerun()
    else:
        st.info("Aucun résultat ne correspond aux filtres sélectionnés.")

if __name__ == "__main__":
    render()