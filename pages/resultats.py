import streamlit as st
import pandas as pd
import io
import base64
from datetime import datetime
import os

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

# DÉSACTIVER LE CACHE STREAMLIT POUR ÉVITER LES NOTIFICATIONS
@st.cache_data(show_spinner=False, persist=False)
def get_cached_data():
    return None

# GÉRER L'ÉTAT DES FILTRES
if 'filters_applied' not in st.session_state:
    st.session_state['filters_applied'] = False
if 'last_filters' not in st.session_state:
    st.session_state['last_filters'] = {}
if 'return_clicked' not in st.session_state:
    st.session_state['return_clicked'] = False
if 'cache_used' not in st.session_state:
    st.session_state['cache_used'] = False

def export_to_pdf(df):
    """Génère un PDF à partir du DataFrame."""
    if not HAS_FPDF:
        st.error("fpdf2 n'est pas installé. Utilisez: pip install fpdf2")
        return None
        
    try:
        pdf = FPDF(orientation='L')
        pdf.add_page()
        pdf.set_font("Arial", size=6)
        
        headers = df.columns.tolist()
        col_width = 280 / len(headers)
        
        for header in headers:
            pdf.cell(col_width, 10, str(header), 1, 0, 'C')
        pdf.ln()

        for index, row in df.iterrows():
            for item in row:
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
    if status == 'OK':
        icon_file = 'status_ok.png'
    elif status == 'Multiples Anomalies':
        icon_file = 'red.png'
    else:
        icon_file = 'danger.png'
    
    try:
        icon_base64 = get_base64_encoded_image(f"styles/icones/{icon_file}")
        icon_html = f'<img src="data:image/png;base64,{icon_base64}" width="16" height="16" style="vertical-align: middle; margin-left: 5px;">'
        return f"{status} {icon_html}"
    except:
        return status

def render():
    """Affiche le contenu de la Page 2."""
    
    # GESTION DU BOUTON RETOUR
    col_retour, col_titre = st.columns([0.1, 0.9])
    with col_retour:
        return_clicked = st.button("←", key="retour_analyse", help="Retour à l'analyse")
        if return_clicked:
            st.session_state['return_clicked'] = True
            st.session_state['current_page'] = 1
            st.session_state['analysis_started'] = False
            st.session_state['filters_applied'] = False
            st.rerun()
            return  # ARRÊTER ICI
    
    # Réinitialiser le flag de retour
    st.session_state['return_clicked'] = False
    
    df_data = st.session_state.get('anomaly_data')
    if df_data is None or (isinstance(df_data, pd.DataFrame) and df_data.empty):
        st.warning("Aucune donnée d'anomalie disponible. Veuillez lancer l'analyse d'abord.")
        return
    
    df_display = df_data.copy()
    
    display_columns = {}
    
    if 'TicketID' in df_display.columns:
        display_columns['ID Ticket'] = df_display['TicketID']
    
    if 'AssigneeFullName' in df_display.columns:
        display_columns['Nom Employé'] = df_display['AssigneeFullName']
    else:
        display_columns['Nom Employé'] = "Non spécifié"
    
    # COLONNE DATE CRÉATION AJOUTÉE ICI
    if 'DateCreation' in df_display.columns:
        try:
            display_columns['Date Création Ticket'] = pd.to_datetime(df_display['DateCreation']).dt.strftime('%Y/%m/%d')
        except:
            display_columns['Date Création Ticket'] = df_display['DateCreation']
    
    if 'TempsHeures' in df_display.columns:
        display_columns['Temps de résolution (h)'] = df_display['TempsHeures'].apply(lambda x: f"{x:.2f}")
    
    if 'NoteTemporelle' in df_display.columns:
        display_columns['Note Temporelle (/10)'] = df_display['NoteTemporelle'].apply(lambda x: f"{x:.2f}")
    
    if 'ScoreSemantique' in df_display.columns:
        display_columns['Score Sémantique (%)'] = df_display['ScoreSemantique'].apply(lambda x: f"{x:.2f}%")
    
    if 'NoteSemantique' in df_display.columns:
        display_columns['Note Sémantique (/10)'] = df_display['NoteSemantique'].apply(lambda x: f"{x:.2f}")
    
    if 'ScoreConcordance' in df_display.columns:
        display_columns['Score Concordance (%)'] = df_display['ScoreConcordance'].apply(lambda x: f"{x:.2f}%")
    
    if 'NoteConcordance' in df_display.columns:
        display_columns['Note Concordance (/10)'] = df_display['NoteConcordance'].apply(lambda x: f"{x:.2f}")
    
    if 'TicketNote' in df_display.columns:
        display_columns['Note Ticket (/10)'] = df_display['TicketNote'].apply(lambda x: f"{x:.2f}")
    
    if 'EmployeeAvgScore' in df_display.columns:
        display_columns['Moyenne Employé (/10)'] = df_display['EmployeeAvgScore'].apply(lambda x: f"{x:.2f}")
    
    if 'Statut' in df_display.columns:
        display_columns['Statut'] = df_display['Statut'].apply(get_status_with_icon)

    df_display_final = pd.DataFrame(display_columns)
        
    st.markdown("""
    <div style='text-align: center; font-family: "Segoe UI", Arial, sans-serif; font-weight: 700; color: #2e2a80; margin-bottom: 20px; font-size: 2rem;'>
    TABLEAU DES ANOMALIES
    </div>
    """, unsafe_allow_html=True)

    # FILTRES
    col1, col2, col3, col4 = st.columns([0.23, 0.23, 0.23, 0.31])
    
    with col1:
        if 'Nom Employé' in df_display_final.columns:
            if 'AssigneeFullName' in df_data.columns:
                employes_list = ['Tous'] + sorted(df_data['AssigneeFullName'].dropna().unique().tolist())
                employe_filtre = st.selectbox("Nom employé", employes_list, key="filtre_employe")
            else:
                employe_filtre = 'Tous'
        else:
            employe_filtre = 'Tous'
    
    with col2:
        all_statuses = ['Tous', 'OK', 'Anomalie de Temps', 'Anomalie Sémantique', 
                       'Anomalie de Concordance', 'Multiples Anomalies', 'Anomalie Indéterminée']
        type_filtre = st.selectbox("Type d'anomalie", all_statuses, key="filtre_type")
    
    with col3:
        # FILTRE DATE AVEC DATE INPUT (COMME SUR L'IMAGE)
        if 'DateCreation' in df_data.columns:
            try:
                # Convertir les dates en format datetime
                df_data['DateCreation_dt'] = pd.to_datetime(df_data['DateCreation'])
                
                # Trouver la date min et max
                min_date = df_data['DateCreation_dt'].min().date()
                max_date = df_data['DateCreation_dt'].max().date()
                
                # Afficher le date picker avec format YYYY/MM/DD
                date_filtre = st.date_input(
                    "Date création ticket",
                    value=None,  # Pas de date par défaut
                    min_value=min_date,
                    max_value=max_date,
                    format="YYYY/MM/DD",
                    key="filtre_date"
                )
                
                # date_filtre peut être None, une date unique, ou un tuple (start, end)
                # On gère comme filtre unique
                if date_filtre:
                    if isinstance(date_filtre, tuple):
                        date_selected = date_filtre[0] if date_filtre[0] else None
                    else:
                        date_selected = date_filtre
                else:
                    date_selected = None
                    
            except Exception as e:
                st.error(f"Erreur traitement dates: {e}")
                date_selected = None
        else:
            date_selected = None

    # TRACKER LES CHANGEMENTS DE FILTRES
    current_filters = {
        'employe_filtre': employe_filtre,
        'type_filtre': type_filtre,
        'date_selected': str(date_selected) if date_selected else None
    }
    
    filters_changed = current_filters != st.session_state.get('last_filters', {})
    
    if filters_changed:
        st.session_state['last_filters'] = current_filters
        st.session_state['filters_applied'] = True
        st.session_state['pagination_offset'] = 0

    # APPLICATION DES FILTRES
    df_filtre = df_data.copy()
    
    if employe_filtre != 'Tous' and 'AssigneeFullName' in df_filtre.columns:
        df_filtre = df_filtre[df_filtre['AssigneeFullName'] == employe_filtre]
    
    if type_filtre != 'Tous' and 'Statut' in df_filtre.columns:
        df_filtre = df_filtre[df_filtre['Statut'] == type_filtre]
    
    if date_selected and 'DateCreation' in df_filtre.columns:
        try:
            df_filtre['DateCreation_dt'] = pd.to_datetime(df_filtre['DateCreation'])
            df_filtre = df_filtre[df_filtre['DateCreation_dt'].dt.date == date_selected]
        except Exception as e:
            st.error(f"Erreur filtrage date: {e}")

    # PRÉPARATION AFFICHAGE FILTRÉ
    if not df_filtre.empty:
        df_display_filtre = df_display_final.loc[df_filtre.index]
    else:
        df_display_filtre = pd.DataFrame(columns=df_display_final.columns)

    # STATISTIQUES
    total_tickets = len(df_filtre)
    
    if 'TempsHeures' in df_filtre.columns and not df_filtre.empty:
        temps_moyen = df_filtre['TempsHeures'].mean()
        ecart_type = df_filtre['TempsHeures'].std()
    else:
        temps_moyen = 0
        ecart_type = 0
    
    st.markdown(f"""
    <div style='display: flex; justify-content: space-between; margin: 20px 0 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;'>
        <div style='text-align: center; flex: 1;'>
            <div style='font-size: 14px; font-weight: bold; color: #2e2a80;'>Total Tickets</div>
            <div style='font-size: 24px; font-weight: bold; color: #333;'>{total_tickets}</div>
        </div>
        <div style='text-align: center; flex: 1;'>
            <div style='font-size: 14px; font-weight: bold; color: #2e2a80;'>Temps Moyen Résolution</div>
            <div style='font-size: 24px; font-weight: bold; color: #333;'>{temps_moyen:.2f}h</div>
        </div>
        <div style='text-align: center; flex: 1;'>
            <div style='font-size: 14px; font-weight: bold; color: #2e2a80;'>Écart Type</div>
            <div style='font-size: 24px; font-weight: bold; color: #333;'>{ecart_type:.2f}h</div>
        </div>
        <div style='text-align: center; flex: 1;'>
            <div style='font-size: 14px; font-weight: bold; color: #2e2a80;'>SLA</div>
            <div style='font-size: 24px; font-weight: bold; color: #333;'>4h</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with col4:
        export_col1, export_col2 = st.columns([0.6, 0.4])
        
        with export_col1:
            export_format = st.selectbox(
                'Format',
                ['CSV', 'Excel', 'PDF'] if HAS_FPDF and HAS_OPENPYXL else ['CSV'],
                index=0,
                key='export_selector'
            )
        
        with export_col2:
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

    # AFFICHAGE TABLEAU
    if not df_display_filtre.empty:
        LINES_PER_PAGE = 50
        total_lines = len(df_display_filtre)
        total_pages = max(1, (total_lines + LINES_PER_PAGE - 1) // LINES_PER_PAGE)
        current_page = st.session_state.get('pagination_offset', 0)
        
        start_index = current_page * LINES_PER_PAGE
        end_index = min(start_index + LINES_PER_PAGE, total_lines)
        df_page = df_display_filtre.iloc[start_index:end_index]
        
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
        
        /* Masquer la notification de cache Streamlit */
        .stAlert {
            display: none !important;
        }
        div[data-testid="stDecoration"] {
            display: none !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(df_page.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        # PAGINATION
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

        with nav_col1:
            if st.button("◀", disabled=(current_page == 0), use_container_width=True, key="prev_button"):
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
            if st.button("▶", disabled=(end_index >= total_lines), use_container_width=True, key="next_button"):
                st.session_state['pagination_offset'] = current_page + 1
                st.rerun()
    else:
        st.info("Aucun résultat ne correspond aux filtres sélectionnés.")

if __name__ == "__main__":
    render()