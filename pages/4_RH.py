#Avec et sans kmeans 
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import folium
import folium.plugins as plugins
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.colors import to_hex, ListedColormap
from streamlit_option_menu import option_menu
from geopy.geocoders import Nominatim
from db_connection import get_connection

st.logo("LOGO.png", icon_image="Logom.png")


# def load_managers_from_db():
#     conn = get_connection()
#     if conn:
#         df = pd.read_sql("SELECT * FROM rh", conn)
#         conn.close()
#         return df
#     return pd.DataFrame()
def load_managers_from_db():
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rh")
            columns = [col[0] for col in cursor.description]
            data = cursor.fetchall()
            df = pd.DataFrame(data, columns=columns)
            cursor.close()
            conn.close()
            return df
        except Exception as e:
            print(f"Erreur lors du chargement de la table RH : {e}")
            conn.close()
    return pd.DataFrame()

def load_stores_from_db():
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pdv")
        columns = [col[0] for col in cursor.description]
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=columns)
        cursor.close()
        conn.close()
        return df
    return pd.DataFrame()

# def load_stores_from_db():
#     conn = get_connection()
#     if conn:
#         df = pd.read_sql("SELECT * FROM pdv", conn)
#         conn.close()
#         return df
#     return pd.DataFrame()

# Charger les données
managers_original = load_managers_from_db()
stores_original = load_stores_from_db()

# 👇 Ajoute ici :
if 'managers_optimized' in st.session_state:
    managers = st.session_state.managers_optimized.copy()
else:
    managers = managers_original.copy()

stores = stores_original.copy()

if managers.empty or stores.empty:
    st.warning("⚠️ Aucune donnée n'a été trouvée dans la base de données.")
    st.stop()
# Initialiser le géocodeur
geolocator = Nominatim(user_agent="my_geocoding_app_v1.0")

def geocode_address(address):
    """Convertit une adresse en latitude et longitude en utilisant geopy."""
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        st.warning("Adresse introuvable. Veuillez vérifier et réessayer.")
        return None, None
# Ajouter une section pour la réaffectation manuelle des chefs de secteurs
st.subheader("Remplacement manuelle des chefs de secteurs")
col1, col2 = st.columns(2)
with col1:
    def blend_cmap(cmap1, cmap2, n_colors):
        colors1 = plt.get_cmap(cmap1)(np.linspace(0, 1, n_colors // 2))
        colors2 = plt.get_cmap(cmap2)(np.linspace(0, 1, n_colors // 2))
        colors = np.vstack((colors1, colors2))
        return ListedColormap(colors) 

    # Load data from Excel files
    #managers = pd.read_excel("CS Data.xlsx")
    #stores = pd.read_excel("Datakiss Template 2024 REVILLARS.xlsx")

    # Convert latitude and longitude from string to numeric (float)
    stores['lat'] = pd.to_numeric(stores['lat'], errors='coerce')
    stores['long'] = pd.to_numeric(stores['long'], errors='coerce')
    managers['Latitude'] = pd.to_numeric(managers['Latitude'], errors='coerce')
    managers['Longitude'] = pd.to_numeric(managers['Longitude'], errors='coerce')

    # Drop rows with NaN values
    stores.dropna(subset=['lat', 'long'], inplace=True)
    managers.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    # Ajout du sélecteur de secteur
    sector_options = managers['Code_secteur'].unique().tolist()
    selected_sector = st.selectbox('Filtrer par secteur', ['Tous'] + sector_options)

    num_clusters = len(managers)
    if len(stores) < num_clusters:
        st.write(f"Not enough stores to match the number of managers. Only {len(stores)} stores available for {num_clusters} managers.")
        num_clusters = len(stores)

    if num_clusters > 0:
        # Target number of stores per manager (assuming roughly equal distribution is desired)
        target_per_manager = stores.shape[0] // num_clusters

        def find_nearest_manager(cluster_centroid, managers):
            distances = cdist([cluster_centroid], managers[['Latitude', 'Longitude']])
            nearest_manager_idx = distances.argmin()
            manager = managers.iloc[nearest_manager_idx]
            return pd.Series([manager['Code_secteur'], manager['Latitude'], manager['Longitude']],
                                index=['Code_secteur', 'Manager Latitude', 'Manager Longitude'])
        stores[['Code_secteur', 'Manager Latitude', 'Manager Longitude']] = stores.apply(
            lambda x: find_nearest_manager((x['lat'], x['long']), managers), axis=1, result_type='expand')
        unique_sectors = stores['Code_secteur'].unique()
        custom_cmap = blend_cmap('Paired', 'tab20b', len(unique_sectors))
        color_palette = custom_cmap(np.linspace(0, 1, len(unique_sectors)))
        sector_to_color = {sector: to_hex(color) for sector, color in zip(unique_sectors, color_palette)}

        def calculate_new_charge(stores, managers):
            temps_clientele_per_sector_new = stores.groupby('Code_secteur').apply(lambda x: (x['Temps'] * x['Frequence']).sum()).reset_index(name='New_Temps passé clientèle')

            charge_per_sector_new = pd.merge(temps_clientele_per_sector_new, managers[['Code_secteur', 'Nb_jour_terrain_par_an', 'Nb_heure_par_jour']], on='Code_secteur', how='left')
            charge_per_sector_new['Temps terrain effectif'] = charge_per_sector_new['Nb_jour_terrain_par_an'] * charge_per_sector_new['Nb_heure_par_jour'] * 60

            temps_route = 25000

            charge_per_sector_new['New_Charge'] = ((charge_per_sector_new['New_Temps passé clientèle'] + temps_route) / charge_per_sector_new['Temps terrain effectif']) * 100
            return charge_per_sector_new[['Code_secteur', 'New_Charge']] 

        map = folium.Map(location=[46.2276, 2.2137], zoom_start=7, tiles=None)
        plugins.Fullscreen(position='topright', force_separate_button=True).add_to(map)
        folium.TileLayer(
            'CartoDB positron',
            name='Fond de carte CartoDB',
            attr='© OpenStreetMap contributors & © CartoDB'
        ).add_to(map)
        folium.TileLayer(
            'OpenStreetMap',
            name='Fond de carte OpenStreetMap',
            attr='© OpenStreetMap contributors'
        ).add_to(map)

        # Ajouter le contrôle de couche
        folium.LayerControl().add_to(map)

        # Filtrage des magasins par secteur sélectionné
        if selected_sector != 'Tous':
            filtered_stores = stores[stores['Code_secteur'] == selected_sector]
        else:
            filtered_stores = stores

        for _, row in filtered_stores.iterrows():
            sector_color = sector_to_color[row['Code_secteur']]
            folium.CircleMarker(
                location=[row['lat'], row['long']],
                radius=5,
                color=sector_color,
                fill=True,
                fill_color=sector_color,
                popup=f"Store ID: {row['Code_mag']} - Sector: {row['Code_secteur']}"
            ).add_to(map)
            folium.PolyLine(
                locations=[[row['Manager Latitude'], row['Manager Longitude']], [row['lat'], row['long']]],
                color=sector_color,
                opacity=0.3,
                weight=2
            ).add_to(map)

        # Filtrage des managers par secteur sélectionné
        if selected_sector != 'Tous':
            filtered_managers = managers[managers['Code_secteur'] == selected_sector]
        else:
            filtered_managers = managers

        for _, manager in filtered_managers.iterrows():
            manager_color = sector_to_color.get(manager['Code_secteur'], "#808080")  # Use gray color if sector not found
            folium.Marker(
                [manager['Latitude'], manager['Longitude']],
                icon=folium.Icon(icon="info-sign", color="red"),
                popup=f"Manager ID: {manager['Code_secteur']}",
                tooltip=folium.Tooltip(text=f"Sector {manager['Code_secteur']}", permanent=True)
            ).add_to(map)

        # Display the map in Streamlit
        st_data = folium.Figure(width=700, height=500).add_child(map)
        st.components.v1.html(st_data.render(), width=650, height=700)
        # Ajouter le graphique optimisé de la nouvelle charge en bas de la carte
        if 'managers_optimized' in st.session_state and st.session_state.managers_optimized is not None:
            st.subheader("Graphique de la charge optimisée par secteur")
            
            # Calcul de la nouvelle charge après optimisation
            new_charge_per_sector = calculate_new_charge(stores, managers)

            # Ordonner les données optimisées comme dans le DataFrame principal
            charge_per_sector_optimized = new_charge_per_sector.set_index('Code_secteur').reindex(managers['Code_secteur']).reset_index()

            # Créer les couleurs en fonction de la charge
            colors_optimized = ['red' if x > 100 else 'green' for x in charge_per_sector_optimized['New_Charge']]

            # Tracer le graphique de la nouvelle charge
            # Créer une colonne pour la légende texte
            charge_per_sector_optimized['Charge_label'] = charge_per_sector_optimized['New_Charge'].apply(
                lambda x: '> 100 %' if x > 100 else '≤ 100 %'
            )

            # Créer le graphique Plotly avec les couleurs actuelles
            fig = px.bar(
                charge_per_sector_optimized,
                x='Code_secteur',
                y='New_Charge',
                color='Charge_label',
                labels={'New_Charge': 'Charge (%)', 'Code_secteur': 'Secteur', 'Charge_label': ''},
                title='Charge par Secteur (Optimisé)',
                text=charge_per_sector_optimized['New_Charge'].round(2),
                color_discrete_map={
                    '> 100 %': 'rgb(31, 119, 180)',   # Bleu foncé (garde tes couleurs actuelles)
                    '≤ 100 %': 'rgb(174, 199, 232)'   # Bleu clair
                }
            )

            # Ligne horizontale à 100 %
            fig.add_hline(y=100, line_dash="dash", line_color="black")

            # Améliorations visuelles
            fig.update_traces(textposition='outside')
            fig.update_layout(
                showlegend=True,
                legend_title_text="Charge",
                xaxis_tickangle=-45,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.sidebar:
                # Ajout d'un formulaire pour saisir une adresse et la convertir en coordonnées
                st.markdown("### Convertir une adresse en coordonnées géographiques")
                address_input = st.text_input("Entrez l'adresse à convertir :")

                if st.button("Convertir l'adresse"):
                    if address_input:
                        lat, lon = geocode_address(address_input)
                        if lat and lon:
                            st.success(f"Latitude: {lat}, Longitude: {lon}")
                    else:
                        st.error("Veuillez entrer une adresse valide.")

with col2:
    # Fonction de style pour colorer la colonne 'Charge'
    def color_charge(val):
        if val is None or val == "None":
            color = 'white'
        else:
            try:
                val = str(val).rstrip('%')  # Convertir en chaîne si ce n'est pas déjà le cas
                val = float(val)  # Convertir en float pour la comparaison
                if val > 100:
                    color = 'rgba(255, 0, 0, 0.3)'  # Rouge avec une opacité de 30%
                else:
                    color = 'rgba(0, 255, 0, 0.3)'  # Vert avec une opacité de 30%
            except ValueError:
                color = 'white'
        return f'background-color: {color}'

    # ✅ Recalculer la charge à partir des magasins optimisés
    if 'managers_optimized' in st.session_state and st.session_state.managers_optimized is not None:
        def calculate_charge_from_optimized(stores_df, managers_df):
            charge_df = stores_df.groupby('Code_secteur').apply(
                lambda x: (x['Temps'] * x['Frequence']).sum()).reset_index(name='Temps_clientèle')

            merged = pd.merge(
                charge_df,
                managers_df[['Code_secteur', 'Nb_jour_terrain_par_an', 'Nb_heure_par_jour']],
                on='Code_secteur',
                how='left'
            )

            merged['Temps terrain effectif'] = merged['Nb_jour_terrain_par_an'] * merged['Nb_heure_par_jour'] * 60
            merged['New_Charge'] = ((merged['Temps_clientèle'] + 25000) / merged['Temps terrain effectif']) * 100
            merged['New_Charge'] = merged['New_Charge'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")

            return merged[['Code_secteur', 'New_Charge']]

        # 🔁 Recalcul des charges
        new_charge_df = calculate_charge_from_optimized(stores, st.session_state.managers_optimized)

        # 🔁 Recalcul PDV affectés & Visites nécessaires
        pdv_visites_df = stores.groupby('Code_secteur').agg(
            New_PDV_affectés=('Code_mag', 'count'),
            New_Visites_nécessaires=('Frequence', 'sum')
        ).reset_index()
        pdv_visites_df.columns = ['Code_secteur', 'New_PDV affectés', 'New_Visites nécessaires']

        # 🔁 Fusionner tout ensemble
        full_merged = pd.merge(new_charge_df, pdv_visites_df, on='Code_secteur', how='left')

        st.session_state.managers_optimized = pd.merge(
            st.session_state.managers_optimized.drop(columns=['New_Charge', 'New_PDV affectés', 'New_Visites nécessaires'], errors='ignore'),
            full_merged,
            on='Code_secteur',
            how='left'
        )

        st.subheader("Données détaillées générales après l'optimisation")
        detailed_data = st.empty()
        optimized_columns_to_display = ['Code_secteur', 'Nom', 'Prenom', 'Adresse', 'New_PDV affectés', 'New_Visites nécessaires', 'New_Charge']

        # Filtrage
        if 'show_modify_form' not in st.session_state:
            st.session_state['show_modify_form'] = False

        if st.button('Modifier infos'):
            st.session_state['show_modify_form'] = not st.session_state['show_modify_form']

        search_query = st.text_input("Rechercher par nom, prénom ou adresse")

        if search_query:
            filtered_managers_optimized = st.session_state.managers_optimized[
                st.session_state.managers_optimized.apply(
                    lambda row: search_query.lower() in (str(row['Nom']) + str(row['Prenom']) + str(row['Adresse'])).lower(), axis=1)
            ]
        else:
            filtered_managers_optimized = st.session_state.managers_optimized
            
        # Afficher les résultats filtrés
        #st.write("Résultats filtrés :", filtered_managers_optimized)

        # Pagination
        rows_per_page = 5
        total_rows = len(st.session_state.managers_optimized)
        total_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page > 0 else 0)
        current_page = st.session_state.get('current_page', 1)

        start_idx = (current_page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        paginated_managers = filtered_managers_optimized.iloc[start_idx:end_idx]

        # Affichage des données paginées
        styled_optimized_managers = paginated_managers[optimized_columns_to_display].style.applymap(color_charge, subset=['New_Charge'])
        detailed_data.dataframe(styled_optimized_managers)

        if st.session_state['show_modify_form']:

            for index, row in paginated_managers.iterrows():
                # Initialisation de l'état pour chaque manager
                if f'simulate_{index}' not in st.session_state:
                    st.session_state[f'simulate_{index}'] = False
                if f'modify_{index}' not in st.session_state:
                    st.session_state[f'modify_{index}'] = False
                if f'validated_{index}' not in st.session_state:
                    st.session_state[f'validated_{index}'] = False

                with st.form(key=f"form_{index}"):
                    cols = st.columns(len(optimized_columns_to_display) + 1)
                    for col, column_name in zip(cols, optimized_columns_to_display):
                        col.write(row[column_name])

                    if cols[-1].form_submit_button("Modifier"):
                        st.session_state[f'modify_{index}'] = True

            for index, row in paginated_managers.iterrows():
                if st.session_state.get(f'modify_{index}', False):
                    st.subheader(f"Modifier les paramètres pour {row['Nom']} {row['Prenom']}")
                    new_name = st.text_input("Nom", row['Nom'], key=f'name_{index}')
                    new_surname = st.text_input("Prénom", row['Prenom'], key=f'surname_{index}')
                    new_address = st.text_input("Nouvelle Adresse", row['Adresse'], key=f'address_{index}')
                    new_latitude = st.number_input("Nouvelle Latitude", value=row['Latitude'], key=f'latitude_{index}')
                    new_longitude = st.number_input("Nouvelle Longitude", value=row['Longitude'], key=f'longitude_{index}')

                    simulate_clicked = st.button('Simuler', key=f'simulate_button_{index}')
                    if simulate_clicked:
                        # Mettre à jour les informations du manager sans valider
                        st.session_state[f'simulate_{index}'] = True
                        st.session_state.managers_optimized.at[index, 'Nom'] = new_name
                        st.session_state.managers_optimized.at[index, 'Prenom'] = new_surname
                        st.session_state.managers_optimized.at[index, 'Adresse'] = new_address
                        st.session_state.managers_optimized.at[index, 'Latitude'] = new_latitude
                        st.session_state.managers_optimized.at[index, 'Longitude'] = new_longitude
                        st.session_state['current_simulation_index'] = index

                    if st.session_state.get(f'simulate_{index}', False) and st.session_state['current_simulation_index'] == index:
                        st.subheader("Résultat de la simulation")
                        new_charge = st.session_state.managers_optimized.at[index, 'New_Charge']
                                
                        # Affichage des résultats de la simulation dans une section distincte
                        with st.expander("Résultat de simulation", expanded=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(label="Charge", value=f"{new_charge}")
                            with col2:
                                st.metric(label="Charge Après Modif", value=f"{new_charge}")
                            # Ajoutez d'autres métriques comme nécessaire
                            # st.metric(label="Autre métrique", value="valeur")

                        validate_clicked = st.button('Valider', key=f'validate_{index}')
                        if validate_clicked:
                            st.session_state[f'modify_{index}'] = False
                            st.session_state[f'simulate_{index}'] = False
                            st.session_state[f'validated_{index}'] = True
                            st.session_state['current_simulation_index'] = None
                            
                            # Mettre à jour les coordonnées des managers dans la liste des magasins
                            stores.loc[stores['Code_secteur'] == row['Code_secteur'], 'Manager Latitude'] = new_latitude
                            stores.loc[stores['Code_secteur'] == row['Code_secteur'], 'Manager Longitude'] = new_longitude

                            # Réinitialiser l'état de modification pour cacher le formulaire
                            st.session_state[f'modify_{index}'] = False
                            
                            # Afficher la carte mise à jour après simulation
                            updated_map = folium.Map(location=[46.2276, 2.2137], zoom_start=7, tiles=None)
                            plugins.Fullscreen(position='topright', force_separate_button=True).add_to(updated_map)
                            folium.TileLayer(
                                'CartoDB positron',
                                name='Fond de carte CartoDB',
                                attr='© OpenStreetMap contributors & © CartoDB'
                            ).add_to(updated_map)
                            folium.TileLayer(
                                'OpenStreetMap',
                                name='Fond de carte OpenStreetMap',
                                attr='© OpenStreetMap contributors'
                            ).add_to(updated_map)

                            # Ajouter le contrôle de couche
                            folium.LayerControl().add_to(updated_map)

                            for _, row in stores.iterrows():
                                sector_color = sector_to_color[row['Code_secteur']]
                                folium.CircleMarker(
                                    location=[row['lat'], row['long']],
                                    radius=5,
                                    color=sector_color,
                                    fill=True,
                                    fill_color=sector_color,
                                    popup=f"Store ID: {row['Code_mag']} - Sector: {row['Code_secteur']}"
                                ).add_to(updated_map)
                                folium.PolyLine(
                                    locations=[[row['Manager Latitude'], row['Manager Longitude']], [row['lat'], row['long']]],
                                    color=sector_color,
                                    opacity=0.3,
                                    weight=2
                                ).add_to(updated_map)

                            for _, manager in st.session_state.managers_optimized.iterrows():
                                manager_color = sector_to_color.get(manager['Code_secteur'], "#808080")  # Use gray color if sector not found
                                folium.Marker(
                                    [manager['Latitude'], manager['Longitude']],
                                    icon=folium.Icon(icon="info-sign", color="red"),
                                    popup=f"Manager ID: {manager['Code_secteur']}",
                                    tooltip=folium.Tooltip(text=f"Sector {manager['Code_secteur']}", permanent=True)
                                ).add_to(updated_map)

                            # Display the updated map in Stte;psreamlit
                            st_data_updated = folium.Figure(width=700, height=500).add_child(updated_map)
                            st.components.v1.html(st_data_updated.render(), width=650, height=700) 
                                    # Vérifier si des modifications ont été faites et afficher le bouton "Enregistrer les modifications"
                            if 'modified_map' not in st.session_state:
                                st.session_state['modified_map'] = updated_map

                            if 'modified_map' in st.session_state and st.button("Enregistrer les modifications"):
                                # Copier la carte modifiée dans la carte principale
                                st.session_state["final_map"] = st.session_state["modified_map"]

                                # Conserver toutes les modifications
                                for index, row in st.session_state.managers_optimized.iterrows():
                                    # Mise à jour de chaque magasin en fonction des modifications enregistrées
                                    stores.loc[stores['Code_secteur'] == row['Code_secteur'], 'Manager Latitude'] = row['Latitude']
                                    stores.loc[stores['Code_secteur'] == row['Code_secteur'], 'Manager Longitude'] = row['Longitude']
                                
                                st.success("Modifications enregistrées et appliquées à la carte principale.")
                                
                                # Affichage de la carte principale après les modifications
                            if 'final_map' in st.session_state:
                                st.components.v1.html(st.session_state["final_map"].render(), width=650, height=700)
                            # Mettre à jour la carte et le tableau
            

        styled_optimized_managers = st.session_state.managers_optimized[optimized_columns_to_display].style.applymap(color_charge, subset=['New_Charge'])
        detailed_data.dataframe(styled_optimized_managers)

        # Pagination controls
        pagination_col1, pagination_col2, pagination_col3 = st.columns([1, 2, 1])
        if current_page > 1:
            if pagination_col1.button('Précédent'):
                st.session_state['current_page'] = current_page - 1
                st.experimental_rerun()

        pagination_col2.write(f"Page {current_page} sur {total_pages}")

        if current_page < total_pages:
            if pagination_col3.button('Suivant'):
                st.session_state['current_page'] = current_page + 1
                st.experimental_rerun()