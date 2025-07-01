import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import folium
import folium.plugins as plugins
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, ListedColormap
from streamlit_option_menu import option_menu
from geopy.geocoders import Nominatim
from db_connection import get_connection
import plotly.express as px

st.logo("LOGO.png", icon_image="Logom.png")

def load_managers_from_db():
    conn = get_connection()
    if conn:
        df = pd.read_sql("SELECT * FROM rh", conn)
        conn.close()
        return df
    return pd.DataFrame()

def load_stores_from_db():
    conn = get_connection()
    if conn:
        df = pd.read_sql("SELECT * FROM pdv", conn)
        conn.close()
        return df
    return pd.DataFrame()

# Charger les données
managers = load_managers_from_db()
stores = load_stores_from_db()

# Charger les données
managers_original = managers.copy()
stores_original = stores.copy()

if managers.empty or stores.empty:
    st.warning("⚠️ Aucune donnée n'a été trouvée dans la base de données.")
    st.stop()

geolocator = Nominatim(user_agent="geoapiExercises")

def calculate_new_charge(stores, managers):
    temps_clientele_per_sector_new = stores.groupby('Code_secteur').apply(lambda x: (x['Temps'] * x['Frequence']).sum()).reset_index(name='New_Temps passé clientèle')

    charge_per_sector_new = pd.merge(temps_clientele_per_sector_new, managers[['Code_secteur', 'Nb_jour_terrain_par_an', 'Nb_heure_par_jour']], on='Code_secteur', how='left')
    charge_per_sector_new['Temps terrain effectif'] = charge_per_sector_new['Nb_jour_terrain_par_an'] * charge_per_sector_new['Nb_heure_par_jour'] * 60

    temps_route = 25000

    charge_per_sector_new['New_Charge'] = ((charge_per_sector_new['New_Temps passé clientèle'] + temps_route) / charge_per_sector_new['Temps terrain effectif']) * 100
    return charge_per_sector_new[['Code_secteur', 'New_Charge']] 

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

def format_charge(value):
    if pd.notnull(value):
        try:
            value = float(value)
            formatted_value = f"{value:.2f}%"
            # Remove the trailing ".0" if the formatted value ends with ".0%"
            if formatted_value.endswith('.0%'):
                formatted_value = formatted_value.replace('.0%', '%')
            return formatted_value
        except ValueError:
            return str(value)
    return "None"

# Fonction pour géocoder une adresse
def geocode_address(address):
    location = geolocator.geocode(address)
    return location.latitude, location.longitude

# Ajouter une section pour la réaffectation manuelle des chefs de secteurs
st.subheader("Réaffectation manuelle des Clients (Magasins)")
col1, col2 = st.columns(2)

with col1:
    def blend_cmap(cmap1, cmap2, n_colors):
        colors1 = plt.get_cmap(cmap1)(np.linspace(0, 1, n_colors // 2))
        colors2 = plt.get_cmap(cmap2)(np.linspace(0, 1, n_colors // 2))
        colors = np.vstack((colors1, colors2))
        return ListedColormap(colors)

    # Charger les données depuis les fichiers Excel
    #managers = pd.read_excel("CS Data.xlsx")
    #stores = pd.read_excel("Datakiss Template 2024 REVILLARS.xlsx")

    # Convertir la latitude et la longitude de chaîne en numérique (float)
    stores['lat'] = pd.to_numeric(stores['lat'], errors='coerce')
    stores['long'] = pd.to_numeric(stores['long'], errors='coerce')
    managers['Latitude'] = pd.to_numeric(managers['Latitude'], errors='coerce')
    managers['Longitude'] = pd.to_numeric(managers['Longitude'], errors='coerce')

    # Supprimer les lignes avec des valeurs NaN
    stores.dropna(subset=['lat', 'long'], inplace=True)
    managers.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    # Ajout du sélecteur de secteur
    # sector_options = managers['Code_secteur'].unique().tolist()
    # selected_sector = st.selectbox('Filtrer par secteur', ['Tous'] + sector_options)

    num_clusters = len(managers)
    if len(stores) < num_clusters:
        st.write(f"Not enough stores to match the number of managers. Only {len(stores)} stores available for {num_clusters} managers.")
        num_clusters = len(stores)

    if num_clusters > 0:
        # Calcul des informations initiales avant la réaffectation
        def calculate_before_after_reallocation(stores, managers, sectors):
            # Calculer les visites, charge et nombre de magasins par secteur
            visits_per_sector = stores.groupby('Code_secteur')['Frequence'].sum().reset_index(name='Visites nécessaires')
            stores_per_sector = stores.groupby('Code_secteur').size().reset_index(name='PDV affectés')
            charge_per_sector = calculate_new_charge(stores, managers)

            sector_info = pd.merge(managers[['Code_secteur', 'Nom']], visits_per_sector, on='Code_secteur', how='left')
            sector_info = pd.merge(sector_info, stores_per_sector, on='Code_secteur', how='left')
            sector_info = pd.merge(sector_info, charge_per_sector, on='Code_secteur', how='left')

            return sector_info[sector_info['Code_secteur'].isin(sectors)]
        def calculate_sectors_before(stores, managers, source_sector, target_sector):
            source_info = calculate_before_after_reallocation(stores, managers, [source_sector])
            target_info = calculate_before_after_reallocation(stores, managers, [target_sector])
            return source_info, target_info

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
        
        # Initialisez la carte principale en vérifiant s'il y a une carte finale sauvegardée dans la session
        if "final_map" in st.session_state:
            st_data = folium.Figure(width=700, height=500).add_child(st.session_state["final_map"])
        else:
            # Utiliser la carte initiale ici si la carte finale n'existe pas
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

            for _, row in stores.iterrows():
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

            for _, manager in managers.iterrows():
                manager_color = sector_to_color.get(manager['Code_secteur'], "#808080")  # Use gray color if sector not found
                folium.Marker(
                    [manager['Latitude'], manager['Longitude']],
                    icon=folium.Icon(icon="info-sign", color="red"),
                    popup=f"Manager ID: {manager['Code_secteur']}",
                    tooltip=folium.Tooltip(text=f"Sector {manager['Code_secteur']}", permanent=True)
                ).add_to(map)

            st_data = folium.Figure(width=700, height=500).add_child(map)

        # Affichez la carte
        st.components.v1.html(st_data.render(), width=650, height=700)

    sector_options = managers['Code_secteur'].unique().tolist()

    # Filtre de sélection multiple pour choisir les secteurs à afficher
    selected_sector = st.selectbox('Choisissez le secteur source à afficher', ['Aucun'] + sector_options)

    # Filtrer les magasins en fonction des secteurs sélectionnés
    if selected_sector:
        filtered_stores = stores[stores['Code_secteur'] == selected_sector]
    else:
        filtered_stores = pd.DataFrame()  # Si aucun secteur n'est sélectionné, ne rien afficher

    # Interface pour réaffecter manuellement un magasin
    new_sector = st.selectbox("Sélectionnez le nouveau secteur destination où transférer les magasins", sector_options)

    # Liste interactive pour sélectionner les magasins à réaffecter
    selected_stores = []
    if not filtered_stores.empty:
        st.write("Sélectionnez les magasins à réaffecter:")
        with st.expander("Cliquez pour voir la liste des magasins", expanded=False):
            for index, row in filtered_stores.iterrows():
                if st.checkbox(f"**Magasin** {row['Code_mag']} - {row['Nom_mag']} - **Secteur** {row['Code_secteur']}", key=index):
                    selected_stores.append(index)
    else:
        st.write("Aucun magasin à afficher pour les secteurs sélectionnés.")

    if st.button("Réaffecter les magasins sélectionnés"):
        if selected_stores:

            selected_stores_info = stores[stores['Code_secteur'] == new_sector]
            target_sector = new_sector

            # Calculer les informations avant la réaffectation pour les deux secteurs
            sector_before_source, sector_before_target = calculate_sectors_before(stores, managers, selected_sector, target_sector)

            # Réaffecter les magasins sélectionnés
            stores.loc[selected_stores, 'Code_secteur'] = new_sector
            manager = managers[managers['Code_secteur'] == new_sector].iloc[0]
            stores.loc[selected_stores, 'Manager Latitude'] = manager['Latitude']
            stores.loc[selected_stores, 'Manager Longitude'] = manager['Longitude']
            

            st.success(f"{len(selected_stores)} magasins réaffecté(s) au secteur {new_sector}.")

            # Calcul des informations après la réaffectation
            sectors_after = [new_sector, selected_sector]
            sector_after = calculate_before_after_reallocation(stores, managers, sectors_after)

            # Enregistrer dans st.session_state
            st.session_state['sectors_after'] = sectors_after
            st.session_state['sector_after'] = sector_after

            # Vérifier s'il y a une carte modifiée déjà existante
            if "modified_map" in st.session_state:
                modified_map = st.session_state["modified_map"]
            else:
                modified_map = folium.Map(location=[46.2276, 2.2137], zoom_start=7, tiles=None)
                plugins.Fullscreen(position='topright', force_separate_button=True).add_to(modified_map)
                folium.TileLayer('CartoDB positron', name='Fond de carte CartoDB', attr='© OpenStreetMap contributors & © CartoDB').add_to(modified_map)
                folium.TileLayer('OpenStreetMap', name='Fond de carte OpenStreetMap', attr='© OpenStreetMap contributors').add_to(modified_map)

            # Réafficher les magasins mis à jour sur la carte, en s'assurant de ne tracer que les lignes correspondant au secteur actuel
            for _, row in stores.iterrows():
                sector_color = sector_to_color.get(row['Code_secteur'], "#808080")
                folium.CircleMarker(
                    location=[row['lat'], row['long']],
                    radius=5,
                    color=sector_color,
                    fill=True,
                    fill_color=sector_color,
                    popup=f"Store ID: {row['Code_mag']} - Sector: {row['Code_secteur']}"
                ).add_to(modified_map)

                # Vérifier que la ligne est tracée uniquement vers le secteur actuel du magasin
                manager_loc = managers[managers['Code_secteur'] == row['Code_secteur']].iloc[0]
                folium.PolyLine(
                    locations=[[manager_loc['Latitude'], manager_loc['Longitude']], [row['lat'], row['long']]],
                    color=sector_to_color.get(row['Code_secteur'], "#808080"),
                    opacity=0.3,
                    weight=2
                ).add_to(modified_map)

            # Affichage des managers sur la carte
            for _, manager in managers.iterrows():
                manager_color = sector_to_color.get(manager['Code_secteur'], "#808080")
                folium.Marker(
                    [manager['Latitude'], manager['Longitude']],
                    icon=folium.Icon(icon="info-sign", color="red"),
                    popup=f"Manager ID: {manager['Code_secteur']}",
                    tooltip=folium.Tooltip(text=f"Sector {manager['Code_secteur']}", permanent=True)
                ).add_to(modified_map)

            # Enregistrer la carte modifiée
            st.session_state["modified_map"] = modified_map

            st_data = folium.Figure(width=700, height=500).add_child(modified_map)
            st.components.v1.html(st_data.render(), width=650, height=700)

            # Appliquer la fonction format_charge pour formater les valeurs de New_Charge
            sector_before_source['New_Charge'] = sector_before_source['New_Charge'].apply(format_charge)
            sector_before_target['New_Charge'] = sector_before_target['New_Charge'].apply(format_charge)
            sector_after['New_Charge'] = sector_after['New_Charge'].apply(format_charge)

            # Calculer la différence de charge entre avant et après réaffectation
            charge_difference = sector_after.set_index('Code_secteur')['New_Charge'].apply(lambda x: float(x.rstrip('%'))) - \
                                pd.concat([sector_before_source, sector_before_target]).set_index('Code_secteur')['New_Charge'].apply(lambda x: float(x.rstrip('%')))

            # Afficher les résultats de la simulation avant et après
            st.subheader("Résultats de la simulation")
            st.write("**Avant la réaffectation**")
            st.dataframe(pd.concat([sector_before_source, sector_before_target]).style.applymap(color_charge, subset=['New_Charge']))

            st.write("**Après la réaffectation**")
            st.dataframe(sector_after.style.applymap(color_charge, subset=['New_Charge']))

            # Ajouter un expander pour afficher les détails sur la différence de charge
            with st.expander("Voir les détails des changements de charge par secteur"):
                for secteur, diff in charge_difference.items():
                    if diff > 0:
                        st.write(f"Le secteur {secteur} a vu sa charge augmenter de {diff:.2f}% :green[🡅] :green[🡅] :green[🡅]")
                    else:
                        st.write(f"Le secteur {secteur} a vu sa charge diminuer de {abs(diff):.2f}% :red[🡇] :red[🡇] :red[🡇]")

        else:
            st.warning("Aucun magasin n'a été sélectionné pour réaffectation.")

    if "modified_map" in st.session_state and st.button("Enregistrer les modifications", key="save_button"):
        st.session_state["final_map"] = st.session_state["modified_map"]

        # Initialiser stores_optimized dans st.session_state si non initialisé
        if 'stores_optimized' not in st.session_state:
            st.session_state['stores_optimized'] = stores.copy()
            st.session_state['stores_optimized'] = st.session_state['stores_optimized'][['Code_mag', 'Code_client_PDV', 'Code_secteur', 'Frequence', 'Enseigne_GMS', 'Nom_mag', 'Potentiel', 'Code_postal', 'Adresse', 'Commune', 'Pays', 'lat', 'long']]
            st.session_state['stores_optimized'] = st.session_state['stores_optimized'].rename(columns={'Code_secteur': 'Nv. Secteur affecté'})
            st.session_state['stores_optimized'] = st.session_state['stores_optimized'].rename(columns={'Frequence': 'Nbr visites nécessaires'})
        else:
            stores_optimized = st.session_state['stores_optimized']

        # Récupérer les secteurs après réaffectation depuis st.session_state
        sectors_after = st.session_state['sectors_after']
        sector_after = st.session_state['sector_after']

        # Mise à jour de la DataFrame "Données détaillées après optimisation"
        for secteur in sectors_after:
            st.session_state.managers_optimized.loc[st.session_state.managers_optimized['Code_secteur'] == secteur, 
                                                    ['New_PDV affectés', 'New_Visites nécessaires', 'New_Charge']] = sector_after.loc[sector_after['Code_secteur'] == secteur, 
                                                                                                                                        ['PDV affectés', 'Visites nécessaires', 'New_Charge']].values
        # Mise à jour de la DataFrame principale "stores" et "stores_optimized" dans st.session_state
        for index in selected_stores:
            # Mettre à jour stores et stores_optimized dans session_state
            st.session_state['stores_optimized'].loc[
                st.session_state['stores_optimized']['Code_mag'] == stores.loc[index, 'Code_mag'],
                'Nv. Secteur affecté'
            ] = new_sector
            
            # Mettre à jour la DataFrame principale stores
            stores.loc[stores['Code_mag'] == stores.loc[index, 'Code_mag'], 'Code_secteur'] = new_sector

        # Appliquer la fonction format_charge pour formater les valeurs de New_Charge
        st.session_state.managers_optimized['New_Charge'] = st.session_state.managers_optimized['New_Charge'].apply(format_charge)
        

        st.success("Modifications enregistrées et appliquées à la carte principale !")
        st.success("Données des magasins après l'optimisation mises à jour avec succès !")
        st.success("Données détaillées générales après optimisation mises à jour avec succès !")
        
        # Afficher les données mises à jour
        st.subheader("Données des magasins après l'optimisation ( Copie )")
        st.dataframe(st.session_state['stores_optimized'])

        st.subheader("Données détaillées générales après l'optimisation ( Copie )")
        detailed_data = st.empty()
        optimized_columns_to_display = ['Code_secteur', 'Nom', 'Prenom', 'Adresse', 'New_PDV affectés', 'New_Visites nécessaires', 'New_Charge']
        # Re-afficher les données détaillées après optimisation mises à jour
        styled_optimized_managers = st.session_state.managers_optimized[optimized_columns_to_display].style.applymap(color_charge, subset=['New_Charge'])
        detailed_data.dataframe(styled_optimized_managers)
        

with col2:

    if 'managers_optimized' in st.session_state and st.session_state.managers_optimized is not None:

        # Afficher les données des magasins après optimisation
        st.subheader("Données des magasins après l'optimisation")
        stores_optimized = stores.copy()
        stores_optimized = stores_optimized[['Code_mag', 'Code_client_PDV', 'Code_secteur', 'Frequence', 'Enseigne_GMS', 'Nom_mag', 'Potentiel', 'Code_postal', 'Adresse','Commune', 'Pays', 'lat', 'long']]
        stores_optimized = stores_optimized.rename(columns={'Code_secteur': 'Nv. Secteur affecté'})
        stores_optimized = stores_optimized.rename(columns={'Frequence': 'Nbr visites nécessaires'})
        st.dataframe(stores_optimized)

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

            return merged[['Code_secteur', 'New_Charge']]

        # 🧮 Calcul PDV & Visites nécessaires
        visites_pdv_df = stores.groupby('Code_secteur').agg(
            **{
                'New_PDV affectés': ('Code_mag', 'count'),
                'New_Visites nécessaires': ('Frequence', 'sum')
            }
        ).reset_index()

        # 🔁 Merge complet
        new_charge_df = calculate_charge_from_optimized(stores, st.session_state.managers_optimized)
        new_charge_df['New_Charge'] = new_charge_df['New_Charge'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")

        result_df = pd.merge(
            st.session_state.managers_optimized.drop(columns=['New_Charge', 'New_PDV affectés', 'New_Visites nécessaires'], errors='ignore'),
            new_charge_df,
            on='Code_secteur', how='left'
        )
        result_df = pd.merge(result_df, visites_pdv_df, on='Code_secteur', how='left')

        # 🚀 Enregistrement final
        st.session_state.managers_optimized = result_df.copy()

        # 👁️‍🗨️ Affichage
        st.subheader("Données détaillées générales après l'optimisation")
        detailed_data = st.empty()
        optimized_columns_to_display = ['Code_secteur', 'Nom', 'Prenom', 'Adresse',
                                        'New_PDV affectés', 'New_Visites nécessaires', 'New_Charge']
        styled_optimized_managers = st.session_state.managers_optimized[
            optimized_columns_to_display
        ].style.applymap(color_charge, subset=['New_Charge'])
        detailed_data.dataframe(styled_optimized_managers)

        # Ajouter un filtre (selectbox) pour choisir quel graphique afficher
        graph_choice = st.selectbox(
            "Choisissez le graphique à afficher :",
            ("Répartition des visites par secteur optimisé", "Charge par secteur optimisé")
        )
        # Vérifier si 'stores_optimized' est dans session_state
        if 'stores_optimized' not in st.session_state:
            st.session_state.stores_optimized = stores_optimized

        if graph_choice == "Répartition des visites par secteur optimisé":
            st.subheader("Graphique de la répartition optimisée des visites par secteur")

            visits_by_sector_optimized = pd.DataFrame({'Code_secteur': st.session_state.stores_optimized['Nv. Secteur affecté'].unique()})
            visits_by_sector_optimized = visits_by_sector_optimized.merge(
                st.session_state.stores_optimized.groupby('Nv. Secteur affecté')['Nbr visites nécessaires'].sum().reset_index(),
                left_on='Code_secteur',
                right_on='Nv. Secteur affecté',
                how='left'
            ).fillna(0)

            visits_by_sector_optimized['Nbr visites nécessaires'] = visits_by_sector_optimized['Nbr visites nécessaires'].astype(int)

            fig = px.bar(
                visits_by_sector_optimized,
                x='Code_secteur',
                y='Nbr visites nécessaires',
                labels={'Code_secteur': 'Code secteur', 'Nbr visites nécessaires': 'Visites nécessaires'},
                title='Répartition des Visites par Secteur (Optimisé)',
                color='Nbr visites nécessaires',
                color_continuous_scale='Blues'
            )

            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        elif graph_choice == "Charge par secteur optimisé":
            st.subheader("Graphique de la charge optimisée par secteur")

            new_charge_per_sector = calculate_new_charge(stores, managers)
            charge_per_sector_optimized = new_charge_per_sector.set_index('Code_secteur').reindex(managers['Code_secteur']).reset_index()

            import plotly.graph_objects as go

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=charge_per_sector_optimized['Code_secteur'],
                y=charge_per_sector_optimized['New_Charge'],
                marker_color=['red' if val > 100 else 'green' for val in charge_per_sector_optimized['New_Charge']],
                name="Charge (%)"
            ))

            fig.add_hline(y=100, line_dash="dash", line_color="black")

            fig.update_layout(
                title='Charge par Secteur (Optimisé)',
                xaxis_title='Code secteur',
                yaxis_title='Charge (%)',
                xaxis_tickangle=-45
            )

            st.plotly_chart(fig, use_container_width=True)
