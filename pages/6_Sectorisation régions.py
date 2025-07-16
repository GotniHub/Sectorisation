import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
import streamlit as st
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster, Fullscreen
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint, Point, Polygon
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from io import BytesIO
from fpdf import FPDF, HTMLMixin
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import tempfile
import os
from folium.plugins import Draw
import matplotlib.colors as mcolors

st.set_page_config(page_title="Analyse Sectorielle", layout="wide")

# Charger le fichier Excel
file_path = 'Calibrage France Direct Test (1).xlsx'  # Utilisation du nouveau fichier
data = pd.read_excel(file_path)

def anonymiser_colonne(df, colonne, prefix):
    mapping = {val: f"{prefix}_{i+1}" for i, val in enumerate(df[colonne].dropna().unique())}
    df[colonne] = df[colonne].map(mapping)
    return df, mapping

# Liste des colonnes à anonymiser avec leur préfixe
colonnes_a_anonymiser = {
    'Directeur_commercial': 'DC',
    'Directeur_des_ventes': 'DV',
    'Chef_de_zone': 'CZ',
    'Chef_de_marché': 'CM',
    'Commercial': 'CO'
}

mappings = {}  # Pour stocker les correspondances si besoin
for col, prefix in colonnes_a_anonymiser.items():
    if col in data.columns:
        data, map_col = anonymiser_colonne(data, col, prefix)
        mappings[col] = map_col  # Optionnel si tu veux garder trace

# Ajouter des coordonnées géographiques synthétiques pour illustration si elles ne sont pas présentes
if 'lat' not in data.columns or 'long' not in data.columns:
    np.random.seed(0)
    data['lat'] = np.random.uniform(40, 50, len(data))
    data['long'] = np.random.uniform(-5, 10, len(data))

# Fonction pour ajouter des marges autour des points
def add_margin(points, margin=0.01):
    points = np.array(points)
    center = points.mean(axis=0)
    new_points = center + (points - center) * (1 + margin)
    return new_points.tolist()

# Fonction pour créer des polygones convexes robustes
def create_convex_hull(points):
    multi_point = MultiPoint(points)
    convex_hull = multi_point.convex_hull
    return list(convex_hull.exterior.coords)

# Convertir les couleurs en format hexadécimal
def convert_colors_to_hex(colors):
    return [mcolors.rgb2hex(color) for color in colors]

# Créer une carte centrée sur la France
def create_map(selected_regions, data, colors):
    map_france = folium.Map(location=[46.603354, 1.888334], zoom_start=6)
    marker_cluster = MarkerCluster().add_to(map_france)

    for _, row in data[data['region'].isin(selected_regions)].iterrows():
        folium.Marker(
            location=[row['lat'], row['long']],
            popup=row['Nom du client'],
            icon=folium.Icon(color=colors[int(row['region'])])
        ).add_to(marker_cluster)

    drawn_polygons = []

    for region in selected_regions:
        subset = data[data['region'] == region]
        if len(subset) > 2:
            points = list(zip(subset['lat'], subset['long']))
            points_with_margin = add_margin(points)
            hull_points = create_convex_hull(points_with_margin)
            polygon = folium.Polygon(
                locations=hull_points,
                color=colors[region],
                fill=True,
                fill_color=colors[region],
                fill_opacity=0.2
            )
            polygon.add_to(map_france)
            drawn_polygons.append(polygon)

            # Ajouter un marqueur central avec une bulle d'information et une icône personnalisée
            icon_url = 'info.png'  # Assurez-vous que ce chemin est correct
            custom_icon = folium.CustomIcon(icon_image=icon_url, icon_size=(30, 30))

            # Ajouter un marqueur central avec une bulle d'information
            folium.Marker(
                location=np.mean(points_with_margin, axis=0).tolist(),
                popup=folium.Popup(f"""
                    <b>Région {region + 1}</b><br>
                    Magasins: {int(subset['Nom du client'].count())}<br>
                    CA: {subset['CA 2023'].sum():,.2f} €<br>
                    Nb Visites: {subset['Nb Visite'].sum()}
                """, max_width=300),
                icon=custom_icon
            ).add_to(map_france)

    # Ajouter les outils de dessin
    draw = Draw(
        draw_options={
            "polyline": False,
            "rectangle": True,
            "circle": False,
            "polygon": True,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={"edit": {"featureGroup": "drawn_polygons"}, "remove": True},
    )
    map_france.add_child(draw)

    Fullscreen(position='topright', force_separate_button=True).add_to(map_france)
    return map_france

def save_map_as_image(map_object, file_path='map.png'):
    temp_html = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    map_object.save(temp_html.name)
    temp_html.close()

    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(f'file://{temp_html.name}')
    driver.save_screenshot(file_path)
    driver.quit()
    
    os.remove(temp_html.name)


st.title('Sectorisation des Magasins en Régions')
# Disposition des colonnes
col1, col2 = st.columns([2,1])

# Ajouter une entrée pour sélectionner le nombre de régions
num_regions = st.sidebar.number_input("Choisir le nombre de régions", min_value=2, max_value=40, value=5)

# Clustering en utilisant KMeans avec le nombre de régions sélectionné
kmeans = KMeans(n_clusters=num_regions, random_state=0)
data['region'] = kmeans.fit_predict(data[['lat', 'long']])

# Couleurs pour les régions
colors = plt.cm.get_cmap('tab10', num_regions).colors
hex_colors = convert_colors_to_hex(colors)

# Calculer le CA total, le nombre de magasins et les visites pour chaque région
region_info = data.groupby('region').agg({
    'CA 2023': 'sum',
    'Nom du client': 'count',
    'Nb Visite': 'sum'
}).reset_index().rename(columns={'CA 2023': 'Chiffre d\'Affaires Total', 'Nom du client': 'Nombre de Magasins', 'Nb Visite': 'Nombre de Visites'})

with col1:
    # Afficher les informations de chaque région
    with st.expander("Afficher & Comparer les résultats de chaque Région"):
        for index, row in region_info.iterrows():
            st.subheader(f'Région {int(row["region"]) + 1}')
            st.write(f"Nombre de Magasins : {int(row['Nombre de Magasins'])}")
            st.write(f"Chiffre d'Affaires Total : {row['Chiffre d\'Affaires Total']:,.2f} €")
            st.write(f"Nombre de Visites : {row['Nombre de Visites']}")

    # Ajouter le filtre multi-select
    selected_regions = st.multiselect(
        'Sélectionnez les régions à afficher',
        options=list(range(num_regions)),
        format_func=lambda x: f'Région {x + 1}',
        default=list(range(num_regions))
    )

    # Afficher les informations des régions sélectionnées
    selected_info = region_info[region_info['region'].isin(selected_regions)]
    total_magasins = selected_info['Nombre de Magasins'].sum()
    total_ca = selected_info['Chiffre d\'Affaires Total'].sum()
    total_visits = selected_info['Nombre de Visites'].sum()

    st.subheader('Total pour les régions sélectionnées')
    st.write(f"Nombre total de Magasins : {total_magasins}")
    st.write(f"Chiffre d'Affaires Total : {total_ca:,.2f} €")
    st.write(f"Nombre total de Visites : {total_visits}")

    # Créer et afficher la carte pour les régions sélectionnées
    map_france = create_map(selected_regions, data, hex_colors)
    folium_static(map_france)

    # Ajouter une fonction pour calculer le nombre de magasins et le chiffre d'affaires dans une zone dessinée
    def calculate_polygon_stats(polygon_coords):
        polygon = Polygon(polygon_coords)
        selected_data = data[data.apply(lambda row: polygon.contains(Point(row['long'], row['lat'])), axis=1)]
        total_stores = selected_data['Nom du client'].count()
        total_revenue = selected_data['CA 2023'].sum()
        total_visits = selected_data['Nb Visite'].sum()
        return total_stores, total_revenue, total_visits, selected_data

    # Liste pour empiler les résultats des statistiques des polygones
    if 'polygon_stats' not in st.session_state:
        st.session_state.polygon_stats = []

    # Ajouter une zone de texte pour entrer les coordonnées du polygone
    polygon_coords = st.text_area("Entrez les coordonnées du polygone (format: [[long, lat], [long, lat], ...])", value="[]")

    if st.button("Calculer les statistiques pour le polygone"):
        try:
            polygon_coords = eval(polygon_coords)
            total_stores, total_revenue, total_visits, selected_data = calculate_polygon_stats(polygon_coords)
            st.session_state.polygon_stats.append({
                'coords': polygon_coords,
                'total_stores': total_stores,
                                'total_revenue': total_revenue,
                'total_visits': total_visits,
                'selected_data': selected_data
            })
        except Exception as e:
            st.error(f"Erreur dans les coordonnées du polygone : {e}")

    # Afficher tous les résultats des polygones empilés
    for i, stats in enumerate(st.session_state.polygon_stats):
        st.write(f"Polygone {i+1}:")
        st.write(f"Nombre de magasins dans le polygone : {stats['total_stores']}")
        st.write(f"Chiffre d'affaires total dans le polygone : {stats['total_revenue']:,.2f} €")
        st.write(f"Nombre de visites dans le polygone : {stats['total_visits']}")
        output = BytesIO()
        stats['selected_data'].to_excel(output, index=False)
        output.seek(0)
        st.download_button(
            label=f"Exporter les données du polygone {i+1} en Excel",
            data=output,
            file_name=f"polygone_{i+1}_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

with col2:
    def create_bar_chart():
        fig, ax = plt.subplots(figsize=(10, 7))  # Ajuster la taille ici
        bar_width = 0.4
        r1 = np.arange(len(selected_info))
        r2 = [x + bar_width for x in r1]

        ax.bar(r1, selected_info['Nombre de Magasins'], color='skyblue', width=bar_width, label='Nombre de Magasins')
        ax.set_xlabel('Région', fontsize=16)
        ax.set_ylabel('Nombre de Magasins', color='skyblue', fontsize=16)

        ax2 = ax.twinx()
        ax2.bar(r2, selected_info['Chiffre d\'Affaires Total'], color='green', width=bar_width, label='Chiffre d\'Affaires Total')
        ax2.set_ylabel('Chiffre d\'Affaires Total (€)', color='green', fontsize=16)

        # Modifier la taille des ticks
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)

        ax.set_xticks([r + bar_width / 2 for r in range(len(selected_info))])
        ax.set_xticklabels([f'Région {int(x) + 1}' for x in selected_info['region']], fontsize=14)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x * 1e-6:.1f}M'))

        # Ajuster les marges pour éviter le chevauchement
        plt.subplots_adjust(top=0.85)

        # Ajouter un titre au graphique
        fig.suptitle('Comparaison du Nombre de Magasins & Chiffre d\'Affaires Total par Région', fontsize=20)
        # Ajuster les marges pour éviter le chevauchement
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        return fig

    def create_pie_chart():
        # Personnaliser les couleurs des régions
        custom_colors = convert_colors_to_hex(plt.cm.get_cmap('tab10', num_regions).colors)
        # Créer le pie chart pour le chiffre d'affaires
        fig, ax = plt.subplots(figsize=(10, 6))  # Ajuster la taille ici
        ax.pie(
            region_info['Chiffre d\'Affaires Total'],
            colors=custom_colors,
            autopct='%1.1f%%',
            startangle=140
        )
        ax.axis('equal')  # Assure que le pie chart est dessiné en cercle.
        ax.legend([f'Région {int(x) + 1}' for x in region_info['region']], loc="best")
        plt.title("Répartition du Chiffre d'Affaires par Région", fontsize=20)
        return fig
    
    class PDF(FPDF, HTMLMixin):
        pass

    def generate_pdf():
        # Créer les graphiques et les enregistrer en fichiers temporaires
        bar_chart_fig = create_bar_chart()
        pie_chart_fig = create_pie_chart()

        bar_chart_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        pie_chart_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')

        bar_chart_fig.savefig(bar_chart_file.name, format='png', bbox_inches='tight')
        pie_chart_fig.savefig(pie_chart_file.name, format='png', bbox_inches='tight')

        # Convertir la carte en image
        map_france = create_map(selected_regions, data, hex_colors)
        map_file_path = 'map.png'
        save_map_as_image(map_france, file_path=map_file_path)

        # Créer le PDF
        pdf = PDF()
        pdf.add_page()

        # Ajouter le titre
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Rapport d'Analyse Sectorielle", ln=True, align='C')

        # Ajouter le détail Total pour les régions sélectionnées
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Nombre total de Magasins : {total_magasins}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Chiffre d'Affaires Total : {total_ca:,.2f} €".encode('latin-1', 'replace').decode('latin-1'), ln=True, align='L')
        pdf.cell(200, 10, txt=f"Nombre total de Visites : {total_visits}", ln=True, align='L')

        # Ajouter la carte
        pdf.cell(200, 10, txt="Carte des Régions Sélectionnées", ln=True, align='L')
        pdf.image(map_file_path, x=10, y=None, w=190)

        # Ajouter les résultats de chaque région
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Les résultats de chaque Région", ln=True, align='L')
        for index, row in region_info.iterrows():
            pdf.cell(200, 10, txt=f"Région {int(row['region']) + 1}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Nombre de Magasins : {int(row['Nombre de Magasins'])}", ln=True, align='L')
            pdf.cell(200, 10, txt=f"Chiffre d'Affaires Total : {row['Chiffre d\'Affaires Total']:,.2f} €".encode('latin-1', 'replace').decode('latin-1'), ln=True, align='L')
            pdf.cell(200, 10, txt=f"Nombre de Visites : {row['Nombre de Visites']}", ln=True, align='L')
            pdf.ln(5)  # Ajouter un espace entre les régions

        pdf.add_page()
        # Ajouter le graphique à barres
        pdf.cell(200, 10, txt="Graphique : Comparaison du Nombre de Magasins et du Chiffre d'Affaires Total par Région", ln=True, align='L')
        pdf.image(bar_chart_file.name, x=10, y=None, w=190)

        # Ajouter le graphique circulaire
        pdf.cell(200, 10, txt="Graphique : Répartition du Chiffre d'Affaires par Région", ln=True, align='L')
        pdf.image(pie_chart_file.name, x=10, y=None, w=190)

        # Sauvegarder le PDF
        pdf_output = BytesIO()
        pdf_output.write(pdf.output(dest='S').encode('latin1'))
        pdf_output.seek(0)

        # Supprimer les fichiers temporaires
        bar_chart_file.close()
        pie_chart_file.close()
        os.remove(bar_chart_file.name)
        os.remove(pie_chart_file.name)

        return pdf_output
    

    # Afficher les graphiques dans Streamlit
    bar_chart_fig = create_bar_chart()
    st.pyplot(bar_chart_fig)

    pie_chart_fig = create_pie_chart()
    st.pyplot(pie_chart_fig)

    # Ajouter des boutons pour exporter les graphiques
    if st.button('Exporter le graphique à barres'):
        bar_chart_fig = create_bar_chart()
        bar_chart_fig.savefig('bar_chart.png', bbox_inches='tight')
        st.success('Le graphique à barres a été exporté avec succès !')
        with open('bar_chart.png', 'rb') as file:
            btn = st.download_button(
                label="Télécharger le graphique à barres",
                data=file,
                file_name="bar_chart.png",
                mime="image/png"
            )

    if st.button('Exporter le graphique circulaire'):
        pie_chart_fig.savefig('pie_chart.png')
        st.success('Le graphique circulaire a été exporté avec succès !')
        with open('pie_chart.png', 'rb') as file:
            btn = st.download_button(
                label="Télécharger le graphique circulaire",
                data=file,
                file_name="pie_chart.png",
                mime="image/png"
            )
    if st.button('Générer le rapport PDF'):
        pdf_data = generate_pdf()
        st.success('Le rapport PDF a été généré avec succès !')
        st.download_button(
            label="Télécharger le rapport PDF",
            data=pdf_data,
            file_name="rapport_analyse_sectorielle.pdf",
            mime="application/pdf"
        )


# Ajouter un expander pour afficher les détails des magasins par région
with st.expander("Afficher les détails des magasins par région"):
    for region in selected_regions:
        st.subheader(f'Détails des magasins pour la Région {region + 1}')
        region_data = data[data['region'] == region]
        st.dataframe(region_data)

# Ajouter un expander pour afficher le tableau global des magasins
with st.expander("Afficher le tableau global des magasins"):
    global_data = data.copy()
    global_data['Région'] = global_data['region'].apply(lambda x: f'Région {x + 1}')
    # Réorganiser les colonnes pour afficher 'Région' en premier
    cols = ['Région'] + [col for col in global_data.columns if col != 'Région' and col != 'region']
    st.dataframe(global_data[cols])

st.caption("ℹ️ Les noms des commerciaux ont été anonymisés pour garantir la confidentialité.")

