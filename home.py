import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Application de Sectorisation Commerciale",
    page_icon="📊",
    layout="wide"
)
st.logo("LOGO.png", icon_image="Logom.png")

# Titre de la page d'accueil
st.title("Bienvenue dans l'Application de Sectorisation Commerciale 👋")
left_column, right_column = st.columns([2, 1])
with left_column: 

    st.subheader("Optimisez vos ressources et améliorez l'efficacité commerciale")

    # Introduction au contexte
    st.write("""
    L'application de sectorisation commerciale est conçue pour aider les entreprises à optimiser leurs opérations de terrain 
    en attribuant efficacement des zones géographiques aux managers et aux équipes commerciales. Grâce à une analyse détaillée 
    des données, l'application permet de **maximiser les visites client**, **réduire les coûts de déplacement**, et **équilibrer les charges de travail**.
    """)

    # Principales fonctionnalités de l'application
    st.markdown("""
    ### Fonctionnalités principales
    - 🚀 **Chargement des Données** : Téléchargez le modèle pour vérifier et importer facilement les données des chefs de secteurs et des points de ventes à partir de fichiers Excel.
                
    - 🎯 **Optimisation des Déplacement** : Utilisez des algorithmes d'optimisation pour assigner efficacement les secteurs aux chefs de secteurs.
                
    - 📊 **Analyse Sectorielle** : Visualisez et analysez la distribution des points de ventes et des visites par secteur et leurs charges.
                
    - 🌍 **Interface Interactive** : Utilisez une carte interactive pour visualiser les affectations des secteurs et ajuster les paramètres.
    """)
with right_column:
    st.image("Secteur2.jpg", width=600)

# Instructions initiales pour l'utilisateur
st.info("Pour commencer, veuillez charger les données des chefs de secteurs ( Importation CS ) et des points de ventes ( Importation PDV ) en utilisant la barre latérale.")

# Image ou diagramme illustratif
st.image("LOGO.png", use_column_width=True)

# Footer ou note supplémentaire
st.write("---")
st.write("Cette application utilise des techniques avancées d'analyse géospatiale et de clustering pour améliorer la performance commerciale.")

