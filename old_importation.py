import streamlit as st
import pandas as pd

def load_stores_data(uploaded_file):
    """Charge les données des magasins à partir d'un fichier Excel."""
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        return data
    else:
        return None

def main():
    st.title("Charger les données des Magasins (PDV)")

    # Bouton pour télécharger le template
    with open("PDV_Data_Template.xlsx", "rb") as template_file:
        template_data = template_file.read()
    
    st.download_button(
        label="📥Télécharger le modèle de données (Template PDV)",
        data=template_data,
        file_name="PDV_Data_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Template de colonnes attendu pour PDV
    expected_columns = ['Code mag', 'Code client PDV', 'Code secteur', 'Temps',
                         'Frequence', 'long', 'lat', 'Enseigne GMS', 'Nom mag',
                        'Format Magasin', 'Potentiel', 'Groupe', 'CP', 'Adresse', 
                        'Commune', 'Pays', 'Enseigne GMS Regroupées', 'Surface']

    # Téléchargement du fichier Excel pour les magasins
    uploaded_stores = st.file_uploader("📂 Importer le fichier Excel pour les magasins (PDV Data)", type=["xlsx"])
    
    if uploaded_stores is not None:
        stores = load_stores_data(uploaded_stores)
        
        if stores is not None:
            # Affiche les colonnes importées à gauche et les colonnes de la template à droite
            st.write("### Vérification des colonnes importées")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Colonnes importées**")
                imported_columns = stores.columns.tolist()
                for col in imported_columns:
                    if col in expected_columns:
                        st.write(f"✔️ {col}")  # Icône de check si la colonne est dans la template
                    else:
                        st.write(f"❌ {col}")  # Icône de croix rouge si la colonne n'est pas dans la template
            with col2:
                st.write("**Colonnes attendues (template)**")
                for col in expected_columns:
                    if col in imported_columns:
                        st.write(f"✔️ {col}")  # Icône de check si la colonne est présente dans les données importées
                    else:
                        st.write(f"❌ {col}")  # Icône de croix rouge si la colonne est manquante dans les données importées

            # Vérifier si toutes les colonnes attendues sont présentes
            if set(expected_columns).issubset(set(imported_columns)):
                st.success("✅ Toutes les colonnes correspondent à la template!")
                
                if st.button("Valider et stocker les données"):
                    # Stocker les données dans session_state
                    st.session_state['stores_data'] = stores
                    st.success("✅ Données des magasins (PDV) chargées et stockées avec succès! Vous pouvez maintenant passer à l'application.")
            else:
                st.error("❌ Les colonnes importées ne correspondent pas à la template. Veuillez vérifier votre fichier.")
                missing_columns = set(expected_columns) - set(imported_columns)
                if missing_columns:
                    st.write("Colonnes manquantes :")
                    for col in missing_columns:
                        st.write(f"- {col}")
        else:
            st.error("Erreur lors du chargement des données.")
    else:
        st.info("Veuillez télécharger un fichier Excel pour afficher les données.")

if __name__ == "__main__":
    main()
