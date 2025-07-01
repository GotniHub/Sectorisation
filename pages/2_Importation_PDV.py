import streamlit as st
import pandas as pd
from db_connection import get_connection
from db_connection import clear_table_PDV
from db_connection import check_table_empty  


def load_stores_data(uploaded_file):
    """Charge les données des stores à partir d'un fichier Excel avec choix de la feuille."""
    if uploaded_file is not None:
        # Lire le fichier sans spécifier de feuille pour obtenir la liste des feuilles disponibles
        xls = pd.ExcelFile(uploaded_file)
        sheets = xls.sheet_names

        # Afficher un sélecteur de feuille
        sheet_name = st.selectbox("📑 Choisissez une feuille :", sheets)

        # Charger la feuille sélectionnée
        data = pd.read_excel(xls, sheet_name=sheet_name)
        return data
    else:
        return None


def save_to_database(stores):
    """Insère les données dans la base de données MySQL."""
    st.write("Appel de get_connection()...")  # DEBUG

    conn = get_connection()
    
    if conn.open == False:
        st.error("❌ La connexion MySQL a été fermée. Recharger la page et réessayer.")
        return
    if conn:
        cursor = conn.cursor()
        try:
            # Requête d'insertion SQL
            query = """
                INSERT INTO pdv (
                    Code_mag, Code_client_PDV, Code_secteur, Temps, Frequence, `long`, `lat`,
                    Enseigne_GMS, Nom_mag, Format_Magasin, Potentiel, Groupe, Code_postal,
                    Adresse, Commune, Pays, Enseigne_GMS_Regroupees, Surface
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            
            # Conversion des données en liste de tuples
            data_to_insert = [
                (
                    None if pd.isna(row.get('Code mag', None)) else row.get('Code mag', None),
                    None if pd.isna(row.get('Code client PDV', None)) else row.get('Code client PDV', None),
                    None if pd.isna(row.get('Code secteur', None)) else row.get('Code secteur', None),
                    None if pd.isna(row.get('Temps', None)) else row.get('Temps', None),
                    None if pd.isna(row.get('Frequence', None)) else row.get('Frequence', None),
                    None if pd.isna(row.get('long', None)) else row.get('long', None),
                    None if pd.isna(row.get('lat', None)) else row.get('lat', None),
                    None if pd.isna(row.get('Enseigne GMS', None)) else row.get('Enseigne GMS', None),
                    None if pd.isna(row.get('Nom mag', None)) else row.get('Nom mag', None),
                    None if pd.isna(row.get('Format Magasin', None)) else row.get('Format Magasin', None),
                    None if pd.isna(row.get('Potentiel', None)) else row.get('Potentiel', None),
                    None if pd.isna(row.get('Groupe', None)) else row.get('Groupe', None),
                    None if pd.isna(row.get('Code postal', None)) else row.get('Code postal', None),
                    None if pd.isna(row.get('Adresse', None)) else row.get('Adresse', None),
                    None if pd.isna(row.get('Commune', None)) else row.get('Commune', None),
                    None if pd.isna(row.get('Pays', None)) else row.get('Pays', None),
                    None if pd.isna(row.get('Enseigne GMS Regroupees', None)) else row.get('Enseigne GMS Regroupees', None),
                    None if pd.isna(row.get('Surface', None)) else row.get('Surface', None)
                )
                for _, row in stores.iterrows()
            ]


            if not data_to_insert:
                st.warning("⚠️ Aucune donnée à insérer !")
                return
            
            # Exécuter l'insertion
            cursor.executemany(query, data_to_insert)
            conn.commit()
            st.success("✅ Données insérées avec succès dans la base de données !")

        except Exception as e:
            conn.rollback()
            st.error(f"❌ Erreur lors de l'insertion des données : {e}")

        finally:
            if cursor:
                cursor.close()
            if conn and conn.open:  # Vérifie si la connexion est encore ouverte avant de la fermer
                conn.close()
                print("✅ Connexion fermée avec succès.")  # DEBUG


def main():
    st.title("Charger les données des Magasins (PDV)")
    # Vérifie la connexion et affiche un message
    conn = get_connection()
    if conn:
        st.success("✅🔌Connexion à la base de données réussie !")
        conn.close()  # ok ici, c'était juste pour test
        # Vérifier si la table contient déjà des données
    row_count = check_table_empty("pdv")
    if row_count is not None:
        if row_count > 0:
            st.warning(f"⚠️ La table contient déjà {row_count} enregistrements.")
        else:
            st.warning("⚠️ La table est vide. Vous pouvez importer de nouvelles données.")
    # Bouton pour télécharger le template
    with open("PDV_Data_Template.xlsx", "rb") as template_file:
        template_data = template_file.read()
    
    st.download_button(
        label="📥Télécharger le modèle de données (Template PDV)",
        data=template_data,
        file_name="PDV_Data_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    if st.button("🗑️ Vider la table PDV"):
        clear_table_PDV()
    # Template de colonnes attendu pour PDV
    expected_columns = ['Code mag', 'Code client PDV', 'Code secteur', 'Temps',
                         'Frequence', 'long', 'lat', 'Enseigne GMS', 'Nom mag',
                        'Format Magasin', 'Potentiel', 'Groupe', 'CP', 'Adresse', 
                        'Commune', 'Pays', 'Enseigne GMS Regroupées', 'Surface']

    # Téléchargement du fichier Excel
    uploaded_stores = st.file_uploader("📂 Importer le fichier Excel pour les magasins (PDV Data)", type=["xlsx"])
    
    if uploaded_stores is not None:
        stores = load_stores_data(uploaded_stores)
        
        if stores is not None:
            st.write("### Vérification des colonnes importées")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Colonnes importées**")
                imported_columns = stores.columns.tolist()
                for col in imported_columns:
                    if col in expected_columns:
                        st.write(f"✔️ {col}")  
                    else:
                        st.write(f"❌ {col}")  

            with col2:
                st.write("**Colonnes attendues (template)**")
                for col in expected_columns:
                    if col in imported_columns:
                        st.write(f"✔️ {col}")  
                    else:
                        st.write(f"❌ {col}")  

            # Vérifier si toutes les colonnes attendues sont présentes
            if set(expected_columns).issubset(set(imported_columns)):
                st.success("✅ Toutes les colonnes correspondent à la template!")

                if not stores.empty:
                    # Transformer les données en DataFrame avant insertion
                    df_to_insert = pd.DataFrame(stores)

                    # Afficher un aperçu du tableau
                    st.write("📌 Aperçu des données à insérer :")
                    st.dataframe(df_to_insert)

                if st.button("Valider et stocker les données"):
                    save_to_database(stores)
                    # Vérifier la table après l'insertion
                    row_count = check_table_empty("pdv")
                    if row_count is not None:
                        st.info(f"ℹ️ La table contient maintenant {row_count} enregistrements.")
            else:
                st.error("❌ Les colonnes importées ne correspondent pas à la template.")
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
