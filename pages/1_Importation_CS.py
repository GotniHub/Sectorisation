import streamlit as st
import pandas as pd
from db_connection import get_connection
from db_connection import clear_table_RH
from db_connection import check_table_empty  


def load_managers_data(uploaded_file):
    """Charge les données des managers à partir d'un fichier Excel avec choix de la feuille."""
    if uploaded_file is not None:
        # Lire le fichier sans spécifier de feuille pour obtenir la liste des feuilles disponibles
        xls = pd.ExcelFile(uploaded_file)
        sheets = xls.sheet_names

        # Afficher un sélecteur de feuille
        sheet_name = st.selectbox("📑 Choisissez une feuille :", sheets)

        # Charger la feuille sélectionnée
        data = pd.read_excel(xls, sheet_name=sheet_name)
                # ✅ Supprimer la première ligne si elle est identique aux noms de colonnes
        if data.iloc[0].tolist() == list(data.columns):
            data = data.iloc[1:].reset_index(drop=True)
        
        return data
    else:
        return None


def save_to_database(managers):
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
                INSERT INTO rh (
                    Code_secteur, Nom, Prenom, Adresse, Code_postal, Ville, Pays, 
                    Nb_heure_par_jour, Nb_jour_terrain_par_an, Nb_nuitees_max_par_an, 
                    Nb_jour_par_semaine, Coef_vitesse, Tolerance,
                    Latitude, Longitude, Nb_visite_max_par_an, Code_DR, Code_DZ, 
                    Type, Temps_max_retour, Cout_KM, Cout_fixe, Cout_Nuitees
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Conversion des données en liste de tuples
            data_to_insert = [
                (
                    None if pd.isna(row.get('Code secteur', None)) else row.get('Code secteur', None),
                    None if pd.isna(row.get('Nom', None)) else row.get('Nom', None),
                    None if pd.isna(row.get('Prenom', None)) else row.get('Prenom', None),
                    None if pd.isna(row.get('Adresse', None)) else row.get('Adresse', None),
                    None if pd.isna(row.get('Code postal', None)) else row.get('Code postal', None),
                    None if pd.isna(row.get('Ville', None)) else row.get('Ville', None),
                    None if pd.isna(row.get('Pays', None)) else row.get('Pays', None),
                    None if pd.isna(row.get('Nb heure par jour', None)) else row.get('Nb heure par jour', None),
                    None if pd.isna(row.get('Nb jour terrain par an', None)) else row.get('Nb jour terrain par an', None),
                    None if pd.isna(row.get('Nb nuitées max par an', None)) else row.get('Nb nuitées max par an', None),
                    None if pd.isna(row.get('Nb jour par semaine', None)) else row.get('Nb jour par semaine', None),
                    None if pd.isna(row.get('Coef vitesse', None)) else row.get('Coef vitesse', None),
                    None if pd.isna(row.get('Tolérance', None)) else row.get('Tolérance', None),
                    None if pd.isna(row.get('Latitude', None)) else row.get('Latitude', None),
                    None if pd.isna(row.get('Longitude', None)) else row.get('Longitude', None),
                    None if pd.isna(row.get('Nb visite max par an', None)) else row.get('Nb visite max par an', None),
                    None if pd.isna(row.get('Code DR', None)) else row.get('Code DR', None),
                    None if pd.isna(row.get('Code DZ', None)) else row.get('Code DZ', None),
                    None if pd.isna(row.get('Type', None)) else row.get('Type', None),
                    None if pd.isna(row.get('Temps max retour', None)) else row.get('Temps max retour', None),
                    None if pd.isna(row.get('Cout KM', None)) else row.get('Cout KM', None),
                    None if pd.isna(row.get('Cout fixe', None)) else row.get('Cout fixe', None),
                    None if pd.isna(row.get('Cout Nuités', None)) else row.get('Cout Nuités', None)
                )
                for _, row in managers.iterrows()
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
    st.title("Charger les données des Chefs de Secteurs")
    # Vérifie la connexion et affiche un message
    conn = get_connection()
    if conn:
        st.success("✅🔌Connexion à la base de données réussie !")
        conn.close()  # ok ici, c'était juste pour test
        # Vérifier si la table contient déjà des données
    row_count = check_table_empty()
    if row_count is not None:
        if row_count > 0:
            st.warning(f"⚠️ La table contient déjà {row_count} enregistrements.")
        else:
            st.warning("⚠️ La table est vide. Vous pouvez importer de nouvelles données.")
    # Bouton pour télécharger le template
    with open("CS_Data_Template.xlsx", "rb") as template_file:
        template_data = template_file.read()
    
    st.download_button(
        label="📥Télécharger le modèle de données (Template CS)",
        data=template_data,
        file_name="CS_Data_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    if st.button("🗑️ Vider la table RH"):
        clear_table_RH()
    # Colonnes attendues
    expected_columns = ['Nom', 'Prenom', 'Code secteur', 'Adresse', 'Code postal', 
                        'Ville', 'Pays', 'Nb heure par jour', 'Nb jour terrain par an',
                        'Nb nuitées max par an','Nb jour par semaine', 'Coef vitesse','Tolérance', 'Latitude', 'Longitude', 
                        'Nb visite max par an', 'Code DR', 'Code DZ', 'Type',
                        'Temps max retour', 'Cout KM', 'Cout fixe', 'Cout Nuités']

    # Téléchargement du fichier Excel
    uploaded_managers = st.file_uploader("📂 Importer le fichier Excel pour les chefs de secteurs (CS Data)", type=["xlsx"])
    
    if uploaded_managers is not None:
        managers = load_managers_data(uploaded_managers)
        
        if managers is not None:
            st.write("### Vérification des colonnes importées")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Colonnes importées**")
                imported_columns = managers.columns.tolist()
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

                if not managers.empty:
                    # Transformer les données en DataFrame avant insertion
                    df_to_insert = pd.DataFrame(managers)

                    # Afficher un aperçu du tableau
                    st.write("📌 Aperçu des données à insérer :")
                    st.dataframe(df_to_insert)

                if st.button("Valider et stocker les données"):
                    save_to_database(managers)
                    # Vérifier la table après l'insertion
                    row_count = check_table_empty()
                    if row_count is not None:
                        st.info(f"ℹ️ La table contient maintenant {row_count} enregistrements.")
                            # 🔄 Recharger les données depuis la base pour les afficher
                    conn = get_connection()
                    if conn:
                        df_rh = pd.read_sql("SELECT * FROM rh", conn)
                        conn.close()

                        st.success("✅ Voici les données désormais stockées dans la base RH :")
                        st.dataframe(df_rh)
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
