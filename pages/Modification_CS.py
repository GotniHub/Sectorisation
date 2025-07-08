import streamlit as st
import pandas as pd
from db_connection import get_connection, update_database, check_table_empty

def load_data_from_database():
    """Charge les données actuelles depuis la base de données MySQL (table RH)."""
    conn = get_connection()
    if conn is None:
        st.error("❌ Erreur de connexion à la base de données.")
        return pd.DataFrame()

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rh")
        columns = [col[0] for col in cursor.description]
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=columns)
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données RH : {e}")
        return pd.DataFrame()
    finally:
        cursor.close()
        conn.close()


def update_selected_rows(selected_rows):
    """Met à jour les lignes sélectionnées dans la base de données."""
    if selected_rows.empty:
        st.warning("⚠️ Aucune ligne sélectionnée pour modification.")
        return
    
    update_database(selected_rows)
    st.success("✅ Modifications enregistrées avec succès !")

def delete_rows_from_database(codes_secteurs):
    """Supprime les lignes sélectionnées via leurs codes secteurs."""
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        try:
            for code in codes_secteurs:
                cursor.execute("DELETE FROM rh WHERE Code_secteur = %s", (code,))
            conn.commit()
            st.success(f"🗑️ {len(codes_secteurs)} ligne(s) supprimée(s) avec succès.")
        except Exception as e:
            conn.rollback()
            st.error(f"❌ Erreur lors de la suppression : {e}")
        finally:
            cursor.close()
            conn.close()

def main():
    st.title("✏️ Modifier les données des Chefs de Secteurs")
        # Vérifie la connexion et affiche un message
    conn = get_connection()
    if conn:
        st.success("✅🔌Connexion à la base de données réussie !")
        conn.close()  # ok ici, c'était juste pour test
    # Initialisation des états
    # Initialiser les variables dans session_state si elles n'existent pas
    if "rows_to_delete" not in st.session_state:
        st.session_state.rows_to_delete = []

    if "show_confirm_delete" not in st.session_state:
        st.session_state.show_confirm_delete = False
        if "selected_rows" not in st.session_state:
            st.session_state.selected_rows = None

    # Vérifier si la table contient déjà des données
    row_count = check_table_empty()
    if row_count == 0:
        st.warning("⚠️ La table est vide. Aucune donnée à modifier.")
        return

    # 📌 Charger les données de la base
    df_existing = load_data_from_database()
    if df_existing is None:
        return
    
    # --- 🎛️ TOGGLE AFFICHAGE FILTRES ---
    with st.sidebar:
        st.markdown("## 🔍 Filtres")
        show_filters = st.toggle("Afficher les filtres", value=True)

    # Appliquer les filtres uniquement si activés
    if show_filters:
        with st.sidebar:
            noms = st.multiselect("Nom", sorted(df_existing["Nom"].dropna().unique()))
            secteurs = st.multiselect("Code secteur", sorted(df_existing["Code_secteur"].dropna().unique()))
            villes = st.multiselect("Ville", sorted(df_existing["Ville"].dropna().unique()))
            pays = st.multiselect("Pays", sorted(df_existing["Pays"].dropna().unique()))
    else:
        noms, secteurs, villes, pays = [], [], [], []

    # Appliquer les filtres
    filtered_df = df_existing.copy()
    if noms:
        filtered_df = filtered_df[filtered_df["Nom"].isin(noms)]
    if secteurs:
        filtered_df = filtered_df[filtered_df["Code_secteur"].isin(secteurs)]
    if villes:
        filtered_df = filtered_df[filtered_df["Ville"].isin(villes)]
    if pays:
        filtered_df = filtered_df[filtered_df["Pays"].isin(pays)]

    # ✅ Ajouter colonnes de sélection et suppression
    filtered_df.insert(0, "Sélection", False)

    st.subheader("🔹 Sélectionnez les lignes à modifier")
    edited_df = st.data_editor(
        filtered_df,
        column_config={
            "Sélection": st.column_config.CheckboxColumn()
        },
        key="edit_existing",
        use_container_width=True
    )

    selected_rows = edited_df[edited_df["Sélection"] == True].copy()

    if st.button("🛠️Modifier"):

        if selected_rows.empty:
            st.warning("⚠️ Aucune ligne sélectionnée.")
        else:
            selected_rows.drop(columns=["Sélection"], inplace=True)
            st.session_state.selected_rows = selected_rows
            st.rerun()  # Redémarrer pour afficher la zone d'édition

    # 1. Définir la fonction de dialogue (comme dans l'exemple)
    @st.dialog("Confirmation de suppression")  # Décorateur pour créer le dialogue
    def confirm_delete_dialog(rows_to_delete):
        st.error(f"⚠️ Êtes-vous sûr de vouloir supprimer {len(rows_to_delete)} ligne(s) ?")
        st.write("**Cette action est irréversible.**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirmer", type="primary"):
                delete_rows_from_database(rows_to_delete)  # Ta fonction existante
                st.session_state.delete_confirmed = True  # Marquer comme confirmé
                st.rerun()
        with col2:
            if st.button("❌ Annuler"):
                st.rerun()  # Ferme le dialogue sans action

    # 2. Logique principale 
    if st.button("🗑️ Supprimer"):
        if selected_rows.empty:
            st.warning("⚠️ Aucune ligne sélectionnée pour suppression.")
        else:
            # Stocker les IDs à supprimer et ouvrir le dialogue
            st.session_state.rows_to_delete = selected_rows["Code_secteur"].tolist()
            confirm_delete_dialog(st.session_state.rows_to_delete)  # Appel du dialogue

    # 3. Feedback après confirmation (comme dans l'exemple)
    if st.session_state.get("delete_confirmed"):
        st.success(f"✅ {len(st.session_state.rows_to_delete)} ligne(s) supprimée(s) !")
        del st.session_state.delete_confirmed  # Nettoyer l'état
        del st.session_state.rows_to_delete
            
                
    # ✅ Afficher en bas les lignes sélectionnées si elles existent
    if st.session_state.selected_rows is not None:
        st.divider()
        st.subheader("📝 Modifier les lignes sélectionnées")
        edited_selection = st.data_editor(
            st.session_state.selected_rows,
            num_rows="dynamic",
            key="edit_selected",
            use_container_width=True
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Enregistrer les modifications"):
                update_database(edited_selection)
                st.success("✅ Modifications enregistrées avec succès.")
                st.session_state.selected_rows = None

        with col2:
            if st.button("💢 Annuler la modification"):
                st.session_state.selected_rows = None
                st.info("🔄 Sélection annulée.")
                st.rerun()

if __name__ == "__main__":
    main()
