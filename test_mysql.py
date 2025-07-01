import mysql.connector

try:
    print("Tentative de connexion à MySQL...")
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # XAMPP n'a pas de mot de passe par défaut
        database="secto",
        port=3306
    )
    print("✅ Connexion réussie !")

    cursor = conn.cursor()
    cursor.execute("SHOW TABLES;")
    tables = cursor.fetchall()
    print("📌 Tables dans la base de données :")
    for table in tables:
        print(table)

    cursor.close()
    conn.close()
except mysql.connector.Error as err:
    print(f"❌ ERREUR MYSQL : {err}")
