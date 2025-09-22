import psycopg2

try:
    conn = psycopg2.connect(
        host="localhost",       # dit moet 'localhost' blijven
        database="movies",   # vervang door de naam van jouw database
        user="postgres",        # jouw database user
        password="Idris123",    # jouw wachtwoord
        port="5432"             # PostgreSQL standaardpoort
    )

    print("✅ Verbinding gemaakt!")

    cur = conn.cursor()
    cur.execute("SELECT version();")
    print("Database versie:", cur.fetchone())

    cur.close()
    conn.close()
    print("✅ Verbinding gesloten.")

except Exception as e:
    print("❌ Fout:", e)
