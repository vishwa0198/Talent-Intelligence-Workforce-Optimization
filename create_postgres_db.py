import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import pandas as pd
import os
import sys
import dotenv

# --------------------------------------------------------
# CONFIG — READ FROM ENV / .env INSTEAD OF HARDCODING
# --------------------------------------------------------
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    # Safe to ignore if dotenv isn't installed; environment vars may still be set.
    pass

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "your_password")
DB_NAME = os.getenv("PG_DB_NAME", "talent_intelligence")

# Folders
PROCESSED_DIR = r"C:\Guvi\Talent Intelligence & Workforce Optimization\notebooks\processed"

HR_CSV = os.path.join(PROCESSED_DIR, "hr_clean.csv")
REVIEWS_CSV = os.path.join(PROCESSED_DIR, "employee_reviews_clean.csv")
PERF_CSV = os.path.join(PROCESSED_DIR, "performance_data.csv")

# --------------------------------------------------------
# FUNCTION: CHECK IF DATABASE EXISTS
# --------------------------------------------------------
def database_exists():
    try:
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, database="postgres"
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname=%s;", (DB_NAME,))
        exists = cur.fetchone()
        cur.close()
        conn.close()
        return bool(exists)
    except Exception as e:
        print("❌ Error checking DB:", e)
        return False

# --------------------------------------------------------
# FUNCTION: CREATE DATABASE
# --------------------------------------------------------
def create_database():
    try:
        conn = psycopg2.connect(
            host=PG_HOST, user=PG_USER, password=PG_PASSWORD, port=PG_PORT, database="postgres"
        )
        conn.autocommit = True
        cur = conn.cursor()

        print(f"🔹 Creating PostgreSQL database '{DB_NAME}'...")

        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))

        cur.close()
        conn.close()
        print(f"✔ Database '{DB_NAME}' created successfully!")

    except Exception as e:
        print(f"⚠ Warning: Could not create DB ({e}).")
        print("➡ You may already have permissions restricted or DB already exists.")

# --------------------------------------------------------
# CONNECT TO talent_intelligence DB
# --------------------------------------------------------
def connect_db():
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database=DB_NAME,
        )
        conn.autocommit = False
        return conn
    except Exception as e:
        print("❌ Failed to connect:", e)
        sys.exit()

# --------------------------------------------------------
# FUNCTION: CREATE TABLES
# --------------------------------------------------------
def create_tables(conn):
    cur = conn.cursor()
    print("\n🔹 Creating tables...")

    # Employees
    cur.execute("""
        DROP TABLE IF EXISTS employees CASCADE;
        CREATE TABLE employees (
            EmployeeID INT PRIMARY KEY,
            Name TEXT,
            Age INT,
            Gender TEXT,
            Department TEXT,
            Salary NUMERIC,
            Tenure NUMERIC
        );
    """)

    # Reviews
    cur.execute("""
        DROP TABLE IF EXISTS reviews CASCADE;
        CREATE TABLE reviews (
            ReviewID SERIAL PRIMARY KEY,
            EmployeeID INT REFERENCES employees(EmployeeID),
            ReviewText TEXT,
            Sentiment INT
        );
    """)

    # Performance table
    cur.execute("""
        DROP TABLE IF EXISTS performance CASCADE;
        CREATE TABLE performance (
            PerfID SERIAL PRIMARY KEY,
            EmployeeID INT REFERENCES employees(EmployeeID),
            PerformanceScore NUMERIC,
            Year INT
        );
    """)

    conn.commit()
    cur.close()
    print("✔ Tables created successfully!")

# --------------------------------------------------------
# FUNCTION: INSERT DATA USING execute_values (FAST)
# --------------------------------------------------------
def insert_data(conn, df, table):
    cur = conn.cursor()
    cols = ",".join(df.columns)
    values = [tuple(x) for x in df.to_numpy()]
    query = f"INSERT INTO {table} ({cols}) VALUES %s"

    execute_values(cur, query, values)
    conn.commit()
    cur.close()
    print(f"✔ Inserted {len(values)} rows into {table}")

# --------------------------------------------------------
# MAIN RUNNER
# --------------------------------------------------------
if __name__ == "__main__":

    print("\n🚀 Starting PostgreSQL Setup...\n")

    # 1. Create DB if not exist
    if not database_exists():
        create_database()
    else:
        print(f"✔ Database '{DB_NAME}' already exists")

    # 2. Connect to DB
    conn = connect_db()
    print("✔ Connected to DB")

    # 3. Create tables
    create_tables(conn)

    # 4. Load CSV data
    print("\n🔹 Loading CSV files...")
    hr_df = pd.read_csv(HR_CSV)
    reviews_df = pd.read_csv(REVIEWS_CSV)
    perf_df = pd.read_csv(PERF_CSV) if os.path.exists(PERF_CSV) else pd.DataFrame()
    print("✔ CSV loaded")

    # 5. Insert data
    insert_data(conn, hr_df, "employees")
    insert_data(conn, reviews_df, "reviews")

    if not perf_df.empty:
        insert_data(conn, perf_df, "performance")

    # 6. JOIN: employee_master
    cur = conn.cursor()
    print("\n🔹 Creating employee_master view/table...")

    cur.execute("DROP TABLE IF EXISTS employee_master CASCADE")
    cur.execute("""
        CREATE TABLE employee_master AS
        SELECT 
            e.EmployeeID,
            e.Name,
            e.Age,
            e.Gender,
            e.Department,
            e.Salary,
            e.Tenure,
            r.ReviewText,
            r.Sentiment
        FROM employees e
        LEFT JOIN reviews r
        ON e.EmployeeID = r.EmployeeID;
    """)

    conn.commit()
    print("✔ employee_master created")

    # 7. Example GROUP BY
    print("\n📌 GROUP BY (Avg Salary by Department)")
    cur.execute("""
        SELECT Department, COUNT(*), AVG(Salary)
        FROM employees
        GROUP BY Department;
    """)
    print(cur.fetchall())

    # 8. Example WINDOW function
    print("\n📌 WINDOW FUNCTION (Rank salary in each department)")
    cur.execute("""
        SELECT 
            EmployeeID, Name, Department, Salary,
            RANK() OVER (PARTITION BY Department ORDER BY Salary DESC)
        FROM employees
        LIMIT 10;
    """)
    print(cur.fetchall())

    # 9. ACID transaction example
    print("\n🔹 ACID Transaction Example:")
    try:
        cur.execute("BEGIN;")
        cur.execute("UPDATE employees SET Salary = Salary + 1000 WHERE Department='Sales';")
        cur.execute("COMMIT;")
        print("✔ Transaction committed")
    except Exception as e:
        cur.execute("ROLLBACK;")
        print("❌ Rolled back due to:", e)

    conn.close()
    print("\n🎉 PostgreSQL Setup Completed Successfully!")
