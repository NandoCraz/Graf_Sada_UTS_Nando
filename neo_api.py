import os
import sys
import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase

URI = "bolt://127.0.0.1:7687"
AUTH = ("neo4j", "firstneo4j")
DB_NAME = "neo4j"

FILTERED_CSV = os.path.join("output", "events_filtered.csv")
BATCH_SIZE = 5000


def create_constraints(session):
    """Membuat constraint unik pada node agar indexing dan query MERGE berjalan optimal."""
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (v:Visitor) REQUIRE v.visitorId IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Item) REQUIRE i.itemId IS UNIQUE")


def import_batch(tx, batch):
    """Cypher query untuk mengimpor batch baris data interaksi ke Neo4j."""
    cypher_query = """
    UNWIND $batch AS row
    MERGE (v:Visitor {visitorId: row.visitorid})
    MERGE (i:Item {itemId: row.itemid})

    WITH v, i, row
    CALL (v, i, row) {
        WITH v, i, row WHERE row.event = 'view'
        CREATE (v)-[:VIEWED {timestamp: row.timestamp}]->(i)

        UNION ALL

        WITH v, i, row WHERE row.event = 'addtocart'
        CREATE (v)-[:ADDED_TO_CART {timestamp: row.timestamp}]->(i)

        UNION ALL

        WITH v, i, row WHERE row.event = 'transaction'
        CREATE (v)-[:BOUGHT {timestamp: row.timestamp, transactionId: row.transactionid}]->(i)
    }
    """
    tx.run(cypher_query, batch=batch)


def import_filtered_data_if_needed(driver):
    """Mengecek apakah data subgraph sudah ada di database, jika belum maka diimpor."""
    if not os.path.exists(FILTERED_CSV):
        print(f"Perhatian: File '{FILTERED_CSV}' belum ada. Silakan jalankan 'persiapan_data.py' terlebih dahulu!")
        sys.exit(1)

    with driver.session(database=DB_NAME) as session:
        count = session.run("MATCH (n) RETURN count(n) AS total").single()["total"]
        if count > 0:
            print(f"Database '{DB_NAME}' sudah memiliki {count:,} simpul.")
            return

        print(f"Database '{DB_NAME}' masih kosong. Mengimpor data dari '{FILTERED_CSV}'...")
        create_constraints(session)
        df = pd.read_csv(FILTERED_CSV)
        total_chunks = (len(df) // BATCH_SIZE) + 1

        for i in tqdm(range(0, len(df), BATCH_SIZE), total=total_chunks, desc="Import Subgraph"):
            chunk = df.iloc[i : i + BATCH_SIZE].copy()
            chunk["transactionid"] = chunk["transactionid"].fillna("").astype(str)
            records = chunk.to_dict(orient="records")
            session.execute_write(import_batch, records)

        print("[OK] Impor data subgraph selesai.")


def query_subgraph_composition(driver):
    """
    Menjalankan query Python API untuk menguraikan jenis dan jumlah simpul serta relasi,
    disertai penjelasan Cypher SQL.
    """
    print("\n" + "=" * 70)
    print("NO 3: KOMPOSISI SUBGRAPH DENGAN PYTHON API NEO4J")
    print("=" * 70)

    # menghitung jenis simpul dan jumlahnya
    cypher_nodes = """
    MATCH (n)
    RETURN labels(n)[0] AS JenisSimpul, count(n) AS Jumlah
    ORDER BY Jumlah DESC
    """

    # menghitung jenis relasi dan jumlahnya
    cypher_edges = """
    MATCH ()-[r]->()
    RETURN type(r) AS JenisRelasi, count(r) AS Jumlah
    ORDER BY Jumlah DESC
    """

    with driver.session(database=DB_NAME) as session:
        nodes_result = session.run(cypher_nodes).data()
        edges_result = session.run(cypher_edges).data()

    print("\n[A. Hasil Query Komposisi Simpul (Nodes)]")
    total_nodes = 0
    for row in nodes_result:
        label = row["JenisSimpul"]
        jumlah = row["Jumlah"]
        total_nodes += jumlah
        print(f"  * Jenis Simpul: {label:<15} | Jumlah: {jumlah:>10,} simpul")
    print(f"  -------------------------------------------------")
    print(f"    Total Simpul (Nodes)        | Jumlah: {total_nodes:>10,} simpul")

    print("\n[B. Hasil Query Komposisi Sisi / Relasi (Edges)]")
    total_edges = 0
    for row in edges_result:
        rel_type = row["JenisRelasi"]
        jumlah = row["Jumlah"]
        total_edges += jumlah
        print(f"  * Jenis Relasi: {rel_type:<15} | Jumlah: {jumlah:>10,} relasi")
    print(f"  -------------------------------------------------")
    print(f"    Total Relasi (Edges)        | Jumlah: {total_edges:>10,} relasi")

    print("\n" + "=" * 70)
    print("PENJELASAN CYPHER SQL YANG DIGUNAKAN:")
    print("=" * 70)
    print("1. Query Jenis dan Jumlah Simpul:")
    print("   --------------------------------------------------------------")
    print("   MATCH (n)")
    print("   RETURN labels(n)[0] AS JenisSimpul, count(n) AS Jumlah")
    print("   ORDER BY Jumlah DESC;")
    print("   --------------------------------------------------------------")
    print("   * MATCH (n)         : Mengambil seluruh simpul (node) yang ada pada database graph.")
    print("   * labels(n)[0]      : Mengambil label/tipe dari simpul (misalnya :Visitor dan :Item).")
    print("   * count(n)          : Fungsi agregasi untuk menghitung banyaknya simpul per kelompok label.")
    print("   * ORDER BY .. DESC  : Mengurutkan hasil dari jumlah simpul terbanyak.")
    print()
    print("2. Query Jenis dan Jumlah Relasi:")
    print("   --------------------------------------------------------------")
    print("   MATCH ()-[r]->()")
    print("   RETURN type(r) AS JenisRelasi, count(r) AS Jumlah")
    print("   ORDER BY Jumlah DESC;")
    print("   --------------------------------------------------------------")
    print("   * MATCH ()-[r]->()  : Mencari seluruh edge/relasi berarah yang menghubungkan dua simpul.")
    print("   * type(r)           : Mengambil nama tipe relasi (seperti VIEWED, ADDED_TO_CART, BOUGHT).")
    print("   * count(r)          : Menghitung total kemunculan setiap jenis interaksi pada subgraph.")
    print("=" * 70)


def main():
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
        print(f"Terhubung ke Neo4j pada {URI} (Database: {DB_NAME})")
    except Exception as e:
        print(f"Gagal terhubung ke database Neo4j: {e}")
        sys.exit(1)

    try:
        import_filtered_data_if_needed(driver)
        query_subgraph_composition(driver)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
