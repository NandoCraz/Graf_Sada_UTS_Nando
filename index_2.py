import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Impor dataset RetailRocket (events.csv) ke database Neo4J."
    )
    parser.add_argument(
        "-u", "--user", default="neo4j", help="Username Neo4j (default: neo4j)"
    )
    parser.add_argument(
        "-p",
        "--password",
        default="firstneo4j",
        help="Password Neo4J (default: firstneo4j)",
    )
    parser.add_argument(
        "-b",
        "--bolt",
        default="bolt://127.0.0.1:7687",
        help="Neo4J URI / Bolt (default: bolt://127.0.0.1:7687)",
    )
    parser.add_argument(
        "-s",
        "--source",
        default="archive",
        help="Direktori/folder sumber data (default: archive)",
    )
    parser.add_argument(
        "-d",
        "--database",
        default="neo4j",
        help="Nama Database Neo4j (default: neo4j)",
    )
    return parser.parse_args()


def create_constraints(session):
    session.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Visitor) REQUIRE v.visitorId IS UNIQUE"
    )
    session.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Item) REQUIRE i.itemId IS UNIQUE"
    )


def import_batch(tx, batch):
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


args = parse_arguments()

csv_file_path = os.path.join(args.source, "events.csv")

if not os.path.exists(csv_file_path):
    print(f"Error: File '{csv_file_path}' tidak ditemukan!")
    sys.exit(1)

try:
    driver = GraphDatabase.driver(args.bolt, auth=(args.user, args.password))
    driver.verify_connectivity()
    print(f"Berhasil terhubung ke database '{args.database}' di {args.bolt}")
except Exception as e:
    print(f"Gagal terhubung ke database: {e}")
    sys.exit(1)

batch_size = 5000  # Ukuran per batch data

with driver.session(database=args.database) as session:
    print("Membuat constraint unik (Visitor & Item)...")
    create_constraints(session)

    print(f"Membaca file data dari: {csv_file_path}")
    chunks = pd.read_csv(csv_file_path, chunksize=batch_size)

    total_rows = 2756101  # Total baris data events.csv
    total_chunks = (total_rows // batch_size) + 1

    print("Memulai proses impor data ke Neo4j...")
    for chunk in tqdm(chunks, total=total_chunks, desc="Mengimpor Batch Data"):
        chunk["transactionid"] = chunk["transactionid"].fillna("").astype(str)
        records = chunk.to_dict(orient="records")
        session.execute_write(import_batch, records)

driver.close()
print("Proses impor berhasil selesai.")