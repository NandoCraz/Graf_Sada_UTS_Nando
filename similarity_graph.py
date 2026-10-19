import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase

URI = "bolt://127.0.0.1:7687"
AUTH = ("neo4j", "firstneo4j")
DB_NAME = "neo4j"

INPUT_CSV = os.path.join("output", "events_filtered.csv")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_item_sparse_matrix():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: File '{INPUT_CSV}' tidak ditemukan!")
        print("Silakan jalankan 'persiapan_data.py' terlebih dahulu.")
        sys.exit(1)

    print(f"Membaca data interaksi dari '{INPUT_CSV}'...")
    df = pd.read_csv(INPUT_CSV)
    bobot_event = {"view": 1, "addtocart": 2, "transaction": 3}
    df["weight"] = df["event"].map(bobot_event).fillna(1)

    print("Membentuk Matriks Item-User (Sparse Matrix CSR)...")
    df_grouped = df.groupby(["itemid", "visitorid"], as_index=False)["weight"].sum()

    item_cat = df_grouped["itemid"].astype("category")
    user_cat = df_grouped["visitorid"].astype("category")

    item_categories = item_cat.cat.categories
    user_categories = user_cat.cat.categories

    row_indices = item_cat.cat.codes
    col_indices = user_cat.cat.codes

    item_user_mat = csr_matrix(
        (df_grouped["weight"], (row_indices, col_indices)),
        shape=(len(item_categories), len(user_categories)),
        dtype=np.float32
    )

    return df, item_user_mat, item_categories


def hitung_dan_tulis_similarity(driver, df, item_user_mat, item_categories):
    print("\n" + "=" * 70)
    print("NO 7: PERHITUNGAN SIMILARITY DENGAN PYTHON & PENULISAN KE GRAF NEO4J")
    print("=" * 70)

    item_sums = np.array(item_user_mat.sum(axis=1)).flatten()
    target_idx = int(np.argmax(item_sums))
    target_item_id = int(item_categories[target_idx])
    
    print(f"Record Acuan yang Dipilih : Item ID {target_item_id}")
    print(f"Total Interaksi Akumulatif: {item_sums[target_idx]:.0f} poin\n")

    print("Menghitung Cosine Similarity menggunakan Python (scikit-learn)...")
    target_vector = item_user_mat[target_idx]
    sim_scores = cosine_similarity(target_vector, item_user_mat).flatten()

    sim_scores[target_idx] = -1.0

    top_5_indices = np.argsort(sim_scores)[::-1][:5]
    top_5 = {int(item_categories[idx]): float(sim_scores[idx]) for idx in top_5_indices}

    print("\n--------------------------------------------------------------")
    print(f"Hasil Top 5 Item Paling Mirip dengan Item {target_item_id}:")
    print("--------------------------------------------------------------")
    for rank, (itm, score) in enumerate(top_5.items(), 1):
        print(f"  {rank}. Item ID: {itm:<10} | Cosine Similarity Score: {score:.4f}")
    print("--------------------------------------------------------------")

    print("\n" + "-" * 70)
    print("(a) CYPHER SQL UNTUK MENDAPATKAN 5 RECORDS DENGAN SIMILARITY TERTINGGI:")
    print("-" * 70)
    cypher_calc = f"""
    // Menghitung Jaccard Similarity langsung via Cypher di Neo4j
    MATCH (i1:Item {{itemId: {target_item_id}}})<-[:VIEWED|ADDED_TO_CART|BOUGHT]-(v:Visitor)-[:VIEWED|ADDED_TO_CART|BOUGHT]->(i2:Item)
    WHERE i1 <> i2
    WITH i1, i2, count(DISTINCT v) AS intersection
    MATCH (v1:Visitor)-[:VIEWED|ADDED_TO_CART|BOUGHT]->(i1)
    WITH i1, i2, intersection, count(DISTINCT v1) AS s1
    MATCH (v2:Visitor)-[:VIEWED|ADDED_TO_CART|BOUGHT]->(i2)
    WITH i1, i2, intersection, s1, count(DISTINCT v2) AS s2
    WITH i2.itemId AS RekomendasiItem, toFloat(intersection) / (s1 + s2 - intersection) AS SimilarityScore
    RETURN RekomendasiItem, SimilarityScore
    ORDER BY SimilarityScore DESC
    LIMIT 5;
    """
    print(cypher_calc)

    print("-" * 70)
    print("(b) MENULISKAN KEMBALI NILAI SIMILARITY KE GRAF NEO4J:")
    print("-" * 70)
    
    write_cypher = """
    UNWIND $sim_data AS row
    MERGE (i1:Item {itemId: row.target})
    MERGE (i2:Item {itemId: row.similar})
    MERGE (i1)-[r:SIMILAR_TO]->(i2)
    SET r.score = row.score
    """

    sim_data = [
        {"target": int(target_item_id), "similar": int(itm), "score": float(score)}
        for itm, score in top_5.items()
    ]

    with driver.session(database=DB_NAME) as session:
        session.run(write_cypher, sim_data=sim_data)
        print("[OK] Relasi 'SIMILAR_TO' berhasil dituliskan ke database Neo4j.")

        verify_cypher = f"""
        MATCH (i1:Item {{itemId: {target_item_id}}})-[r:SIMILAR_TO]->(i2:Item)
        RETURN i1.itemId AS ItemAcuan, i2.itemId AS ItemMirip, r.score AS NilaiSimilarity
        ORDER BY NilaiSimilarity DESC;
        """
        hasil_db = session.run(verify_cypher).data()

    print("\nHasil Query Verifikasi dari Database Neo4j:")
    print("---------------------------------------------------------------------------------")
    print(f"{'Item Acuan':<15} | {'Item Mirip':<15} | {'Relasi':<15} | {'Similarity Score'}")
    print("---------------------------------------------------------------------------------")
    for r in hasil_db:
        print(f"{r['ItemAcuan']:<15} | {r['ItemMirip']:<15} | {'[:SIMILAR_TO]':<15} | {r['NilaiSimilarity']:.4f}")
    print("---------------------------------------------------------------------------------")

    return target_item_id, top_5


def visualisasi_sebelum_dan_sesudah(driver, target_item_id, top_5):
    print("\n" + "=" * 70)
    print("NO 8: KONDISI SEBELUM DAN SESUDAH PERHITUNGAN SIMILARITY")
    print("=" * 70)

    print("\n[A. Kondisi SEBELUM Perhitungan Similarity]")
    print("Deskripsi: Graf hanya terdiri dari relasi bipartit antara User dan Item")
    print("           (tidak ada hubungan langsung antar-item).")
    cypher_sebelum = f"""
    MATCH (v:Visitor)-[r]->(i:Item {{itemId: {target_item_id}}})
    RETURN v.visitorId AS visitorId, type(r) AS relType, i.itemId AS itemId
    LIMIT 20;
    """
    print("\nCYPHER SQL SEBELUM:")
    print(cypher_sebelum)

    print("\n[B. Kondisi SESUDAH Perhitungan Similarity]")
    print("Deskripsi: Pada graf telah ditambahkan edge relasi langsung [:SIMILAR_TO] antar-item")
    print("           yang membawa properti score kemiripan.")
    cypher_sesudah = f"""
    MATCH (i1:Item {{itemId: {target_item_id}}})-[r:SIMILAR_TO]->(i2:Item)
    RETURN i1.itemId AS ItemAcuan, r.score AS SimilarityScore, i2.itemId AS ItemMirip;
    """
    print("\nCYPHER SQL SESUDAH:")
    print(cypher_sesudah)

    with driver.session(database=DB_NAME) as session:
        data_sebelum = session.run(cypher_sebelum).data()

    print("\nMembuat visualisasi graf SEBELUM similarity...")
    G_prev = nx.DiGraph()
    center_node = f"Item {target_item_id}"
    G_prev.add_node(center_node, color="#E91E63", size=900)

    for row in data_sebelum:
        u_node = f"User {row['visitorId']}"
        G_prev.add_node(u_node, color="#4CAF50", size=300)
        G_prev.add_edge(u_node, center_node)

    plt.figure(figsize=(9, 7))
    pos_p = nx.spring_layout(G_prev, seed=42)
    colors_p = [d["color"] for _, d in G_prev.nodes(data=True)]
    sizes_p = [d["size"] for _, d in G_prev.nodes(data=True)]
    nx.draw_networkx_nodes(G_prev, pos_p, node_color=colors_p, node_size=sizes_p, alpha=0.9)
    nx.draw_networkx_edges(G_prev, pos_p, edge_color="#B0BEC5", arrows=True, arrowsize=14, alpha=0.7)
    nx.draw_networkx_labels(G_prev, pos_p, font_size=7)
    plt.title(f"Kondisi SEBELUM Similarity (Bipartite User-Item {target_item_id})", fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    out_prev = os.path.join(OUTPUT_DIR, "graf_sebelum_similarity.png")
    plt.savefig(out_prev, dpi=300)
    plt.close()
    print(f"[OK] Plot graf SEBELUM similarity disimpan di: '{out_prev}'")

    print("Membuat visualisasi graf SESUDAH similarity...")
    G_post = nx.Graph()
    acuan_label = f"Item Acuan\nID: {target_item_id}"
    G_post.add_node(acuan_label, color="#E91E63", size=1000)

    for itm, score in top_5.items():
        mirip_label = f"Item {itm}\n(sim: {score:.3f})"
        G_post.add_node(mirip_label, color="#2196F3", size=800)
        G_post.add_edge(acuan_label, mirip_label, weight=score)

    plt.figure(figsize=(9, 7))
    pos_post = nx.spring_layout(G_post, seed=42)
    colors_post = [d["color"] for _, d in G_post.nodes(data=True)]
    sizes_post = [d["size"] for _, d in G_post.nodes(data=True)]
    nx.draw_networkx_nodes(G_post, pos_post, node_color=colors_post, node_size=sizes_post, alpha=0.9)
    nx.draw_networkx_edges(G_post, pos_post, edge_color="#FF9800", width=2.5, alpha=0.85)
    nx.draw_networkx_labels(G_post, pos_post, font_size=8, font_weight="bold")
    plt.title(f"Kondisi SESUDAH Similarity (Relasi SIMILAR_TO pada Item {target_item_id})", fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    out_post = os.path.join(OUTPUT_DIR, "graf_sesudah_similarity.png")
    plt.savefig(out_post, dpi=300)
    plt.close()
    print(f"[OK] Plot graf SESUDAH similarity disimpan di: '{out_post}'")
    print("=" * 70)


def main():
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
    except Exception as e:
        print(f"Gagal terhubung ke Neo4j: {e}")
        sys.exit(1)

    try:
        df, item_user_mat, item_categories = load_item_sparse_matrix()
        target_item_id, top_5 = hitung_dan_tulis_similarity(driver, df, item_user_mat, item_categories)
        visualisasi_sebelum_dan_sesudah(driver, target_item_id, top_5)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
