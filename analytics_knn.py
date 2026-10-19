import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase

# ==============================================================================
# KONFIGURASI DAN PATH
# ==============================================================================
P1_TRAIN_CSV = os.path.join("output", "events_p1_train.csv")
P2_TEST_CSV  = os.path.join("output", "events_p2_test.csv")
OUTPUT_DIR   = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Konfigurasi Neo4j (Opsional untuk integrasi graf)
NEO4J_URI  = "bolt://127.0.0.1:7687"
NEO4J_AUTH = ("neo4j", "firstneo4j")
NEO4J_DB   = "neo4j"

# Bobot aksi interaksi user
EVENT_WEIGHTS = {"view": 1, "addtocart": 2, "transaction": 3}

# Seed acak untuk reproducibility
RANDOM_SEED = 10
K_NEIGHBORS = 5
TOP_N_RECS  = 5


def muat_data():
    """Memuat data training P1 dan testing P2 dari folder output."""
    if not os.path.exists(P1_TRAIN_CSV) or not os.path.exists(P2_TEST_CSV):
        print(f"Error: File '{P1_TRAIN_CSV}' atau '{P2_TEST_CSV}' tidak ditemukan!")
        print("Silakan jalankan 'persiapan_data.py' terlebih dahulu.")
        sys.exit(1)

    print(f"Membaca data Training (P1) dari '{P1_TRAIN_CSV}'...")
    df_train = pd.read_csv(P1_TRAIN_CSV)
    print(f"Membaca data Testing  (P2) dari '{P2_TEST_CSV}'...")
    df_test = pd.read_csv(P2_TEST_CSV)

    df_train["weight"] = df_train["event"].map(EVENT_WEIGHTS).fillna(1)
    df_test["weight"]  = df_test["event"].map(EVENT_WEIGHTS).fillna(1)

    return df_train, df_test


def pilih_user_reproducible(df_train, df_test, seed=RANDOM_SEED):
    """
    Memilih user secara acak namun 100% reproducible.
    User dipilih dari irisan (intersection) antara Train (P1) dan Test (P2)
    agar hasil rekomendasi dapat diukur performanya (Hit@5 dan NDCG@5).
    """
    train_users = set(df_train["visitorid"].unique())
    test_users  = set(df_test["visitorid"].unique())
    common_users = sorted(list(train_users & test_users))

    print("\n" + "=" * 75)
    print("BAGIAN 1: PEMILIHAN USER SECARA ACAK (REPRODUCIBLE)")
    print("=" * 75)
    print(f"1. Total User di Fold Training (P1) : {len(train_users):,} user")
    print(f"2. Total User di Fold Testing  (P2) : {len(test_users):,} user")
    print(f"3. User yang aktif di P1 & P2       : {len(common_users):,} user (Kandidat Evaluasi)")

    # Set seed acak deterministik
    np.random.seed(seed)
    selected_user = int(np.random.choice(common_users))

    # Riwayat user di P1 (Training Profile)
    p1_items = df_train[df_train["visitorid"] == selected_user]["itemid"].tolist()
    # Riwayat user di P2 (Testing Ground Truth)
    p2_items = df_test[df_test["visitorid"] == selected_user]["itemid"].tolist()

    print(f"\n[HASIL PEMILIHAN]")
    print(f"- Parameter Seed Digunakan : np.random.seed({seed})")
    print(f"- User Terpilih (VisitorID): {selected_user}")
    print(f"- Riwayat Item di P1 (Train): {p1_items} (Total: {len(p1_items)} interaksi)")
    print(f"- Ground Truth di P2 (Test) : {p2_items} (Total: {len(p2_items)} interaksi)")
    print("-" * 75)

    return selected_user, p1_items, p2_items


def bangun_vsm_item(df_train):
    """Membentuk Matriks Item-User (CSR Matrix) dari data P1."""
    df_grouped = df_train.groupby(["itemid", "visitorid"], as_index=False)["weight"].sum()

    item_cat = df_grouped["itemid"].astype("category")
    user_cat = df_grouped["visitorid"].astype("category")

    item_cats = item_cat.cat.categories
    user_cats = user_cat.cat.categories

    row_indices = item_cat.cat.codes
    col_indices = user_cat.cat.codes

    item_user_mat = csr_matrix(
        (df_grouped["weight"], (row_indices, col_indices)),
        shape=(len(item_cats), len(user_cats)),
        dtype=np.float32
    )

    return item_user_mat, item_cats, user_cats


def rekomendasi_item_knn(item_user_mat, item_cats, user_items_p1, k=K_NEIGHBORS):
    """
    (1a) Menjalankan Item-based KNN:
    Untuk setiap item yang pernah dibuka user di P1, cari k=5 tetangga terdekat.
    Kemudian agregasi skor similarity untuk membentuk peringkat rekomendasi.
    """
    print("\n" + "=" * 75)
    print(f"BAGIAN 1 (a): SKRIP KNN ITEM-BASED DENGAN k = {k}")
    print("=" * 75)

    item_scores = {}
    item_neighbors_detail = {}

    for itm in set(user_items_p1):
        if itm in item_cats:
            idx = item_cats.get_loc(itm)
            # Hitung cosine similarity vektor item ini terhadap seluruh item lainnya
            sim_vector = cosine_similarity(item_user_mat[idx], item_user_mat).flatten()
            # Kecualikan item itu sendiri
            sim_vector[idx] = -1.0

            # Ambil k indeks dengan similarity tertinggi
            top_k_indices = np.argsort(sim_vector)[::-1][:k]

            neighbors = []
            print(f"\nItem Acuan User: {itm}")
            print(f"  Top {k} Tetangga Terdekat (Nearest Neighbors):")
            for rank, n_idx in enumerate(top_k_indices, 1):
                neighbor_item = int(item_cats[n_idx])
                sim_score = float(sim_vector[n_idx])
                neighbors.append((neighbor_item, sim_score))
                print(f"    {rank}. Item ID: {neighbor_item:<10} | Cosine Similarity: {sim_score:.4f}")

                # Agregasi skor untuk rekomendasi
                item_scores[neighbor_item] = item_scores.get(neighbor_item, 0.0) + sim_score

            item_neighbors_detail[itm] = neighbors

    # Urutkan seluruh kandidat item berdasarkan akumulasi skor similarity
    sorted_recs = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "-" * 75)
    print(f"HASIL AKHIR REKOMENDASI ITEM (TOP {TOP_N_RECS}):")
    print("-" * 75)
    print(f"{'Peringkat':<10} | {'Item ID':<12} | {'Akumulasi Skor':<18}")
    print("-" * 75)
    for rank, (itm, score) in enumerate(sorted_recs[:TOP_N_RECS], 1):
        print(f"Peringkat {rank:<2} : Item ID {itm:<10} | Skor: {score:.4f}")
    print("-" * 75)

    return sorted_recs, item_neighbors_detail


def hitung_metrik_evaluasi(rekomendasi, ground_truth_p2, k_eval=5):
    """
    (2a & 2b) Menghitung Hit@5 dan NDCG@5 berdasarkan Ground Truth data P2.
    """
    print("\n" + "=" * 75)
    print(f"BAGIAN 2: PERHITUNGAN EVALUASI REKOMENDASI (Hit@{k_eval} & NDCG@{k_eval})")
    print("=" * 75)

    gt_set = set(ground_truth_p2)
    top_k_recs = rekomendasi[:k_eval]

    # 1. Hit@K
    # Hit bernilai 1 jika ada setidaknya satu item rekomendasi yang ada di ground truth
    hit_at_k = 0
    hit_item = None
    hit_rank = None

    relevance_scores = []

    for rank, (itm, score) in enumerate(top_k_recs, 1):
        is_hit = itm in gt_set
        relevance_scores.append(1 if is_hit else 0)
        status_str = "[HIT RELEVAN]" if is_hit else "Tidak Relevan"
        print(f"Peringkat {rank:<2} : Item ID {itm:<10} | Skor: {score:.4f} | {status_str}")

        if is_hit and hit_at_k == 0:
            hit_at_k = 1
            hit_item = itm
            hit_rank = rank

    print("-" * 75)
    print(f"(a) HIT@{k_eval}:")
    print(f"    Item Ground Truth P2 : {list(gt_set)}")
    print(f"    Item Terkena Hit     : {hit_item} (pada peringkat ke-{hit_rank})")
    print(f"    NILAI Hit@{k_eval}          : {hit_at_k} (atau {hit_at_k * 100:.1f}%)")

    # 2. DCG@K
    # DCG = sum(rel_i / log2(i + 1))
    dcg = 0.0
    print(f"\n(b) NDCG@{k_eval} (Normalized Discounted Cumulative Gain):")
    print(f"    Langkah 1: Menghitung Discounted Cumulative Gain (DCG@{k_eval})")
    for i, rel in enumerate(relevance_scores, 1):
        discount = np.log2(i + 1)
        gain = rel / discount
        dcg += gain
        print(f"      - Peringkat {i}: rel={rel}, diskon=log2({i}+1)={discount:.4f} -> Gain = {gain:.4f}")
    print(f"    Total DCG@{k_eval} = {dcg:.4f}")

    # 3. IDCG@K (Ideal DCG)
    # Urutkan ground truth ideal (relevansi 1 di atas)
    ideal_relevance = sorted(relevance_scores, reverse=True)
    if sum(ideal_relevance) == 0:
        # Jika tidak ada yang relevan tapi ada di gt_set, idealnya ada 1 di peringkat 1
        ideal_relevance = [1] + [0] * (k_eval - 1)

    idcg = 0.0
    print(f"\n    Langkah 2: Menghitung Ideal DCG (IDCG@{k_eval})")
    for i, rel in enumerate(ideal_relevance, 1):
        if rel > 0:
            discount = np.log2(i + 1)
            gain = rel / discount
            idcg += gain
            print(f"      - Peringkat Ideal {i}: rel={rel}, diskon=log2({i}+1)={discount:.4f} -> Gain = {gain:.4f}")
    print(f"    Total IDCG@{k_eval} = {idcg:.4f}")

    # 4. NDCG@K
    ndcg = dcg / idcg if idcg > 0 else 0.0
    print(f"\n    Langkah 3: Menghitung NDCG@{k_eval} = DCG@{k_eval} / IDCG@{k_eval}")
    print(f"    NILAI NDCG@{k_eval}         : {ndcg:.4f} (atau {ndcg * 100:.2f}%)")
    print("=" * 75)

    return hit_at_k, ndcg, top_k_recs


def simpan_plot_evaluasi(top_k_recs, ground_truth_p2, hit_val, ndcg_val):
    """Menyimpan visualisasi grafik batang hasil Top Rekomendasi."""
    gt_set = set(ground_truth_p2)
    items = [f"Item {itm}" for itm, _ in top_k_recs]
    scores = [score for _, score in top_k_recs]
    colors = ["#4CAF50" if itm in gt_set else "#2196F3" for itm, _ in top_k_recs]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(items, scores, color=colors, edgecolor="black", alpha=0.85, width=0.55)

    for bar, (itm, score) in zip(bars, top_k_recs):
        height = bar.get_height()
        label = f"{score:.4f}\n(Hit!)" if itm in gt_set else f"{score:.4f}"
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, label,
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.ylim(0, max(scores) * 1.35 if scores else 1)
    plt.title(f"Top 5 Rekomendasi KNN (k=5)\nHit@5 = {hit_val} | NDCG@5 = {ndcg_val:.4f}",
              fontsize=12, fontweight="bold")
    plt.xlabel("Item Rekomendasi", fontsize=10)
    plt.ylabel("Skor Similarity Akumulatif", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    custom_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4CAF50", label="Relevan (Ada di P2 Testing)"),
        plt.Rectangle((0, 0), 1, 1, color="#2196F3", label="Kandidat Rekomendasi Lain")
    ]
    plt.legend(handles=custom_handles, loc="upper right")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "knn_recommendations_plot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\n[OK] Visualisasi evaluasi rekomendasi disimpan di: '{out_path}'")


def verifikasi_neo4j_cypher(selected_user, top_3_advertised):
    """Menampilkan dan mengecek ekuivalen query Cypher pada Neo4j."""
    print("\n" + "=" * 75)
    print("BONUS: QUERY CYPHER EKUIVALEN DI DATABASE NEO4J")
    print("=" * 75)
    cypher_rec = f"""
    // Rekomendasi Item untuk Visitor {selected_user} berdasarkan relasi SIMILAR_TO
    MATCH (v:Visitor {{visitorId: {selected_user}}})-[:VIEWED|ADDED_TO_CART|BOUGHT]->(i:Item)-[s:SIMILAR_TO]->(rec:Item)
    WHERE NOT (v)-[:VIEWED|ADDED_TO_CART|BOUGHT]->(rec)
    RETURN rec.itemId AS ItemRekomendasi, sum(s.score) AS TotalSkor
    ORDER BY TotalSkor DESC
    LIMIT 3;
    """
    print(cypher_rec.strip())
    print("-" * 75)

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
        with driver.session(database=NEO4J_DB) as session:
            # Pastikan user terdaftar di Neo4j
            res = session.run(
                f"MATCH (v:Visitor {{visitorId: {selected_user}}}) RETURN count(v) AS c"
            ).single()
            if res and res["c"] > 0:
                print(f"[OK] Visitor ID {selected_user} terverifikasi aktif di graf Neo4j.")
            else:
                print(f"[INFO] Visitor ID {selected_user} belum dimuat langsung ke simpul graf Neo4j.")
        driver.close()
    except Exception as e:
        print(f"[INFO] Neo4j tidak aktif atau dilewati: {e}")


def main():
    print("*" * 75)
    print("TUGAS GRAF SAINS DATA - BAGIAN 2.2 ANALYTICS")
    print("Sistem Rekomendasi K-Nearest Neighbors (KNN) & Evaluasi Metrik")
    print("*" * 75)

    # 1. Muat data P1 dan P2
    df_train, df_test = muat_data()

    # 2. Pemilihan user acak reproducible
    selected_user, p1_items, p2_items = pilih_user_reproducible(df_train, df_test, seed=RANDOM_SEED)

    # 3. Bentuk VSM Item-User
    print("\nMembentuk Matriks Item-User Sparse CSR...")
    item_user_mat, item_cats, user_cats = bangun_vsm_item(df_train)
    print(f"Dimensi Matriks Item: {item_user_mat.shape[0]:,} item x {item_user_mat.shape[1]:,} user")

    # 4. Jalankan Rekomendasi KNN (1a)
    rekomendasi, _ = rekomendasi_item_knn(item_user_mat, item_cats, p1_items, k=K_NEIGHBORS)

    # 5. Item yang diiklankan (1b)
    print("\n" + "=" * 75)
    print("BAGIAN 1 (b): 3 ITEM YANG DIIKLANKAN UNTUK k = 5")
    print("=" * 75)
    top_3_advertised = rekomendasi[:3]
    for i, (itm, score) in enumerate(top_3_advertised, 1):
        print(f"  Item Iklan {i}: Item ID {itm:<10} | Skor Kemiripan: {score:.4f}")
    print("=" * 75)

    # 6. Evaluasi Hit@5 dan NDCG@5 (2a & 2b)
    hit_val, ndcg_val, top_k = hitung_metrik_evaluasi(rekomendasi, p2_items, k_eval=5)

    # 7. Simpan grafik evaluasi
    simpan_plot_evaluasi(top_k, p2_items, hit_val, ndcg_val)

    # 8. Tampilkan query Cypher
    verifikasi_neo4j_cypher(selected_user, top_3_advertised)

    print("\n" + "*" * 75)
    print("EKSEKUSI SELESAI DENGAN SUKSES!")
    print("*" * 75)


if __name__ == "__main__":
    main()
