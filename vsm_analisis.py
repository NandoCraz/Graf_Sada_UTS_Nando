import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

INPUT_CSV = os.path.join("output", "events_filtered.csv")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: File '{INPUT_CSV}' tidak ditemukan.")
        print("Silakan jalankan 'persiapan_data.py' terlebih dahulu!")
        sys.exit(1)

    print(f"Membaca data interaksi terfilter dari '{INPUT_CSV}'...")
    df = pd.read_csv(INPUT_CSV)
    
    bobot_event = {"view": 1, "addtocart": 2, "transaction": 3}
    df["weight"] = df["event"].map(bobot_event).fillna(1)
    return df


def vsm_user(df):
    print("\n" + "=" * 70)
    print("NO 4: VECTOR SPACE MODELLING (VSM) - SIMPUL USER (VISITOR)")
    print("=" * 70)
    print("Konsep:")
    print("Pada VSM User, setiap entitas User direpresentasikan sebagai sebuah vektor")
    print("di dalam ruang fitur yang beranggotakan seluruh Item yang ada.")
    print("Vektor_User = [w(u, item_1), w(u, item_2), ..., w(u, item_M)]")
    print()

    print("Membentuk Matriks VSM User (Sparse Matrix CSR)...")
    df_grouped = df.groupby(["visitorid", "itemid"], as_index=False)["weight"].sum()

    user_cat = df_grouped["visitorid"].astype("category")
    item_cat = df_grouped["itemid"].astype("category")

    user_categories = user_cat.cat.categories
    item_categories = item_cat.cat.categories

    n_users = len(user_categories)
    n_items = len(item_categories)

    row_indices = user_cat.cat.codes
    col_indices = item_cat.cat.codes

    user_item_matrix = csr_matrix(
        (df_grouped["weight"], (row_indices, col_indices)),
        shape=(n_users, n_items),
        dtype=np.float32
    )

    total_sel = n_users * n_items
    terisi = user_item_matrix.nnz
    sparsity = (1.0 - (terisi / total_sel)) * 100

    print("\n[Karakteristik VSM User]")
    print(f"  * Dimensi Ruang Vektor (Jumlah Item) : {n_items:,} dimensi")
    print(f"  * Shape Matriks (Users x Items)      : {user_item_matrix.shape} (Baris x Kolom)")
    print(f"  * Total Potensi Elemen Matriks       : {total_sel:,}")
    print(f"  * Elemen Terisi Interaksi (Non-Zero) : {terisi:,}")
    print(f"  * Sparsity (Tingkat Kereggangan)    : {sparsity:.5f}%")
    print(f"  * Kepadatan Matriks (Density)       : {100 - sparsity:.5f}%")

    print("\nMembuat visualisasi graf proyeksi User (User-User Projection Network)...")
    top_users = df["visitorid"].value_counts().head(25).index.tolist()
    sub_df = df[df["visitorid"].isin(top_users)]

    B = nx.Graph()
    for _, row in sub_df.iterrows():
        u = f"User_{row['visitorid']}"
        i = f"Item_{row['itemid']}"
        B.add_node(u, bipartite=0)
        B.add_node(i, bipartite=1)
        B.add_edge(u, i)

    user_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]
    user_proj = nx.bipartite.projected_graph(B, user_nodes)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(user_proj, seed=42)
    nx.draw_networkx_nodes(user_proj, pos, node_color="#4CAF50", node_size=400, alpha=0.9)
    nx.draw_networkx_edges(user_proj, pos, edge_color="gray", alpha=0.5, width=1.5)
    nx.draw_networkx_labels(user_proj, pos, font_size=7, font_family="sans-serif")
    plt.title("Proyeksi Graf User (User-User Network via Common Items)", fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    out_file = os.path.join(OUTPUT_DIR, "vsm_user_plot.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[OK] Plot graf proyeksi User disimpan di: '{out_file}'")

    return n_users, n_items


def vsm_item(df):
    print("\n" + "=" * 70)
    print("NO 5: VECTOR SPACE MODELLING (VSM) - SIMPUL ITEM")
    print("=" * 70)
    print("Konsep:")
    print("Pada VSM Item, setiap produk/Item direpresentasikan sebagai vektor di dalam")
    print("ruang fitur yang beranggotakan seluruh User yang pernah berinteraksi dengannya.")
    print("Vektor_Item = [w(item, user_1), w(item, user_2), ..., w(item, user_N)]")
    print()

    print("Membentuk Matriks VSM Item (Sparse Matrix CSR)...")
    df_grouped = df.groupby(["itemid", "visitorid"], as_index=False)["weight"].sum()

    item_cat = df_grouped["itemid"].astype("category")
    user_cat = df_grouped["visitorid"].astype("category")

    item_categories = item_cat.cat.categories
    user_categories = user_cat.cat.categories

    n_items = len(item_categories)
    n_users = len(user_categories)

    row_indices = item_cat.cat.codes
    col_indices = user_cat.cat.codes

    item_user_matrix = csr_matrix(
        (df_grouped["weight"], (row_indices, col_indices)),
        shape=(n_items, n_users),
        dtype=np.float32
    )

    total_sel = n_items * n_users
    terisi = item_user_matrix.nnz
    sparsity = (1.0 - (terisi / total_sel)) * 100

    print("\n[Karakteristik VSM Item]")
    print(f"  * Dimensi Ruang Vektor (Jumlah User) : {n_users:,} dimensi")
    print(f"  * Shape Matriks (Items x Users)      : {item_user_matrix.shape} (Baris x Kolom)")
    print(f"  * Total Potensi Elemen Matriks       : {total_sel:,}")
    print(f"  * Elemen Terisi Interaksi (Non-Zero) : {terisi:,}")
    print(f"  * Sparsity (Tingkat Kereggangan)    : {sparsity:.5f}%")
    print(f"  * Kepadatan Matriks (Density)       : {100 - sparsity:.5f}%")

    print("\nMembuat visualisasi graf proyeksi Item (Item-Item Projection Network)...")
    top_items = df["itemid"].value_counts().head(25).index.tolist()
    sub_df = df[df["itemid"].isin(top_items)]

    B = nx.Graph()
    for _, row in sub_df.iterrows():
        u = f"User_{row['visitorid']}"
        i = f"Item_{row['itemid']}"
        B.add_node(u, bipartite=0)
        B.add_node(i, bipartite=1)
        B.add_edge(u, i)

    item_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 1]
    item_proj = nx.bipartite.projected_graph(B, item_nodes)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(item_proj, seed=42)
    nx.draw_networkx_nodes(item_proj, pos, node_color="#2196F3", node_size=400, alpha=0.9)
    nx.draw_networkx_edges(item_proj, pos, edge_color="gray", alpha=0.5, width=1.5)
    nx.draw_networkx_labels(item_proj, pos, font_size=7, font_family="sans-serif")
    plt.title("Proyeksi Graf Item (Item-Item Network via Common Visitors)", fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    out_file = os.path.join(OUTPUT_DIR, "vsm_item_plot.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[OK] Plot graf proyeksi Item disimpan di: '{out_file}'")

    return n_users, n_items


def komparasi_proyeksi(n_users, n_items):
    print("\n" + "=" * 70)
    print("NO 6: ANALISIS KOMPARASI PROYEKSI (USER-USER VS ITEM-ITEM)")
    print("=" * 70)
    
    print(f"Perbandingan Ukuran Entitas:")
    print(f"- Jumlah Simpul User (N) : {n_users:,} simpul")
    print(f"- Jumlah Simpul Item (M) : {n_items:,} simpul")
    print()
    print("Uraian Analisis Manakah Proyeksi yang Lebih Baik (Efektif/Efisien):")
    print("-" * 70)
    print("1. Efisiensi Komputasi (Kompleksitas Perhitungan):")
    if n_items < n_users:
        print(f"   -> Jumlah Item ({n_items:,}) LEBIH SEDIKIT daripada jumlah User ({n_users:,}).")
        print(f"      Komputasi pairwise similarity pada Item membutuhkan O(M^2) = {n_items**2:,} operasi,")
        print(f"      sedangkan pada User membutuhkan O(N^2) = {n_users**2:,} operasi.")
        print(f"      Artinya, PROYEKSI ITEM jauh lebih hemat memori dan jauh lebih cepat dihitung.")
    else:
        print(f"   -> Jumlah simpul relatif seimbang, namun struktur Item lebih terfokus.")

    print("\n2. Stabilitas Informasi (Temporal Stability):")
    print("   -> Preferensi dan perilaku pengunjung (User) sangat dinamis, sering berganti minat,")
    print("      serta banyak user yang hanya berkunjung 1 kali (cold-start / sesaat).")
    print("   -> Sebaliknya, karakteristik suatu Item (katalog produk e-commerce) jauh lebih stabil")
    print("      dan bertahan lama. Hubungan kemiripan antar-item tidak mudah kadaluarsa.")

    print("\n3. Efektivitas Rekomendasi:")
    print("   -> Graf proyeksi Item menghasilkan hubungan substitusi/komplementer produk yang langsung")
    print("      dapat disajikan ke pengguna ('Pelanggan yang melihat item ini juga melihat item itu').")
    print()
    print("=> KESIMPULAN AKHIR:")
    print("   PROYEKSI ITEM dinilai LEBIH BAIK, LEBIH EFEKTIF, DAN LEBIH EFISIEN untuk digunakan")
    print("   sebagai dasar perhitungan kemiripan (similarity) pada sistem e-commerce RetailRocket.")
    print("=" * 70)


def main():
    df = load_data()
    n_users, n_items = vsm_user(df)
    vsm_item(df)
    komparasi_proyeksi(n_users, n_items)


if __name__ == "__main__":
    main()
