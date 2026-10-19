import os
import sys
import pandas as pd

CSV_SOURCE = os.path.join("archive", "events.csv")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

P1_START_STR = "2015-06-30 00:00:00"
P1_END_STR   = "2015-07-07 23:59:59"

P2_START_STR = "2015-07-08 00:00:00"
P2_END_STR   = "2015-07-15 23:59:59"


def main():
    print("=" * 70)
    print("PROSES 1 & 2: FILTERING DATA INTERAKSI RETAILROCKET BERDASARKAN WAKTU")
    print("=" * 70)

    if not os.path.exists(CSV_SOURCE):
        print(f"Error: File sumber '{CSV_SOURCE}' tidak ditemukan!")
        sys.exit(1)

    print(f"Membaca dataset awal dari '{CSV_SOURCE}'...")
    df = pd.read_csv(CSV_SOURCE)
    print(f"Total baris data awal: {len(df):,} baris")

    p1_start_ts = int(pd.to_datetime(P1_START_STR).timestamp() * 1000)
    p1_end_ts   = int(pd.to_datetime(P1_END_STR).timestamp() * 1000)

    p2_start_ts = int(pd.to_datetime(P2_START_STR).timestamp() * 1000)
    p2_end_ts   = int(pd.to_datetime(P2_END_STR).timestamp() * 1000)

    print("\nMelakukan filter berdasarkan timestamp:")
    print(f"1. Fold Training (P1): {P1_START_STR} s/d {P1_END_STR}")
    print(f"   -> Epoch ms: {p1_start_ts} s/d {p1_end_ts}")
    print(f"2. Fold Testing  (P2): {P2_START_STR} s/d {P2_END_STR}")
    print(f"   -> Epoch ms: {p2_start_ts} s/d {p2_end_ts}")

    df_p1 = df[(df["timestamp"] >= p1_start_ts) & (df["timestamp"] <= p1_end_ts)].copy()
    df_p2 = df[(df["timestamp"] >= p2_start_ts) & (df["timestamp"] <= p2_end_ts)].copy()

    df_filtered = pd.concat([df_p1, df_p2], ignore_index=True)

    file_p1 = os.path.join(OUTPUT_DIR, "events_p1_train.csv")
    file_p2 = os.path.join(OUTPUT_DIR, "events_p2_test.csv")
    file_filtered = os.path.join(OUTPUT_DIR, "events_filtered.csv")

    print("\nMenyimpan file hasil filter ke folder 'output/'...")
    df_p1.to_csv(file_p1, index=False)
    df_p2.to_csv(file_p2, index=False)
    df_filtered.to_csv(file_filtered, index=False)

    print(f"[OK] File P1 (Training) tersimpan di : {file_p1} ({len(df_p1):,} baris)")
    print(f"[OK] File P2 (Testing)  tersimpan di : {file_p2} ({len(df_p2):,} baris)")
    print(f"[OK] File Gabungan      tersimpan di : {file_filtered} ({len(df_filtered):,} baris)")

    print("\n" + "-" * 70)
    print("RINGKASAN STATISTIK DATA:")
    print("-" * 70)
    print(f"- Fold P1: {df_p1['visitorid'].nunique():,} unique users | {df_p1['itemid'].nunique():,} unique items")
    print(f"- Fold P2: {df_p2['visitorid'].nunique():,} unique users | {df_p2['itemid'].nunique():,} unique items")
    print(f"- Total Data Terfilter : {len(df_filtered):,} interaksi")
    print(f"- Total Unique Users   : {df_filtered['visitorid'].nunique():,} users")
    print(f"- Total Unique Items   : {df_filtered['itemid'].nunique():,} items")
    print(f"- Distribusi Event     :")
    for ev, count in df_filtered["event"].value_counts().items():
        print(f"    * {ev:<12}: {count:,} ({count/len(df_filtered)*100:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
