# Tugas Graf Sains Data (UTS)

## 1. Nama Database Yang Digunakan Pada Tutorial
Nama Database: **neo4j**

## 2. Akses Lain untuk Mengakses Database Lain dalam 1 Instance Neo4j

### A. Klausul `USE <databaseName>` (Cypher DML/Query)
`USE` digunakan untuk mengarahkan eksekusi perintah Cypher (`MATCH`, `CREATE`, `MERGE`, dll.) ke database tertentu secara langsung:

```cypher
USE databaseLain
MATCH (n:Person)
RETURN n.name;
```

### B. Perintah Administratif Cypher (DDL / Database Management)
Kita juga bisa mengelola atau mengakses database lain lewat perintah DDL pada database bawaan `system`:

- **Melihat daftar database yang ada:**
  ```cypher
  SHOW DATABASES;
  ```
- **Membuat database baru:**
  ```cypher
  CREATE DATABASE dbPertemanan;
  ```
- **Menghentikan dan menjalankan database:**
  ```cypher
  STOP DATABASE dbPertemanan;
  START DATABASE dbPertemanan;
  ```
- **Menghapus database:**
  ```cypher
  DROP DATABASE dbPertemanan;
  ```

---

### C. Lewat Interface / Tool yang Digunakan
Selain lewat query Cypher langsung, kita juga bisa ganti target database dari tool atau script yang dipakai:

- **Neo4j Browser:** Tinggal ketik perintah `:use <nama_db>` di command bar, misalnya:
  ```text
  :use databaseLain
  ```
- **Cypher Shell (Terminal):** Menentukan database langsung saat login pakai parameter `-d`:
  ```bash
  cypher-shell -d databaseLain -u neo4j -p password
  ```
- **Python Driver (Session level):** Kalau di Python pakai session, kita tentukan nama databasenya waktu buka session:
  ```python
  with driver.session(database="databaseLain") as session:
      result = session.run("MATCH (n) RETURN count(n)")
  ```

---

## 3. Jenis Graf Pertemanan yang Dibuat

Kalau dilihat dari potongan kode di `index.py`:

```python
driver.execute_query(
    "MERGE (a:Person {name: $name}) "
    "MERGE (friend:Person {name: $friend_name}) "
    "MERGE (a)-[:KNOWS]->(friend)",
    name=name, friend_name=friend_name, database_="neo4j"
)
```

Graf yang terbentuk memiliki beberapa jenis/karakteristik:

1. **Graf Berarah (*Directed Graph*)**
   Relasinya punya tanda panah `->` pada `(a)-[:KNOWS]->(friend)`. Artinya relasi pertemanan ini sifatnya satu arah (asimetris), mirip sistem *following* di Twitter atau Instagram. Jadi Arthur mengenal Guinevere, tapi Guinevere belum tentu otomatis mengenal balik Arthur di dalam graf.

2. **Topologi Bintang (*Star Graph*)**
   Struktur jaringannya memusat ke satu titik. Node Arthur bertindak sebagai simpul pusat (*hub*), sedangkan teman-temannya (Guinevere, Lancelot, dan Merlin) jadi simpul daun di sekelilingnya. Teman-temannya ini hanya terhubung ke Arthur dan belum ada relasi pertemanan antar sesama mereka.

3. **Graf Sederhana (*Simple Graph*)**
   Karena menggunakan perintah `MERGE` (bukan `CREATE` biasa), data graf tidak akan membuat relasi ganda (*multiple edges*) untuk teman yang sama, dan tidak ada hubungan yang mengarah ke diri sendiri (*no self-loop*).

4. **Graf Homogen (*Homogeneous Graph*)**
   Semua node di graf ini memiliki tipe/label yang seragam yaitu `:Person`, dan jenis relasinya juga hanya satu macam yaitu `:KNOWS`.

5. **Property Graph**
   Menerapkan model *Property Graph* khas Neo4j, di mana entitasnya memiliki label (`Person`) beserta atribut/properti (`name`), dan relasinya memiliki label tipe (`KNOWS`).
