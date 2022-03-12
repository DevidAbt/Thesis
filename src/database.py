import sqlite3


class Database:
    def __init__(self) -> None:
        conn = sqlite3.connect("data.db")
        self._conn = conn
        self._cursor = conn.cursor()

        self._cursor.execute('''CREATE TABLE IF NOT EXISTS Comparison (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            graph_name TEXT,
            alpha REAL,
            p REAL,
            degree_binary REAL,
            degree_random_walk REAL,
            degree_clustering_coefficient REAL,
            degree_local_closure REAL,
            binary_random_walk REAL,
            binary_clustering_coefficient REAL,
            binary_local_closure REAL,
            random_walk_clustering_coefficient REAL,
            random_walk_local_closure REAL,
            clustering_coefficient_local_closure REAL,
            UNIQUE(graph_name, alpha, p) ON CONFLICT IGNORE
        )''')

    @property
    def conn(self):
        return self._conn

    def insert_comparsion(self, graph_name, alpha, p, degree_binary, degree_random_walk, degree_clustering_coefficient, degree_local_closure, binary_random_walk, binary_clustering_coefficient, binary_local_closure, random_walk_clustering_coefficient, random_walk_local_closure, clustering_coefficient_local_closure):
        self._cursor.execute(
            f'''DELETE FROM Comparsion WHERE graph_name='{graph_name}' AND alpha={alpha} AND p={p};''')
        self._cursor.execute(f'''INSERT INTO Comparsion (graph_name, alpha, p, degree_binary, degree_random_walk, degree_clustering_coefficient, degree_local_closure, binary_random_walk, binary_clustering_coefficient, binary_local_closure, random_walk_clustering_coefficient, random_walk_local_closure, clustering_coefficient_local_closure)
                                VALUES ('{graph_name}', {alpha}, {p}, {degree_binary}, {degree_random_walk}, {degree_clustering_coefficient}, {degree_local_closure}, {binary_random_walk}, {binary_clustering_coefficient}, {binary_local_closure}, {random_walk_clustering_coefficient}, {random_walk_local_closure}, {clustering_coefficient_local_closure});''')
