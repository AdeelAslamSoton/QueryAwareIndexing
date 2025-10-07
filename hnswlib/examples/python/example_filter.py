import pandas as pd
import numpy as np

# ==========================
# Read embeddings from CSV
# ==========================
def read_embeddings_from_csv(filename: str, embedding_col: str):
    """Reads embeddings from CSV and parses them into numpy arrays."""
    df = pd.read_csv(filename, sep=";")

    def parse_embedding(x):
        if isinstance(x, str):
            x = x.replace('[', '').replace(']', '')
            return np.array([float(i) for i in x.split(',') if i.strip() != ""], dtype=np.float32)
        elif isinstance(x, (list, np.ndarray)):
            return np.array(x, dtype=np.float32)
        else:
            raise ValueError(f"Unexpected format in embedding column: {x}")

    df[embedding_col] = df[embedding_col].apply(parse_embedding)
    return df


# ==========================
# Progressive reshuffling
# ==========================
def progressive_reshuffle_embeddings(df_dataset, df_queries, embedding_col,
                                     step=10000, max_queries=1000, output_file="", debug=False):
    """
    Progressive reshuffling:
    - Q0 → sort [0:N]
    - Qi → keep first (i*step) and last (i*step) fixed,
           reshuffle the middle [i*step : N - i*step] by descending Euclidean distance
    """
    N = len(df_dataset)
    reshuffled_idx = np.arange(N)
    dataset_embeddings = np.stack(df_dataset[embedding_col].values)

    total_queries = min(len(df_queries), max_queries)

    for query_index, query_row in enumerate(df_queries.itertuples(), start=0):  # Q0 starts at 0
        if query_index >= total_queries:
            break

        query_embedding = getattr(query_row, embedding_col)

        # Define untouched front and back
        fixed = min(query_index * step, N // 2)  # don't freeze more than half
        mid_start = fixed
        mid_end = N - fixed

        if mid_start < mid_end:  # Only reshuffle if middle exists
            mid_indexes = reshuffled_idx[mid_start:mid_end]
            mid_embeddings = dataset_embeddings[mid_indexes]

            # Euclidean distances
            distances = np.linalg.norm(mid_embeddings - query_embedding, axis=1)

            # Sort descending (farthest first)
            sorted_mid = mid_indexes[np.argsort(-distances)]
            reshuffled_idx[mid_start:mid_end] = sorted_mid

        # Debug / progress output
        if debug:
            print(f"\n--- After Q{query_index} ---")
            print("Fixed front:", mid_start, " Fixed back:", N - mid_end)
            print("First 20 indices:", reshuffled_idx[:20].tolist())
            print("Last 20 indices:", reshuffled_idx[-20:].tolist())
        else:
            if query_index % 50 == 0 or query_index == total_queries - 1:
                print(f"✅ Processed query {query_index}/{total_queries - 1}")

    # Apply reshuffled order
    reshuffled_df = df_dataset.iloc[reshuffled_idx].copy()

    # Track original indices for debugging
    reshuffled_df = reshuffled_df.reset_index().rename(columns={"index": "original_index"})
    reshuffled_df[embedding_col] = reshuffled_df[embedding_col].apply(lambda x: ', '.join(map(str, x)))

    # Save if output file is given
    if output_file:
        reshuffled_df.to_csv(output_file, index=False, sep=";")
        print(f"\n🎉 Saved reshuffled dataset to {output_file}")

    return reshuffled_df


# ==========================
# Example usage
# ==========================
if __name__ == "__main__":
    dataset_csv = "/data4/hnsw/TripClick/documents_full.csv"
    queries_csv = "/data4/hnsw/TripClick/QueriesForQueriesAware/queries_samples_filtered.csv"
    output_csv = "/data3/Adeel/DatasetReshuffle/Trip_click_sequential_1000queries-10000.csv"

    # Load dataset
    df_dataset = read_embeddings_from_csv(dataset_csv, embedding_col="embedding")
    print(f"✅ Dataset loaded: {df_dataset.shape}")

    # Load queries
    df_queries = read_embeddings_from_csv(queries_csv, embedding_col="embedding")
    print(f"✅ Queries loaded: {df_queries.shape} (processing first 1000 only)")

    # Progressive reshuffling (limit to first 1000 queries)
    final_reshuffled_df = progressive_reshuffle_embeddings(
        df_dataset,
        df_queries,
        embedding_col="embedding",
        step=10000,
        max_queries=1000,
        output_file=output_csv,
        debug=False   # change to True for small-scale debugging
    )
