# 🧠 Query-Aware Indexing for Hybrid Query Performance

This project implements a **novel query-aware indexing approach** designed to enhance navigation and entry-point selection for **approximate nearest neighbor (ANN) search**, specifically targeting improved performance in **agnostic hybrid queries**.

## Core Strategy

- **Learning from Past Queries:** Leverages historical query data to inform index structure and traversal.  
- **Attribute-Specific Indices:** Builds separate index trees based on selected parameter attributes.  
- **HNSWlib Integration:** Built on top of the efficient **HNSWlib** library.

---

## 🛠️ Setup and Building Instructions

Follow these steps to clone the repository and compile the project executables.

### 1️⃣ Clone the Project
```bash
git clone https://github.com/AdeelAslamSoton/QueryAwareIndexing.git
```
### 2️⃣ **Prepare and Build**

Navigate to the `hnswlib` directory, clean up any previous builds, and compile using CMake:

```bash
cd QueryAwareIndexing/hnswlib

# IMPORTANT: Remove the existing build folder to ensure a clean compilation
rm -rvf build 

# Create a new build directory and navigate into it
mkdir build
cd build

# Run CMake to configure the build
cmake ..

# Compile the project
make
```
### 📂 Data Reading and Example Usage

The project assumes embeddings and attributes are stored in CSV-like files where columns are separated by semicolons (`;`) and embedding values themselves are separated by commas (`,`).  

The following function reads embeddings and attributes from a file:

```cpp
pair<vector<vector<float>>, vector<string>> reading_files(const string &file_path, int &dim, bool queries = false)
{
    ifstream file(file_path);
    vector<vector<float>> total_embeddings;
    vector<string> attributes; // store string IDs
    string line;
    getline(file, line); // skip header
    while (getline(file, line))
    {
        stringstream ss(line);

        string embedding, attribute;
        if (!queries)
        {
            // Assumes embedding is first, then attribute, separated by semicolon (';')
            getline(ss, embedding, ';');
            getline(ss, attribute, ';');
        }
        else
        {
            // Same logic for queries
            getline(ss, embedding, ';');
            getline(ss, attribute, ';');
        }
        if (!isNullOrEmpty(embedding))
        {
            // Assumes embedding values are separated by comma (',')
            vector<float> embeddingVector = splitToFloat(embedding, ',');
            if (embeddingVector.size() == dim)
            {
                total_embeddings.push_back(embeddingVector);
                attributes.push_back(attribute); // store string ID
            }
        }
    }
    return {total_embeddings, attributes};
}
```
### 🚀 Running Executables

All commands are executed from the `/build` directory.

#### 1️⃣ Running Point Queries

To compute query-aware point query results, navigate to the `/build` folder and run:

```bash
./example_query_aware_point
```
#### 2️⃣ Compute Ground Truth

To obtain ground truth results:

1. Open the source file (e.g., `example_query_aware_point.cpp`).
2. Comment out the lines related to the proposed query-aware search method.
3. Uncomment the line marked `// ground_truth` (or similar) for ground truth computation.
4. Re-run `make` in the `/build` directory.
5. Execute the example again:

```bash
./example_query_aware_point
```
#### 3️⃣ Compute Selectivity and Distance

To calculate selectivity and distance metrics:

1. Ensure the `computeSelectivityAndDistance()` method is called in your main function:

```cpp
computeSelectivityAndDistance();
```
--

## ✉️ Questions or Queries

If you have any questions, suggestions, or issues regarding this project, please feel free to contact:

**Adeel Aslam**  
📧 Email: [A.Aslam@soton.ac.uk](mailto:A.Aslam@soton.ac.uk)

