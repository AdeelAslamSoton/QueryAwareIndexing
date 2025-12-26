#include "../../hnswlib/hnswlib.h"
#include "../../query_aware_hnsw/query_aware_hnswlib.h"
#include <thread>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <map>
#include <iostream>
#include <random>
#include "query_aware_hnsw/linear_regression.h"
using namespace std;
pair<vector<vector<float>>, vector<string>> reading_files(const string &file_path, int &dim, bool queries);
vector<std::string> reading_files_attributes(const string &file_path);
vector<string> splitString(const string &str, char delimiter);
vector<float> splitToFloat(const string &str, char delimiter);
size_t getRandomIndex(size_t max_elements);
bool isNullOrEmpty(const string &str);
string trim(const string &s);
vector<string> split_and_clean(const string &s);
vector<vector<string>> reading_files_attributes_and_cleaning(const string &file_path);

// Compute the selectivity and write it in a file
void writeSelectivityCSV(const std::vector<std::string> &attributes,
                         const std::vector<std::string> &meta_data_attributes,
                         const std::string &filename)
{
    std::ofstream csv_file(filename);
    if (!csv_file.is_open())
    {
        std::cerr << "Error: could not open file for writing\n";
        return;
    }

    csv_file << "Index,Selectivity\n"; // CSV header

    for (size_t i = 0; i < attributes.size(); ++i)
    {
        const std::string &attr = attributes[i];
        // std::cout<<"attr...."<<attr<<std::endl;
        int counter = 0;

        for (const auto &meta_attr : meta_data_attributes)
        {

            if (attr == meta_attr)
            {
                counter++;
            }
        }

        csv_file << i << "," << counter << "\n";
    }

    csv_file.close();
}
// It loads the groundtruth files id

std::unordered_map<std::string, std::string> reading_constants(std::string &path);

std::map<int, std::unordered_set<int>> groundTruthIds(const std::string &folder, int topK = 10)
{
    std::map<int, std::unordered_set<int>> groundtruth_map;

    int queryIndex = 0;
    while (true)
    {
        std::string filename = folder + "/Q" + std::to_string(queryIndex) + ".csv";

        std::ifstream infile(filename);
        if (!infile.is_open())
        {
            std::cerr << "No more files. Stopping at: " << filename << std::endl;
            break;
        }

        std::unordered_set<int> ids;
        std::string line;
        int lineCount = 0;
        std::getline(infile, line);
        while (std::getline(infile, line) && lineCount < topK)
        {
            if (line.empty())
                continue;

            size_t pos = line.find(',');
            if (pos == std::string::npos)
                continue;

            int id = std::stoi(line.substr(0, pos));
            ids.insert(id);

            lineCount++;
        }

        groundtruth_map[queryIndex] = ids;
        infile.close();
        queryIndex++;
    }

    return groundtruth_map;
}

std::vector<std::pair<float, float>> readDistanceSelectivity(const std::string &filename)
{

    std::vector<std::pair<float, float>> distance_selectivity;
    std::ifstream infile(filename);

    if (!infile.is_open())
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return distance_selectivity;
    }

    std::string line;
    // Skip header if present
    std::getline(infile, line);

    while (std::getline(infile, line))
    {

        if (line.empty())
            continue;

        std::istringstream iss(line);
        float distance, selectivity;

        size_t pos = line.find(',');
        if (pos != std::string::npos)
        {
            distance = std::stof(line.substr(0, pos));     // use stof for float
            selectivity = std::stof(line.substr(pos + 1)); // use stof for float
        }
        else
        {
            if (!(iss >> distance >> selectivity))
                continue; // skip invalid lines
        }

        distance_selectivity.emplace_back(distance, selectivity);
    }

    return distance_selectivity;
}

// Save filter map for reuse
void saveFilterMap(
    const std::vector<char> &filter_ids_map,
    const std::string &filename)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open())
    {
        throw std::runtime_error(
            "❌ Cannot open file for writing: " + filename);
    }
    out.write(filter_ids_map.data(), filter_ids_map.size());
    out.close();
}
// Simple file-exists check (works everywhere)
bool fileExists(const std::string &filename)
{
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}
// Load filter map
std::vector<char> loadFilterMap(
    const std::string &filename,
    size_t expected_size)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error(
            "❌ Cannot open file for reading: " + filename);
    }

    std::vector<char> buffer(expected_size);
    in.read(buffer.data(), expected_size);
    if (in.gcount() != static_cast<std::streamsize>(expected_size))
    {
        throw std::runtime_error("❌ Filter file size mismatch!");
    }
    in.close();
    return buffer;
}

// Create the directory if not exist

void create_directory_if_not_exists(const std::string &path)
{
    if (mkdir(path.c_str(), 0777) == -1)
    {
        if (errno == EEXIST)
        {
            // Directory already exists, that's fine
        }
        else
        {
            std::cerr << "Error creating directory: " << path << std::endl;
        }
    }
}
// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn)
{
    if (numThreads <= 0)
    {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1)
    {
        for (size_t id = start; id < end; id++)
        {
            fn(id, 0);
        }
    }
    else
    {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId)
        {
            threads.push_back(std::thread([&, threadId]
                                          {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                } }));
        }
        for (auto &thread : threads)
        {
            thread.join();
        }
        if (lastException)
        {
            std::rethrow_exception(lastException);
        }
    }
}
// Compute the selectivity
void computeSelectivityAndDistance()
{

    std::string constant_path = "../examples/constants/Selectivity_distance.txt";
    std::unordered_map<std::string, std::string> constants = reading_constants(constant_path);

    int dim = std::stoi(constants["DIM"]);
    std::map<int, std::unordered_set<int>> gts_ids = groundTruthIds(constants["GROUND_TRUTH_FOLDER"], 10);
    vector<std::string> meta_data_attributes = reading_files_attributes(constants["META_DATA_PATH"]);
    // Reading file for queries
    pair<vector<vector<float>>, vector<string>> queries = reading_files(constants["QUERIES_PATH"], dim, true);
    size_t elements = meta_data_attributes.size();

    writeSelectivityCSV(queries.second,
                        meta_data_attributes,
                        constants["SELECTIVITY"]);
    hnswlib::L2Space space(dim);
    qwery_aware::QweryAwareHNSW<float> *alg_query_aware = new qwery_aware::QweryAwareHNSW<float>(&space, elements);
    std::cout << "Index Loaded  " << elements << std::endl;
    alg_query_aware->loadIndex(constants["INDEX_PATH"], &space);
    std::cout << "Index Loaded  " << elements << std::endl;

    alg_query_aware->compute_distance_for_queries(queries.first, elements, constants["DISTANCE"]);
}

int main()
{
    // computeSelectivityAndDistance();
    // exit(-1);
    std::string constant_path = "../examples/constants/constants_and_filepaths_point.txt";
    std::unordered_map<std::string, std::string> constants = reading_constants(constant_path);

    int dim = std::stoi(constants["DIM"]);
    // Dimension of the elements

    int M = std::stoi(constants["M"]);
    // Tightly connected with internal dimensionality of the data
    // strongly affects the memory consumption
    int ef_construction = std::stoi(constants["EFC"]);
    // Controls index search speed/build speed tradeoff
    int num_threads = 40; // Number of threads for operations with index

    bool read_index = false; // For True if you have already index it is only for construction of index
    hnswlib::L2Space space(dim);

    if (read_index)
    {
        // std::cout << "Start" << std::endl;

        auto [embeddings, string_ids] = reading_files(constants["DATASET_FILE"], dim, false);
        int max_elements = embeddings.size();

        // Map int -> string IDs
        unordered_map<int, string> id_map;
        float *data = new float[dim * max_elements];
        for (int i = 0; i < max_elements; i++)
        {
            for (int j = 0; j < dim; j++)
                data[i * dim + j] = embeddings[i][j];
            id_map[i] = string_ids[i];
        }
        //  qwery_aware::QweryAwareHNSW<float> *alg_query_aware = new qwery_aware::QweryAwareHNSW<float>(&space, max_elements);

        hnswlib::HierarchicalNSW<float> *alg_query_aware =
            new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
        // Build HNSW index
        ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId)
                    { alg_query_aware->addPoint((void *)(data + dim * row), (int)row); });
        alg_query_aware->saveIndex(constants["INDEX_PATH"]);
        delete[] data;
        delete alg_query_aware;
    }
    else
    {

        std::map<int, std::unordered_set<int>> gts_ids = groundTruthIds(constants["GROUND_TRUTH_FOLDER"], 10);

        // vector<std::string> meta_data_attributes = reading_files_attributes(constants["META_DATA_PATH"]);
        vector<vector<string>> meta_data_attributes_cleaned = reading_files_attributes_and_cleaning(constants["META_DATA_PATH"]);

        // Reading file for queries
        pair<vector<vector<float>>, vector<string>> queries = reading_files(constants["QUERIES_PATH"], dim, true);

        size_t elements = meta_data_attributes_cleaned.size();

        std::vector<std::pair<float, float>> avg_dis_selectivity = readDistanceSelectivity(constants["DISTANCE_SELECTIVITY"]);

        qwery_aware::QweryAwareHNSW<float> *alg_query_aware = new qwery_aware::QweryAwareHNSW<float>(&space, elements, gts_ids, avg_dis_selectivity, constants);

        alg_query_aware->loadIndex(constants["INDEX_PATH"], &space);
        std::cout << "Index Loaded  " << gts_ids.size() << std::endl;
        // Searching

        // Step 3: Run searches for this batch
      //  std::vector<size_t> efs_array = {20, 80, 140, 200, 260, 320, 380, 440, 500, 560, 700, 900, 1100, 1300, 1500};
         std::vector<size_t> efs_array = {560};
        //  creating models
        std::vector<std::unique_ptr<LinearRegression>> models;

        for (size_t i = 0; i < efs_array.size(); i++)
        {
            models.push_back(std::unique_ptr<LinearRegression>(new LinearRegression(2, 0.15f)));
        }

        std::cout << "Queries Loaded" << queries.second.size() << std::endl;
        int k = 10;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        size_t total_elements = alg_query_aware->max_elements_;
        size_t batch_size = 1000;

        size_t num_batches = (queries.first.size() + batch_size - 1) / batch_size;
        int counter = 0;

        // Distance Finding
        float maximumDistance = std::numeric_limits<float>::lowest();

        for (int i = 0; i < 30; i++)
        {
            size_t random_index = getRandomIndex(queries.first.size());
            alg_query_aware->setEf(560);

            auto results = alg_query_aware->searchKnn(
                queries.first[random_index].data(), 10);

            if (!results.empty())
            {
                float currentMax = results.top().first; // largest for this query
                maximumDistance = std::max(maximumDistance, currentMax);
            }
        }

        // Find largest distance for normalization

        for (size_t b = 0; b < num_batches; b++)
        {

            counter++;
            size_t start = b * batch_size;
            size_t end = std::min(start + batch_size, queries.first.size());

            // std::cout << "Processing batch " << (b + 1) << "/" << num_batches
            //           << " (queries " << start << " to " << end - 1 << ")" << std::endl;
            create_directory_if_not_exists(constants["FILTER_PATH"]);
            std::string filter_file = constants["FILTER_PATH"] + "/filter_batch_" + std::to_string(start) + ".bin";

            std::vector<char> filter_ids_map;

            if (fileExists(filter_file))
            {
                // Load precomputed map
                filter_ids_map = loadFilterMap(filter_file, (end - start) * total_elements);
                //  std::cout << "✅ Loaded cached filter map for batch " << b + 1 << std::endl;
            }
            else
            {
                // Compute fresh
                std::cout << "💾 Saving " << start << " to cache" << std::endl;
                filter_ids_map.resize((end - start) * total_elements);

                for (size_t i = start; i < end; i++)
                {
                    const std::string &attribute = queries.second[i];
                    vector<string> query_attributes = split_and_clean(attribute);

                    ParallelFor(0, alg_query_aware->max_elements_, num_threads, [&](size_t row, size_t threadId)
                                {
                    if (row >= meta_data_attributes_cleaned.size()) {
                        std::cerr << "⚠️ meta_data_attributes size exceeded for row " << row << std::endl;
                        return;
                    }
                    //bool match_found = (attribute == meta_data_attributes[row]);
                    const auto &row_attributes = meta_data_attributes_cleaned[row];

                        bool match_found = false;

                        // Outer loop: each attribute in the query
                            for (const auto &q_attr : query_attributes)
                            {
                                // Inner loop: each attribute in the current metadata row
                                for (const auto &row_attr : row_attributes)
                                {
                                    if (q_attr == row_attr)
                                    {
                                        match_found = true;
                                        break; // Break inner loop
                                    }
                                }
                                if (match_found) break; // Break outer loop if a match was found
                            }

                   //  (attribute >= meta_data_attributes[row]);
                    filter_ids_map[(i - start) * total_elements + row] = match_found; });
                }

                saveFilterMap(filter_ids_map, filter_file);
                std::cout << "💾 Saved filter map for batch " << b + 1 << " to cache" << std::endl;
            }

            //  std::cout << "Predicate Set alignment (batch " << b + 1 << ")" << std::endl;

            // Step 2: Apply predicate filter
            alg_query_aware->predicateCondition(filter_ids_map.data());
            

            // Step 3: Run searches for this batch
            for (size_t i = 0; i < efs_array.size(); i++)
            {
                alg_query_aware->setEf(efs_array[i]);
                auto t1 = std::chrono::high_resolution_clock::now();
                // loop
                ParallelFor(0, end - start, 1, [&](size_t row_batch, size_t threadId)
                            {
                                size_t global_query_index = start + row_batch;
                                const std::vector<float> &emb = queries.first[global_query_index];

                                std::vector<float> new_query_reg = {avg_dis_selectivity[global_query_index].first,
                                                                    avg_dis_selectivity[global_query_index].second};

                                //float score = models[i]->predict(new_query_reg);
                                float score = 0.0f;
                                

                                std::pair<float, float> max_distance_and_normalized_score=std::make_pair(2.8f, score);
                                alg_query_aware->search(emb, k, queries.second[global_query_index], max_distance_and_normalized_score, row_batch, start, models[i].get());
                                //  alg_query_aware->searchingWithoutTree(emb, k, queries.second[global_query_index], score, row_batch, start, models[i].get());

                                // alg_query_aware->groundTruthBatchSelectivity(emb.data(), k, global_query_index,
                                //                                   filter_ids_map.data(), total_elements,
                                //                                   b, start);
                            });
                auto t2 = std::chrono::high_resolution_clock::now();
                auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

                std::cout << "Finished batch " << start
                          << " EFS: " << efs_array[i]
                          << " | Time: " << duration_ms << " ms" << std::endl;
                   // break;      
            }
          // break;
        }

        // Final cleanup

        delete alg_query_aware;
    }
}

// Read embeddings from CSV-like file for Point predicate
pair<vector<vector<float>>, vector<string>> reading_files(const string &file_path, int &dim, bool queries = false)
{
    cout << "Reading file: " << file_path << endl;
    ifstream file(file_path);
    vector<vector<float>> total_embeddings;
    vector<string> attributes; // store string IDs
    string line;
    getline(file, line); // skip header
    while (getline(file, line))
    {

        stringstream ss(line);

        string embedding, attribute, skip;
        if (!queries)
        {

            // getline(ss, skip, ';');
            getline(ss, embedding, ';');
            getline(ss, attribute, ';');
            //  getline(ss, embedding, ';');
        }
        else
        {
            // getline(ss, skip, ';');
            getline(ss, embedding, ';');
            getline(ss, attribute, ';');
        }
        if (!isNullOrEmpty(embedding))
        {
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

// Also make sure to remove the memory address occupied by char *after insertion.

// Function to split a string by a delimiter and convert to doubles
vector<float> splitToFloat(const string &str, char delimiter)
{
    vector<float> tokens;
    string token;
    istringstream tokenStream(str);
    while (getline(tokenStream, token, delimiter))
    {
        try
        {
            tokens.push_back(stof(token));
        }
        catch (const invalid_argument &e)
        {
            // Handle the case where conversion to double fails
            cerr << "Warning: Invalid float value encountered: " << token << endl;
        }
    }
    return tokens;
}

bool isNullOrEmpty(const string &str)
{
    return str.empty() || str == "null";
}

// Read embeddings from CSV-like file
vector<std::string> reading_files_attributes(const string &file_path)
{
    ifstream file(file_path);
    vector<std::string> attributes;
    string line;

    // Skip header
    getline(file, line);

    while (getline(file, line))
    {
        stringstream ss(line);
        string id, embedding, attribute; // audio, genreStr, publicationdate, viewsStr, likesStr, binNumberStr, rangebinStr, binNumberLike, binRangeLike;

        // Read required columns
        getline(ss, embedding, ';');
        getline(ss, attribute, ';');
        attributes.push_back(attribute);
    }

    return attributes;
}

// Define a variant to hold different types
std::unordered_map<std::string, std::string> reading_constants(std::string &path)
{
    std::unordered_map<std::string, std::string> constants; // Store values as strings
    std::ifstream file(path);

    if (!file)
    {
        std::cerr << "Error opening file!" << std::endl;
        return constants;
    }

    std::string key, value;

    while (file >> key >> value)
    {
        constants[key] = value;
    }

    file.close();

    return constants;
}
// Helper fucntion to trim whitespace
string trim(const string &s)
{
    size_t start = s.find_first_not_of(" \t\n\r");
    size_t end = s.find_last_not_of(" \t\n\r");
    return (start == string::npos) ? "" : s.substr(start, end - start + 1);
}
// Helper function to split and clean a string
vector<string> split_and_clean(const string &s)
{
    vector<string> result;
    stringstream ss(s);
    string token;

    while (getline(ss, token, ','))
    {
        // Trim whitespace
        token = trim(token);
        // Remove quotes
        token.erase(remove(token.begin(), token.end(), '"'), token.end());
        token.erase(remove(token.begin(), token.end(), '\''), token.end());
        // Convert to lowercase
        transform(token.begin(), token.end(), token.begin(), ::tolower);
        // Add if not empty
        if (!token.empty())
        {
            result.push_back(token);
        }
    }
    return result;
}

vector<vector<string>> reading_files_attributes_and_cleaning(const string &file_path)
{
    ifstream file(file_path);
    vector<vector<string>> attributes_all;
    string line;

    if (!file.is_open())
    {
        cerr << "Could not open file: " << file_path << endl;
        return attributes_all;
    }

    // Skip header
    getline(file, line);

    while (getline(file, line))
    {
        stringstream ss(line);
        string id, embedding, attribute;

        // Read required columnss
        getline(ss, embedding, ';'); // first column
        getline(ss, attribute, ';'); // attribute column

        // Split and clean attributes
        vector<string> cleaned_attributes = split_and_clean(attribute);

        // Add to final vector
        attributes_all.push_back(cleaned_attributes);
    }

    return attributes_all;
}

size_t getRandomIndex(size_t max_elements)
{
    static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, max_elements - 1);
    return dist(gen);
}