#pragma once
#include <map>
#include <memory> // for std::unique_ptr
#include <string> // for std::string
#include <vector>
#include <mutex>
#include <atomic>
#include <random>
#include <fstream>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <shared_mutex>
#include "query_property.h"
#include "bst.h"
#include "hnswlib/hnswlib.h"
#include "hnswlib/visited_list_pool.h"
#include "query_aware_hnsw/linear_regression.h"
namespace qwery_aware
{
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template <typename dist_t>
    class QweryAwareHNSWRange : public hnswlib::HierarchicalNSW<dist_t>
    {

        // Comparator
        struct CompareByFirstElement
        {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept
            {
                return a.first < b.first;
            }
        };
        using Candidate = std::pair<dist_t, tableint>;
        using CandidateQueue = std::priority_queue<Candidate, std::vector<Candidate>, CompareByFirstElement>;
        // struct for Equi-bins

    private:
        int queryLimit; // private variable
        // Custom member variable
        QueryAttributes params;
        int query_count;
        std::unique_ptr<hnswlib::VisitedListPool> visited_list_pool_{nullptr};
        mutable std::atomic<long> metric_distance_computations{0};
        mutable std::atomic<long> metric_hops{0};
        char *filter_id_map;
        size_t total_vectors;

        std::map<int, std::unordered_set<int>> gts_ids; //= groundTruthIds("/data4/hnsw/TripClick/QueriesForQueriesAware/GroundTruth/", 10);

        std::vector<std::pair<float, float>> avg_dis_selectivity; //= readDistanceSelectivity("/data4/hnsw/TripClick/QueriesForQueriesAware/Selectivity/distance_selectivity_.csv");
        const std::map<int, pair<float, float>> *mapWithBin;
        std::unordered_map<std::string, std::string> constants;

    public:
        // Constructor: initialize both parent and child with maximum element

        QweryAwareHNSWRange(hnswlib::SpaceInterface<dist_t> *space,
                            size_t max_elements, std::map<int, std::unordered_set<int>> &gts_ids_, std::vector<std::pair<float, float>> &avg_dis_selectivity_, const std::map<int, pair<float, float>> *mapWithBin_, std::unordered_map<std::string, std::string> &constants_)
            : hnswlib::HierarchicalNSW<dist_t>(space, max_elements) // parent constructor
                                                                    // child constructor
        {
            // You can add extra initialization logic here
            query_count = 0;
            visited_list_pool_ = std::unique_ptr<hnswlib::VisitedListPool>(new hnswlib::VisitedListPool(1, max_elements));
            total_vectors = max_elements;
            gts_ids = gts_ids_;
            avg_dis_selectivity = avg_dis_selectivity_;
            mapWithBin = mapWithBin_;
            constants = constants_;
        }

        QweryAwareHNSWRange(
            hnswlib::SpaceInterface<dist_t> *space,
            size_t max_elements)
            : hnswlib::HierarchicalNSW<dist_t>(space, max_elements)
        {
            query_count = 0;
            visited_list_pool_ = std::unique_ptr<hnswlib::VisitedListPool>(new hnswlib::VisitedListPool(1, max_elements));
            total_vectors = max_elements;

            // gts_ids and avg_dis_selectivity remain empty
        }

        // Compute the result
        // This map is a key value pair for attribute and its associated binary tree for search
        // Map from attribute name to its BST
        std::map<int, std::unique_ptr<BST>> attribute_tree_mapping;
        // Map from attribute name to its mutex for thread-safe operations
        std::map<int, std::mutex> attribute_mutex_map;
        // Global mutex for creation of new BST and mutex
        std::mutex creation_mutex;

        /* Initializes a BST for each bin range.
         * Creates a new BST and mutex per bin to manage hybrid queries.
         * Ensures thread-safe mapping of bins to their index trees.
         */

        void bst_initialization(const std::map<int, std::pair<float, float>> &mapWithBin)
        {
            for (const auto &entry : mapWithBin)
            {
                int bin = entry.first; // fetch the int value (bin)

                // Initialize BST for this bin
                std::unique_ptr<BST> new_bst(new BST(static_cast<hnswlib::HierarchicalNSW<float> *>(this)));
                attribute_tree_mapping[bin] = std::move(new_bst);

                // Initialize mutex for this bin
                attribute_mutex_map[bin]; // default-constructs the mutex if not exists
            }
        }

        void search(const std::vector<float> &embeddings, size_t k, vector<int> &bins,  std::pair<float,float> &max_distance_estimated_recall, size_t &query_num, size_t batch_start, LinearRegression *linear_reg = nullptr)
        {

            std::vector<BST *> bst_list;
            std::unordered_set<size_t> visited_nodes;
            std::pair<float, size_t> entrypoint_node;

            for (int bin : bins)
            {
                auto it = attribute_tree_mapping.find(bin);
                if (it != attribute_tree_mapping.end()) // <-- check it exists
                {
                    BST *bst = it->second.get();
                    if (bst)
                    {
                        bst->search(embeddings, max_distance_estimated_recall, entrypoint_node, visited_nodes);
                    }
                    if (visited_nodes.size() >= 100 || visited_nodes.size() == 0)
                        break;
                }
            }

            if (!visited_nodes.empty())
            {
                //  std::cout<<"visted"<<visited_nodes.size()<<std::endl;
                std::vector<size_t> touching_ids;
                auto top_candidates = searchBaseLayerTwoHop(entrypoint_node.second, embeddings.data(), this->ef_, query_num, &visited_nodes, false, touching_ids);

                // auto top_candidates = searchBaseLayer(entrypoint_node.second, embeddings.data(), this->ef_, query_num, &visited_nodes, false, touching_ids);
                std::priority_queue<std::pair<dist_t, size_t>> result;

                while (!top_candidates.empty())
                {
                    auto [dist, id] = top_candidates.top();
                    top_candidates.pop();
                    result.emplace(dist, static_cast<size_t>(id));
                }

                std::pair<float, size_t> dist_id_pair_;
                std::vector<std::pair<dist_t, size_t>> tmp;
                auto nearest_nodes_array = getNearestNodes(&result, k, tmp, &dist_id_pair_);

                nearest_nodes_array.insert(
                    nearest_nodes_array.end(),
                    touching_ids.begin(),
                    touching_ids.end());

                if (!nearest_nodes_array.empty())
                {

                    std::vector<float> avg_dist_sel(2);
                    int id = query_num + batch_start;

                    // Count matches with ground truth
                    const auto &gt_set = gts_ids[id];
                    float match_count = 0.0f;
                    for (const auto &p : tmp)
                    {
                        if (gt_set.count(p.second))
                        {
                            match_count++;
                        }
                    }

                    // Compute recall
                    match_count = match_count / k;

                    // Fill the reusable vector for linear regression
                    avg_dist_sel[0] = avg_dis_selectivity[id].first;
                    avg_dist_sel[1] = avg_dis_selectivity[id].second;

                    linear_reg->update(avg_dist_sel, match_count);
                    dist_id_pair_.first += match_count;

                    // Insert into BST under lock

                    for (int bin : bins)
                    {
                        auto it = attribute_tree_mapping.find(bin);
                        if (it != attribute_tree_mapping.end())
                        {
                            BST *bst_ = it->second.get();
                            if (bst_)
                            {
                                std::lock_guard<std::mutex> lock(attribute_mutex_map[bin]);
                                bst_->insert(dist_id_pair_, id, embeddings, nearest_nodes_array);
                            }
                        }
                        break;
                    }
                }

                std::string dir =
                    constants["RESULT_FOLDER"] + std::to_string(this->ef_);

                create_directory_if_not_exists(dir);

                std::string filename =
                    dir + "/Q" + std::to_string(query_num + batch_start) + ".csv";

                //  std::cout<<  batch_start + query_num << "," << tmp.size() << "\n";
                std::ofstream csv_file(filename, std::ios::app);
                if (csv_file.is_open())
                {
                    size_t considered = std::min(tmp.size(), k);
                    csv_file << "ID,Distance\n";

                    if (considered > 0)
                    {
                        // Write actual results
                        for (size_t i = 0; i < considered; ++i)
                        {
                            csv_file << tmp[i].second << "," << tmp[i].first << "\n";
                        }
                    }
                    else
                    {
                        // No results → write k dummy rows
                        for (size_t i = 0; i < k; ++i)
                        {
                            csv_file << -1 << "," << std::numeric_limits<float>::max() << "\n";
                        }
                    }

                    csv_file.flush();
                    csv_file.close();
                }
                else
                {
                    std::cerr << "Error: could not open results.csv for writing\n";
                }
            }

            else
            {

                handleColdStartInsertion(embeddings, k,  max_distance_estimated_recall, query_num, batch_start, bins, linear_reg);
            }
        }

        // Returning the nearest neighbor node IDs for a query, limited to top-K results.
        // Optionally, also update the distance value of the closest (top-1) neighbor.
        /**
         * @param results is the priority queue from the result set
         * @param k is the top-K item that needs to be returned
         * @param distance_for_top_1 is the closed distance between the query and returned points
         */
        std::vector<size_t> getNearestNodes(
            std::priority_queue<std::pair<dist_t, size_t>> *results, // pointer to the priority queue
            size_t k, std::vector<std::pair<dist_t, size_t>> &tmp,
            std::pair<float, size_t> *dist_id_pair)
        {
            // Copy all results into a vector

            tmp.reserve(results->size());

            while (!results->empty())
            {
                tmp.push_back(results->top());
                results->pop();
            }

            // Sort vector by ascending distance
            std::sort(tmp.begin(), tmp.end(),
                      [](const std::pair<dist_t, size_t> &a, const std::pair<dist_t, size_t> &b)
                      { return a.first < b.first; });

            // Prepare the nearest nodes array
            std::vector<size_t> nearest_nodes_array;
            nearest_nodes_array.reserve(std::min(k, tmp.size()));

            dist_t smallest_distance = std::numeric_limits<dist_t>::max();
            size_t best_id = std::numeric_limits<size_t>::max();
            size_t limit = std::min<size_t>(k, tmp.size());
            for (size_t i = 0; i < tmp.size(); i++) // previously it was limit for low selectivity
            {
                nearest_nodes_array.push_back(tmp[i].second);
            }

            // Fill dist_id_pair with the closest neighbor info
            if (!nearest_nodes_array.empty() && dist_id_pair)
            {
                *dist_id_pair = {static_cast<float>(tmp[1].first), tmp[1].second};
            }

            return nearest_nodes_array;
        }

        /**
         * @param ep_id   Entry point internal node ID.
         * @param data_point   Query vector data pointer.
         * @param ef   Search depth / size of dynamic candidate list.
         * @param visited_nodes_by_previous_queries   Optional visited array reused across queries.
         * @param query_number it is the query number
         * @param isIdAllowed   Optional filter functor for allowed node IDs.
         * @param stop_condition   Optional stopping condition functor for early termination.
         */

        template <bool bare_bone_search = true, bool collect_metrics = false>
        CandidateQueue searchBaseLayer(tableint ep_id, const void *data_point, size_t ef, size_t query_number, std::unordered_set<size_t> *visited_nodes_by_previous_queries = nullptr, bool cold_start = false, std::vector<size_t> &touching_ids = {})
        {

            hnswlib::VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            hnswlib::vl_type *visited_array = vl->mass;
            hnswlib::vl_type visited_array_tag = vl->curV;
            CandidateQueue top_candidates;
            CandidateQueue candidate_set;
            dist_t lowerBound;

            char *ep_data = this->getDataByInternalId(ep_id);
            dist_t dist = this->fstdistfunc_(data_point, ep_data, this->dist_func_param_);
            lowerBound = dist;
            //  top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);

            tableint node_id = ep_id;

            if ((cold_start && filter_id_map[query_number * total_vectors + node_id]) || !cold_start)
            {
                top_candidates.emplace(dist, ep_id);
                visited_array[ep_id] = visited_array_tag;
                twoHopSearch(data_point, node_id, query_number, top_candidates, candidate_set,
                             lowerBound, ef, visited_array, visited_array_tag, touching_ids);
            }

            while (!candidate_set.empty())
            {

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                dist_t candidate_dist = -current_node_pair.first;

                bool flag_stop_search;
                if (bare_bone_search)
                {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
                if (flag_stop_search)
                {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *)this->get_linklist0(current_node_id);
                size_t size = this->getListCount((linklistsizeint *)data);
                //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                if (collect_metrics)
                {
                    metric_hops++;
                    metric_distance_computations += size;
                }

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(this->data_level0_memory_ + (*(data + 1)) * this->size_data_per_element_ + this->offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++)
                {
                    int candidate_id = *(data + j);
                    if (visited_nodes_by_previous_queries && (visited_nodes_by_previous_queries->find(candidate_id) != visited_nodes_by_previous_queries->end()))
                    {
                        tableint candidate_node_id = candidate_id; // lvalue

                        twoHopSearch(data_point, candidate_node_id, query_number, top_candidates, candidate_set, lowerBound, ef, visited_array, visited_array_tag, touching_ids);
                        continue;
                    }
                    //   std::cout<<"Inside the While Loop"<<std::endl;

#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(this->data_level0_memory_ + (*(data + j + 1)) * this->size_data_per_element_ + this->offsetData_,
                                 _MM_HINT_T0); ////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag) && filter_id_map[query_number * total_vectors + candidate_id])
                    {

                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (this->getDataByInternalId(candidate_id));

                        dist_t dist = this->fstdistfunc_(data_point, currObj1, this->dist_func_param_);

                        // Load the current value

                        bool flag_consider_candidate;

                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;

                        if (flag_consider_candidate)
                        {
                            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(this->data_level0_memory_ + candidate_set.top().second * this->size_data_per_element_ +
                                             this->offsetLevel0_, ///////////
                                         _MM_HINT_T0);            ////////////////////////
#endif

                            if (bare_bone_search)
                            {
                                top_candidates.emplace(dist, candidate_id);
                                // first check the un ordered set has candidate id then
                            }

                            bool flag_remove_extra = false;

                            flag_remove_extra = top_candidates.size() > ef;

                            while (flag_remove_extra)
                            {
                                //  tableint id = top_candidates.top().second;
                                top_candidates.pop();

                                flag_remove_extra = top_candidates.size() > ef;
                            }

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                        if (cold_start == true)
                        {
                            tableint candidate_node_id = candidate_id; // lvalue
                            twoHopSearch(data_point, candidate_node_id, query_number, top_candidates, candidate_set, lowerBound, ef, visited_array, visited_array_tag, touching_ids);
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }
        // Call this method from the main function for item presence check it can be simple.
        void predicateCondition(char *filters_array)
        {
            filter_id_map = filters_array;
        }

        // fill the

        /**
         * @brief Inserts a query embedding during cold-start.
         *
         * Performs a KNN search to find nearest nodes, computes a cumulative score,
         * and inserts the embedding into the BST for future retrieval.
         *
         * @param embeddings  The query embedding vector.
         * @param k           Number of nearest neighbors to consider.
         * @param score       Reference to the score, updated cumulatively.
         * @param query_num   Index of the query.
         */

        void handleColdStartInsertion(const std::vector<float> &embeddings, size_t k,  std::pair<float, float> &max_distance_estimated_recall, int query_num, size_t batch_start, std::vector<int> &bins, LinearRegression *linear_reg = nullptr)
        {

            auto result = this->searchKnn(embeddings.data(), k);

            std::priority_queue<std::pair<dist_t, size_t>> results;

            // Iterate over original results
            while (!result.empty())
            {
                auto top = result.top(); // by value
                size_t id = top.second;

                // Apply your predicate
                if (filter_id_map[query_num * total_vectors + id])
                {
                    results.push(top);
                }

                result.pop();
            }

            // auto results = coldStartKnn(embeddings.data(), this->ef_, k, query_num);

            // auto results = coldStartPreFiltering(embeddings.data(), this->ef_, k, query_num);

            std::string dir =
                constants["RESULT_FOLDER"] +
                std::to_string(this->ef_);
            create_directory_if_not_exists(dir);

            std::string filename =
                dir + "/Q" + std::to_string(query_num + batch_start) + ".csv";
            std::ofstream csv_file(filename, std::ios::app);

            if (results.size() > 0)
            {
                // Step 1: Find nearest nodes
                std::pair<float, size_t> dist_id_pair;
                std::vector<std::pair<dist_t, size_t>> tmp;
                auto nearest_nodes_array =
                    getNearestNodes(&results, k, tmp, &dist_id_pair);

                if (nearest_nodes_array.size() > 0)
                {
                    std::vector<float> avg_dist_sel(2);
                    int id = query_num + batch_start;

                    // Step 2: Compute recall (ground truth comparison)
                    const auto &gt_set = gts_ids[id];
                    float match_count = 0.0f;
                    for (const auto &p : tmp)
                    {
                        if (gt_set.count(p.second))
                        {
                            match_count++;
                        }
                    }
                    match_count /= k;

                    // Step 3: Update linear regression
                    avg_dist_sel[0] = avg_dis_selectivity[id].first;
                    avg_dist_sel[1] = avg_dis_selectivity[id].second;
                    linear_reg->update(avg_dist_sel, match_count);
                    dist_id_pair.first += match_count;

                    // Step 4: Insert into BST
                    for (int bin : bins)
                    {
                        auto it = attribute_tree_mapping.find(bin);
                        if (it != attribute_tree_mapping.end())
                        {
                            BST *bst_ = it->second.get();
                            if (bst_)
                            {
                                std::lock_guard<std::mutex> lock(attribute_mutex_map[bin]);
                                bst_->insert(dist_id_pair, id, embeddings, nearest_nodes_array);
                            }
                        }
                        // break;
                    }
                }

                // Step 5: Write results to CSV
                if (csv_file.is_open())
                {
                    size_t limit = std::min<size_t>(k, tmp.size());
                    csv_file << "ID,Distance\n";
                    for (size_t i = 0; i < limit; ++i)
                    {
                        csv_file << tmp[i].second << "," << tmp[i].first << "\n";
                    }
                    csv_file.flush();
                    csv_file.close();
                }
                else
                {
                    std::cerr << "Error: could not open results.csv for writing\n";
                }
            }
            else
            {
                // Fallback branch: no results found → write 10 dummy rows
                if (csv_file.is_open())
                {
                    csv_file << "ID,Distance\n";
                    for (int i = 0; i < 10; i++)
                    {
                        csv_file << -1 << "," << std::numeric_limits<float>::max() << "\n";
                    }
                    csv_file.flush();
                    csv_file.close();
                }
                else
                {
                    std::cerr << "Error: could not open results.csv for writing (empty case)\n";
                }
            }
        }

        /**
         * @brief Performs a two-hop neighbor search from a given node.
         *
         * @param query_data        Pointer to the query vector.
         * @param node              Current node (entry point).
         * @param query_number      Query index for filtering.
         * @param top_candidates    Priority queue of best candidates.
         * @param candidate_set     Priority queue of candidates to explore.
         * @param lowerBound        Current search bound.
         * @param ef                Candidate list size.
         * @param visited_array     Visited nodes tracker.
         * @param visited_array_tag Marker for visited nodes.
         */

        void twoHopSearch(const void *query_data, tableint node, size_t &query_number, CandidateQueue &top_candidates, CandidateQueue &candidate_set, dist_t &lowerBound, size_t &ef, hnswlib::vl_type *visited_array = nullptr, hnswlib::vl_type visited_array_tag = 0, std::vector<size_t> &touching_ids = {})
        {
            int *data;
            data = (int *)this->get_linklist0(node);
            size_t size = this->getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);

//             // Getting the nearest neighbours size
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);

#endif

            for (size_t j = 0; j < size; j++)
            {
                tableint candidate_id = *(datal + j);

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
#endif

                // bool is_visited = visited_bitmap[byte_index_cand] & (1 << bit_index_cand);
                if (visited_array[candidate_id] == visited_array_tag)
                    continue;
                visited_array[candidate_id] = visited_array_tag;
                // visited_bitmap[byte_index_cand] |= (1 << bit_index_cand);

                if (filter_id_map[query_number * total_vectors + candidate_id]) // Check if the item satisfy the predicate

                {
                    // std::cout<<"I am here---Searching"<<candidate_id<<std::endl;
                    char *currObj1 = (this->getDataByInternalId(candidate_id));

                    dist_t dist = this->fstdistfunc_(query_data, currObj1, this->dist_func_param_);

                    updatePriorityQueue(candidate_id, dist, top_candidates, candidate_set, lowerBound, ef, touching_ids);
                }
                // Two hop searching
                int *twoHopData = (int *)this->get_linklist0(candidate_id);
                if (!twoHopData)
                    continue; // Error handling

                size_t twoHopSize = this->getListCount((linklistsizeint *)twoHopData);
                tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                for (size_t k = 0; k < twoHopSize; k++)
                {
                    tableint candidateIdTwoHop = *(twoHopDatal + k);

#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(twoHopDatal + k + 1)), _MM_HINT_T0);
#endif

                    // size_t byte_index_two_hop_cand = candidateIdTwoHop / 8;
                    // size_t bit_index_two_hop_cand = candidateIdTwoHop % 8;
                    // bool is_visited = visited_bitmap[byte_index_two_hop_cand] & (1 << bit_index_two_hop_cand);
                    if (visited_array[candidateIdTwoHop] == visited_array_tag)
                        continue;
                    visited_array[candidateIdTwoHop] = visited_array_tag;
                    // visited_bitmap[byte_index_two_hop_cand] |= (1 << bit_index_two_hop_cand);

                    if (filter_id_map[query_number * total_vectors + candidateIdTwoHop]) // Check if the item satisfy the predicate
                    {

                        char *currObj1 = (this->getDataByInternalId(candidateIdTwoHop));

                        // dist_t dist1 = this->fstdistfunc_(query_data, currObj1, this->dist_func_param_);

                        dist_t dist1 = this->fstdistfunc_(query_data, currObj1, this->dist_func_param_);
                        updatePriorityQueue(candidateIdTwoHop, dist1, top_candidates, candidate_set, lowerBound, ef, touching_ids);

                        //  result_vector.push_back(std::make_pair(candidateIdTwoHop, dist1)); call methods
                    }
                }
            }

            // visited_list_pool_->releaseVisitedList(vl);
        }

        // One hop search for NAvix
        void oneHopSearch(const void *query_data, tableint node, size_t &query_number, CandidateQueue &top_candidates, CandidateQueue &candidate_set, dist_t &lowerBound, size_t &ef, hnswlib::vl_type *visited_array = nullptr, hnswlib::vl_type visited_array_tag = 0, std::vector<size_t> &pre_items = {})
        {
            int *data;
            data = (int *)this->get_linklist0(node);
            size_t size = this->getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);

//             // Getting the nearest neighbours size
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);

#endif

            for (size_t j = 0; j < size; j++)
            {
                tableint candidate_id = *(datal + j);

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
#endif

                // bool is_visited = visited_bitmap[byte_index_cand] & (1 << bit_index_cand);
                if (visited_array[candidate_id] == visited_array_tag)
                    continue;
                visited_array[candidate_id] = visited_array_tag;
                // visited_bitmap[byte_index_cand] |= (1 << bit_index_cand);

                if (filter_id_map[query_number * total_vectors + candidate_id]) // Check if the item satisfy the predicate

                {
                    // std::cout<<"I am here---Searching"<<candidate_id<<std::endl;
                    char *currObj1 = (this->getDataByInternalId(candidate_id));

                    dist_t dist = this->fstdistfunc_(query_data, currObj1, this->dist_func_param_);

                    updatePriorityQueue(candidate_id, dist, top_candidates, candidate_set, lowerBound, ef, pre_items);
                }
                // Two hop searching
            }

            // visited_list_pool_->releaseVisitedList(vl);
        }
        // Directed Two-hop Search

        void directedTwoHopSearch(const void *query_data, tableint node, size_t &query_number, CandidateQueue &top_candidates, CandidateQueue &candidate_set, dist_t &lowerBound, size_t &ef, hnswlib::vl_type *visited_array = nullptr, hnswlib::vl_type visited_array_tag = 0, std::vector<size_t> &filtered_nodes = {})
        {
            int *data;
            data = (int *)this->get_linklist0(node);
            size_t size = this->getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);

//             // Getting the nearest neighbours size
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);

#endif

            for (size_t j = 0; j < size; j++)
            {
                tableint candidate_id = *(datal + j);

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
#endif

                // bool is_visited = visited_bitmap[byte_index_cand] & (1 << bit_index_cand);
                if (visited_array[candidate_id] == visited_array_tag)
                    continue;
                visited_array[candidate_id] = visited_array_tag;
                // visited_bitmap[byte_index_cand] |= (1 << bit_index_cand);

                if (filter_id_map[query_number * total_vectors + candidate_id]) // Check if the item satisfy the predicate

                {
                    // std::cout<<"I am here---Searching"<<candidate_id<<std::endl;
                    char *currObj1 = (this->getDataByInternalId(candidate_id));

                    dist_t dist = this->fstdistfunc_(query_data, currObj1, this->dist_func_param_);

                    updatePriorityQueue(candidate_id, dist, top_candidates, candidate_set, lowerBound, ef, filtered_nodes);

                    // Two hop searching
                    int *twoHopData = (int *)this->get_linklist0(candidate_id);
                    if (!twoHopData)
                        continue; // Error handling

                    size_t twoHopSize = this->getListCount((linklistsizeint *)twoHopData);
                    tableint *twoHopDatal = (tableint *)(twoHopData + 1);
                    for (size_t k = 0; k < twoHopSize; k++)
                    {
                        tableint candidateIdTwoHop = *(twoHopDatal + k);

#ifdef USE_SSE
                        _mm_prefetch((char *)(visited_array + *(twoHopDatal + k + 1)), _MM_HINT_T0);
#endif

                        if (visited_array[candidateIdTwoHop] == visited_array_tag)
                            continue;
                        visited_array[candidateIdTwoHop] = visited_array_tag;

                        if (filter_id_map[query_number * total_vectors + candidateIdTwoHop]) // Check if the item satisfy the predicate
                        {

                            char *currObj1 = (this->getDataByInternalId(candidateIdTwoHop));
                            dist_t dist1 = this->fstdistfunc_(query_data, currObj1, this->dist_func_param_);
                            updatePriorityQueue(candidateIdTwoHop, dist1, top_candidates, candidate_set, lowerBound, ef, filtered_nodes);
                        }
                    }
                }
            }

            // visited_list_pool_->releaseVisitedList(vl);
        }

        /**
         * @brief Updates the candidate and top priority queues with a new node.
         *
         * @param candidate_id     ID of the candidate node.
         * @param dist             Distance of the candidate to the query.
         * @param top_candidates   Priority queue of current best candidates.
         * @param candidate_set    Priority queue of nodes to explore.
         * @param lowerBound       Current search bound, updated if needed.
         * @param ef               Maximum size of the top_candidates queue.
         */

        void updatePriorityQueue(size_t candidate_id, dist_t dist, CandidateQueue &top_candidates, CandidateQueue &candidate_set, dist_t &lowerBound, size_t ef, std::vector<size_t> &touching_ids = {})
        {
            // Decide whether to consider candidate
            bool flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
            if (!flag_consider_candidate)
                return;

            // Always insert into candidate set
            candidate_set.emplace(-dist, candidate_id);

#ifdef USE_SSE
            _mm_prefetch(this->data_level0_memory_ + candidate_set.top().second * this->size_data_per_element_ +
                             this->offsetLevel0_,
                         _MM_HINT_T0);
#endif

            // Always insert into top_candidates since bare_bone_search = true
            top_candidates.emplace(dist, candidate_id);

            // Remove extra if needed
            while (top_candidates.size() > ef)
            {
                auto removed = top_candidates.top();    // pair<dist, id>
                touching_ids.push_back(removed.second); // store the id
                top_candidates.pop();
            }

            if (!top_candidates.empty())
                lowerBound = top_candidates.top().first;
        }

        std::priority_queue<std::pair<dist_t, size_t>> coldStartKnn(const void *query_data, size_t ef, size_t k, size_t query_num)
        {
            std::priority_queue<std::pair<dist_t, size_t>> result;
            tableint currObj = this->enterpoint_node_;

            dist_t curdist = this->fstdistfunc_(query_data, this->getDataByInternalId(this->enterpoint_node_), this->dist_func_param_);
            for (int level = this->maxlevel_; level > 0; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *)this->get_linklist(currObj, level);
                    int size = this->getListCount(data);

                    tableint *datal = (tableint *)(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > total_vectors)
                            throw std::runtime_error("cand error");

                        if (!filter_id_map[query_num * total_vectors + cand])
                        {
                            continue;
                        }

                        dist_t d = this->fstdistfunc_(query_data, this->getDataByInternalId(cand), this->dist_func_param_);

                        if (d < curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            auto top_candidates = searchBaseLayer(currObj, query_data, ef, query_num, nullptr, true);

            while (top_candidates.size() > k)
            {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0)
            {
                std::pair<dist_t, tableint> rez = top_candidates.top();

                result.push(std::pair<dist_t, size_t>(rez.first, this->getExternalLabel(rez.second)));
                top_candidates.pop();
            }

            return result;
        }

        std::priority_queue<std::pair<dist_t, size_t>>
        coldStartPreFiltering(const void *query_data, size_t ef, size_t k, size_t query_num)
        {
            std::priority_queue<std::pair<dist_t, size_t>> top_candidates;

            for (tableint i = 0; i < total_vectors; i++)
            {
                char *ep_data = this->getDataByInternalId(i);

                // apply filter
                if (filter_id_map[query_num * total_vectors + i])
                {
                    dist_t dist = this->fstdistfunc_(query_data, ep_data, this->dist_func_param_);
                    top_candidates.emplace(dist, i);

                    // keep heap size under ef
                    if (top_candidates.size() > ef)
                    {
                        top_candidates.pop();
                    }
                }
            }

            return top_candidates;
        }

        void groundTruthBatch(const void *query_data, size_t k, size_t query_num_batch,
                              char *filter_ids_map, size_t total_elements, size_t batch_num, size_t batch_start)
        {

            std::priority_queue<std::pair<dist_t, size_t>> top_candidates;

            for (tableint i = 0; i < total_vectors; i++)
            {
                // std::cout<<"I am executing"<<total_vectors<<std::endl;
                char *ep_data = this->getDataByInternalId(i);

                // apply batch-local filter
                if (filter_ids_map[query_num_batch * total_elements + i])
                {

                    dist_t dist = this->fstdistfunc_(query_data, ep_data, this->dist_func_param_);
                    top_candidates.emplace(dist, i);

                    if (top_candidates.size() > k)
                        top_candidates.pop();
                }
            }
            // std::cout<<"I am executing..."<<total_vectors<<std::endl;

            // Extract results in ascending distance
            std::vector<std::pair<dist_t, size_t>> results;
            while (!top_candidates.empty())
            {
                results.push_back(top_candidates.top());
                top_candidates.pop();
            }
            std::reverse(results.begin(), results.end());

            // Write CSV with global query index
            std::string filename = "/data3/Adeel/Point_queries/UNG/GT/Q" +
                                   std::to_string(batch_start + query_num_batch) + ".csv";
            std::ofstream out(filename);
            if (!out.is_open())
            {
                std::cerr << "❌ Failed to open file: " << filename << std::endl;
                return;
            }

            out << "ID,Distance\n";
            if (!results.empty())
            {
                for (const auto &p : results)
                    out << p.second << "," << p.first << "\n";
            }
            else
            {
                // 🔹 Fallback: no results → write k rows of dummy values
                for (size_t i = 0; i < k; i++)
                    out << -1 << "," << std::numeric_limits<float>::max() << "\n";
            }

            out.close();
            std::cout << "✅ Saved results for query " << (batch_start + query_num_batch) << " to " << filename << std::endl;
        }

        template <bool bare_bone_search = true, bool collect_metrics = false>
        CandidateQueue searchBaseLayerTwoHop(tableint ep_id, const void *data_point, size_t ef, size_t query_number, std::unordered_set<size_t> *visited_nodes_by_previous_queries = nullptr, bool cold_start = false, std::vector<size_t> &touching_ids = {})
        {

            hnswlib::VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            hnswlib::vl_type *visited_array = vl->mass;
            hnswlib::vl_type visited_array_tag = vl->curV;
            CandidateQueue top_candidates;
            CandidateQueue candidate_set;
            dist_t lowerBound;

            std::vector<size_t> visited_sorted(visited_nodes_by_previous_queries->begin(),
                                               visited_nodes_by_previous_queries->end());
            std::sort(visited_sorted.begin(), visited_sorted.end());

            for (size_t id : visited_sorted)
            {

                if (filter_id_map[query_number * total_vectors + id])
                {

                    char *ep_data = this->getDataByInternalId(id);
                    dist_t dist = this->fstdistfunc_(data_point, ep_data, this->dist_func_param_);
                    candidate_set.emplace(-dist, id);
                    top_candidates.emplace(dist, id);
                    visited_array[id] = visited_array_tag;
                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
            while (!candidate_set.empty())
            {

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                dist_t candidate_dist = -current_node_pair.first;

                bool flag_stop_search;
                if (bare_bone_search)
                {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
                if (flag_stop_search)
                {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;

                // Computing the local selectivity

                twoHopSearch(data_point, current_node_id, query_number, top_candidates, candidate_set,
                             lowerBound, ef, visited_array, visited_array_tag, touching_ids);
                // oneHopSearch(data_point, current_node_id, query_number, top_candidates, candidate_set,
                //              lowerBound, ef, visited_array, visited_array_tag, touching_ids);
                // directedTwoHopSearch(data_point, current_node_id, query_number, top_candidates, candidate_set,
                //                      lowerBound, ef, visited_array, visited_array_tag, touching_ids);
            }

            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

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

        vector<int> findBinsForRange(int &query_left, int &query_right) const
        {
            vector<int> bins;
            for (const auto &entry : *mapWithBin)
            {
                int bin_left = entry.second.first;
                int bin_right = entry.second.second;

                // Check for overlap
                if (query_right > bin_left && query_left < bin_right)
                {
                    bins.push_back(entry.first);
                }
            }
            return bins;
        }
    };
}
