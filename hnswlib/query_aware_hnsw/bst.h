#pragma once
#include <iostream>
#include <memory>
#include <vector>
#include <stack>
#include "hnswlib/hnswlib.h"

class BST
{
private:
    struct Node
    {
        std::pair<float, size_t> smallest_score_id;
        // float score;                           // cumulative score
        // size_t smallest_score_id;
        size_t node_id;                        // unique node ID
        std::vector<size_t> nearest_nodes_ids; // nearest neighbors
        std::vector<float> embeddings;         // node embeddings
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;

        Node(std::pair<float, size_t> &smallest_id_distance_pair, size_t id,
             const std::vector<float> &node_embeddings,
             const std::vector<size_t> &nearest_nodes)
            : smallest_score_id(smallest_id_distance_pair), node_id(id),
              embeddings(node_embeddings),
              nearest_nodes_ids(nearest_nodes),
              left(nullptr), right(nullptr) {}
    };

    std::unique_ptr<Node> root;
    int total_item;
    hnswlib::HierarchicalNSW<float> *hnsw_BST;
    int testCounterNodes;

public:
    size_t next_tree_id; // new: reference to next tree in chain
    BST(hnswlib::HierarchicalNSW<float> *hnsw)
        : root(nullptr), total_item(0), testCounterNodes(0), hnsw_BST(hnsw), next_tree_id(0) {}

    BST(int dim)
        : root(nullptr), total_item(0), testCounterNodes(0), next_tree_id(0)
    {

        hnswlib::L2Space space(dim);
        hnsw_BST = new hnswlib::HierarchicalNSW<float>(&space, 2000);
    }

    int getTotalItems() const { return total_item; }

    float computeDistanceWithHNSW(const std::vector<float> &node_embedding,
                                  const std::vector<float> &key_embedding) const
    {
        // if (node_embedding.size() != key_embedding.size())
        // {
        //     std::cerr << "Error: vectors must have same size!" << std::endl;
        //     return -1;
        // }

        // float sum = 0.0f;
        // for (size_t i = 0; i < node_embedding.size(); i++)
        // {
        //     float diff = node_embedding[i] - key_embedding[i];
        //     sum += diff * diff;
        // }

        // // std::cout << "Sum of squared differences: " << sum << std::endl;

        // float dist = std::sqrt(sum);
        // std::cout << "L2 distance: " << dist << std::endl;

        //   std::cout<<"Computing distance between node and key embedding"<<node_embedding.size()<<"END ...."<<key_embedding.size()<< <<std::endl;
        return hnsw_BST->fstdistfunc_(node_embedding.data(),
                                      key_embedding.data(),
                                      hnsw_BST->dist_func_param_);
    }

    // Iterative insertion using cumulative score at each step
    Node *insert(std::pair<float, size_t> &smallest_score_id_pair, size_t id,
                 const std::vector<float> &node_embeddings,
                 const std::vector<size_t> &nearest_nodes)
    {

        // float cumulative_score = base_score; // It include recall/selectivity + closet distance
        if (!root)
        {
            root.reset(new Node(smallest_score_id_pair, id, node_embeddings, nearest_nodes));

            total_item++;
            return root.get();
        }

        Node *curr = root.get();
        Node *parent = nullptr;

        while (curr)
        {
            parent = curr;

            // cumulative score changes as we traverse
            // float cumulative_score = base_score + computeDistanceWithHNSW(node_embeddings, curr->embeddings);

            if (smallest_score_id_pair.first < curr->smallest_score_id.first)
            {
                if (!curr->left)
                {
                    curr->left.reset(new Node(smallest_score_id_pair, id, node_embeddings, nearest_nodes));
                    total_item++;
                    return curr->left.get();
                }
                curr = curr->left.get();
            }
            else if (smallest_score_id_pair.first > curr->smallest_score_id.first)
            {
                if (!curr->right)
                {
                    curr->right.reset(new Node(smallest_score_id_pair, id, node_embeddings, nearest_nodes));
                    total_item++;
                    return curr->right.get();
                }
                curr = curr->right.get();
            }
            else
            {
                // duplicate score, ignore insertion
                return nullptr;
            }
        }

        return nullptr;
    }

    // Iterative search/floor

    void printInOrder() const
    {
        std::stack<Node *> s;
        Node *curr = root.get();

        while (curr || !s.empty())
        {
            // Go as left as possible
            while (curr)
            {
                s.push(curr);
                curr = curr->left.get();
            }

            // Visit node
            curr = s.top();
            s.pop();

            curr = curr->right.get();
        }
    }

    void search(const std::vector<float> &query_embedding,
                std::pair<float, float> &max_distance_estimated_recall,
                std::pair<float, size_t> &smallest_score_id_,
                std::unordered_set<size_t> &visited_nodes)
    {
        if (!root)
            return;

        Node *curr = root.get();
        // float cumulative_dist = base_score;          // start with base_score
        float best_dist = std::numeric_limits<float>::max();
        size_t best_id = 0;

        while (curr)
        {
            visited_nodes.insert(curr->nearest_nodes_ids.begin(),
                                 curr->nearest_nodes_ids.end());

            // Distance to current node
            float dist = computeDistanceWithHNSW(query_embedding, curr->embeddings);
            // Ensure max_distance_estimated_recall.first is positive to avoid division by zero
            float max_dist = std::max(max_distance_estimated_recall.first, 1e-6f);

            // Normalize distance so that closer nodes give higher values
            float normalized_dist = 1.0f - (dist / max_dist);

            // Clamp to [0,1] for safety
            normalized_dist = std::max(0.0f, std::min(normalized_dist, 1.0f));


            // Now you can combine with recall (higher is better)
            float recall_score = max_distance_estimated_recall.second; // assumed in [0,1]
            float node_score = 0.5f * recall_score + 0.5f * normalized_dist;

            // std::cout << "Node ID: " << curr->node_id
            //           << " | Dist: " << dist
            //           << " | Recall: " << recall_score
            //           << " | Normalized Dist: " << normalized_dist
            //           << " | Score: " << node_score << std::endl;
            // Update best node if cumulative distance is smaller
            if (node_score < best_dist)
            {
                best_dist = node_score;
                best_id = curr->smallest_score_id.second;
            }

            // Decide traversal using cumulative distance vs node’s smallest_score_id
            // You can either compare using cumulative_dist or just dist for BST ordering
            if (node_score < curr->smallest_score_id.first)
            {
                curr = curr->left.get();
            }
            else if (node_score > curr->smallest_score_id.first)
            {
                curr = curr->right.get();
            }
            else
                break;
        }

        smallest_score_id_ = {best_dist, best_id};
    }

private:
    void printTree(const Node *node, const std::string &prefix, bool isLeft) const
    {
        if (!node)
            return;

        std::cout << prefix;

        // Print branch marker
        std::cout << (isLeft ? "├──" : "└──");

        // Print node details
        std::cout << "[Score: " << node->smallest_score_id.first
                  << ", ID: " << node->node_id << "]" << std::endl;

        // Recurse for children
        printTree(node->left.get(), prefix + (isLeft ? "│   " : "    "), true);
        printTree(node->right.get(), prefix + (isLeft ? "│   " : "    "), false);
    }

public:
    void printTree() const
    {
        if (!root)
        {
            std::cout << "(empty tree)" << std::endl;
            return;
        }
        std::cout << "[ROOT: Score: " << root->smallest_score_id.first
                  << ", ID: " << root->node_id << "]" << std::endl;

        printTree(root->left.get(), "", true);
        printTree(root->right.get(), "", false);
    }
};
