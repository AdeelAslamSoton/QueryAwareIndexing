// query_params.h
#pragma once
#include <string>

class QueryAttributes {
private:
    size_t k;             // number of neighbors
    float max_distance;    // example custom parameter
    std::string filter_tag;

public:
    // Constructor
    QueryAttributes(size_t k_=10, float max_distance_=1.0f, std::string filter_tag_="")
        : k(k_), max_distance(max_distance_), filter_tag(filter_tag_) {}

    // Getters
    size_t getK() const { return k; }
    float getMaxDistance() const { return max_distance; }
    std::string getFilterTag() const { return filter_tag; }

    // Setters
    void setK(size_t k_) { k = k_; }
    void setMaxDistance(float d) { max_distance = d; }
    void setFilterTag(const std::string& tag) { filter_tag = tag; }
};
