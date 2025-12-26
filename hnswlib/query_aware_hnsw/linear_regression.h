#pragma once
#include <iostream>
#include <mutex>
#include <vector>
using namespace std;

class LinearRegression
{
private:
    vector<float> theta;    // model parameters (bias + weights)
    double alpha;           // learning rate
    mutable std::mutex mtx; // mutex to protect theta

public:
    // Constructor
    LinearRegression(int num_features, float learning_rate = 0.01f)
    {
        theta = vector<float>(num_features + 1, 0.0); // +1 for bias
        alpha = learning_rate;
    }

    // Predict recall for given features
    double predict(const vector<float> &x) const
    {
        float y = theta[0]; // bias
        for (size_t i = 0; i < x.size(); ++i)
            y += theta[i + 1] * x[i];
        return y;
    }

    // Update model parameters using one query (online learning) x is the attributes
    void update(const std::vector<float> &x, float y_true)
    {
        // lock during update
        double y_pred = predict(x);
        double error = y_pred - y_true;

        // Use a small learning rate to avoid exploding updates
        double effective_alpha = alpha / (1.0 + x.size()); // scale by number of features

        // Update bias
        {
            std::lock_guard<std::mutex> lock(mtx);
            theta[0] -= effective_alpha * error;

            // Update weights
            for (size_t i = 0; i < x.size(); ++i)
            {
                // Scale the update by a factor to prevent huge jumps
                double delta = effective_alpha * error * x[i];

                // Optional: clip delta to avoid overflow
                if (delta > 1e3)
                    delta = 1e3;
                else if (delta < -1e3)
                    delta = -1e3;

                theta[i + 1] -= delta;
            }
        }
    }

    // Print current theta
    void printTheta() const
    {
        cout << "Theta: ";
        for (double t : theta)
            cout << t << " ";
        cout << endl;
    }
};
