// cc -fPIC -shared -o my_functions.so my_functions.c
#include <math.h>

#include "my_functions.h"
#include "Trees.h"

#include <stdio.h>

uint8_t getJ()
{
    return J_VALUE;
}

uint8_t getPrediction(float* X, uint16_t index)
{
    return (*func_ptr[index]) (X);
}

uint16_t argmin(float *arr, uint16_t size)
{
    uint16_t minIdx = 0;

    for (uint16_t i = 1; i < size; i++)
    {
        if (arr[i] < arr[minIdx])
        {
            minIdx = i;
        }
    }
    return minIdx;
}

// src: https://blog.cloudflare.com/computing-euclidean-distance-on-144-dimensions
float compute_euclidean_distance(float* x, float* y)
{
	float distance = 0;
	for (uint16_t i = 0; i < PROBE_RES; i++)
	{
		float value = x[i] - y[i];
		distance += value * value;
	}
	return distance;
}
// This function returns the squared distance. We avoid computing the actual distance to save us from running the
// square root function - it's slow. Inside the code, for performance and simplicity, we'll mostly operate on the
// squared value. We don't need the actual distance value, we just need to find the vector with the smallest one.
// In our case it doesn't matter if we'll compare distances or squared distances!

uint16_t mostFrequent(uint8_t* arr, uint16_t n)
{
    // code here
    uint16_t maxcount = 0;
    uint16_t element_having_max_freq;
    for (uint16_t i = 0; i < n; i++) {
        uint16_t count = 0;
        for (uint16_t j = 0; j < n; j++) {
            if (arr[i] == arr[j])
                count++;
        }

        if (count > maxcount) {
            maxcount = count;
            element_having_max_freq = arr[i];
        }
    }

    return element_having_max_freq;
}

uint8_t cpredict(float* X, uint32_t length)
{
    // calculate distance between X and all cluster centers
    float euclidean_distances[CLUSTERS_NO];

    for(uint8_t i = 0; i < CLUSTERS_NO; i++)
    {
        euclidean_distances[i] = compute_euclidean_distance(X, cluster_centers[i]);
    }

    // get the closest cluster
    uint16_t cluster_index = argmin(euclidean_distances, CLUSTERS_NO);
    uint8_t* selected_classifiers_idx = indices[cluster_index];

    uint8_t votes[J_VALUE];
    // calculate clfs from this cluster
    for (uint8_t i = 0; i < J_VALUE; i++)
    {
        uint8_t clf_index = selected_classifiers_idx[i];
//        printf("I\n");
        votes[i] = getPrediction(X, clf_index);
//        printf("O\n");
    }

    // majority vote
    return mostFrequent(votes, J_VALUE);
}
