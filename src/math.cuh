#pragma once

/**
 * @file
 * @brief Mathematical operations, such as the binomial coefficient and related functions.
 */

#include <stdio.h>
#include <iostream>
#include <cassert>

/// binomial coefficient / nCr
__host__ __device__ u64 binomial(u32 n, u32 k) {
    /* if (k > n - k) { */
    /*     k = n - k; */
    /* } */

    k = min(k, n - k);
    u64 result = 1;

    for (u32 denominator = 1, numerator = n; denominator <= k; denominator++, numerator--) {
        result = result * numerator / denominator;
    }

    return result;
}

/// Number of combinations with repetition
__host__ __device__ u64 combinations_with_repetitions(u32 k, u32 n) {
    return binomial(k + n - 1, k);
}

__host__ __device__ u32 combination_index_with_repetition_inner(u32 state_count_len, u8* state_count, u32 k) {
    u32 n = state_count_len;
    u32 index = 0;
    u32 states_left = k;

    for (u32 state = 0; state < n - 1; state += 1) {
        states_left -= state_count[state];

        if (states_left <= 0) {
            break;
        }

        index += combinations_with_repetitions(states_left - 1, n - state);
    }

    return index;
}

/**
 * @param state_count The combination represented by the count of each state at its index in the returned array
 * @return The index of the given combination with repetition
 */
__host__ __device__ u32 combination_index_with_repetition(u32 state_count_len, u8* state_count) {
    u32 k = 0;

    for (u32 i = 0; i < state_count_len; i++) {
        k += state_count[i];
    }

    return combination_index_with_repetition_inner(state_count_len, state_count, k);
}

/**
 *  Computes the index of a rule within a ruleset.
 *
 *  @param cell_neighbourhood_combinations: number of total possible combinations of states in the neighbourhood (precomputed)
 *  @param state: the current cell state
 *  @param state_count_len: The number of possible cell states.
 *  @param state_count: The number of cells with given state, for each state.
 */
__inline__ __host__ __device__ u32 get_rule_index(
        u32 cell_neighbourhood_combinations,
        u8 state,
        u32 state_count_len,
        u8* state_count
) {
    return ((u32) state) * cell_neighbourhood_combinations + combination_index_with_repetition(state_count_len, state_count);
}

/// Compute `cell_neighbourhood_combinations` for use in `get_rule_index`.
__host__ __device__ int compute_neighbouring_state_combinations(int neighbourhood_size, int states) {
    return combinations_with_repetitions(neighbourhood_size, states);
}

/// Computes the total number of rules in a ruleset.
__host__ __device__ int compute_ruleset_size(int neighbourhood_size, int states) {
    return states * compute_neighbouring_state_combinations(neighbourhood_size, states);
}

/// Modulus operation (as opposed to a remainder).
__inline__ __host__ __device__ int mod(int x, int n) {
    return (x % n + n) % n;
}

/// Integer power operation.
__inline__ __host__ __device__ int powi(int x, int p) {
    int i = 1;

    for (int j = 1; j <= p; j++) {
        i *= x;
    }

    return i;
}

/// Long integer power operation.
__inline__ __host__ __device__ long int powli(long int x, long int p) {
    long int i = 1;

    for (long int j = 1; j <= p; j++) {
        i *= x;
    }

    return i;
}
