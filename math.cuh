#pragma once
#include <stdio.h>
#include <iostream>
#include <cassert>

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

__host__ __device__ u64 combinations_with_repetitions(u32 k, u32 n) {
    return binomial(k + n - 1, k);
}

__host__ __device__ void combination_with_repetition_inner(u32 k, u32 n, u32 index, u32 state_count_len, u8* state_count) {
    i32 found_state = -1;
    u32 index_offset = 0;
    u32 total_state_combos = 0;

    for (int state = 0; state < n; state += 1) {
        int stateCombos = (int) combinations_with_repetitions(k - 1, n - state);
        index_offset = total_state_combos;
        total_state_combos += stateCombos;

        if (index < total_state_combos) {
            found_state = state;
            break;
        }
    }

    assert(found_state >= 0); // Combination index out of bounds.

    state_count[found_state + state_count_len - n] += 1;

    if (k > 1) {
        combination_with_repetition_inner(k - 1, n - found_state, index - index_offset, state_count_len, state_count);
    }
}

/**
 * @param k Number of selected items
 * @param n Number of different states to select from
 * @param index Index of the combination with repetition to return
 * @return The combination at the given {@param index} represented by the count of each state at its index in the returned array
 */
/* int[] combinationWithRepetition(int k, int n, int index) { */
/*     assertm(k > 0, "The combination class (k) must be positive."); */
/*     assertm(n > 0, "The number of states (n) must be positive."); */
/*     int[] state_count = new int[n]; */

/*     combination_with_repetition_inner(k, n, index, state_count); */

/*     return state_count; */
/* } */

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

__inline__ __host__ __device__ u32 get_rule_index(
        u32 cell_neighbourhood_combinations,
        u8 state,
        u32 state_count_len,
        u8* state_count
) {
    return ((u32) state) * cell_neighbourhood_combinations + combination_index_with_repetition(state_count_len, state_count);
}

__host__ __device__ int compute_neighbouring_state_combinations(int neighbourhood_size, int states) {
    return combinations_with_repetitions(neighbourhood_size, states);
}

__host__ __device__ int compute_ruleset_size(int neighbourhood_size, int states) {
    return states * compute_neighbouring_state_combinations(neighbourhood_size, states);
}

__inline__ __host__ __device__ int mod(int x, int n) {
    return (x % n + n) % n;
}

__inline__ __host__ __device__ int powi(int x, int p) {
    int i = 1;

    for (int j = 1; j <= p; j++) {
        i *= x;
    }

    return i;
}

__inline__ __host__ __device__ long int powli(long int x, long int p) {
    long int i = 1;

    for (long int j = 1; j <= p; j++) {
        i *= x;
    }

    return i;
}
