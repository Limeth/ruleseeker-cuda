#pragma once

// All possible grid geometries
#define GRID_GEOMETRY_SQUARE   0 // square tiling, like conway's game of life
#define GRID_GEOMETRY_TRIANGLE 1 // triangular tiling
#define GRID_GEOMETRY_HEXAGON  2 // hexagonal tiling

// All possible cell neighbourhoods
#define CELL_NEIGHBOURHOOD_TYPE_VERTEX 0 // a cell is in the current cell's neighbourhood iff it shares a vertex
#define CELL_NEIGHBOURHOOD_TYPE_EDGE   1 // a cell is in the current cell's neighbourhood iff it shares an edge

// All possible ways to decide whether a cell is "fit", or not.
#define FITNESS_EVAL_STATE 0 // counts the number of cells with a given state compared to the overall number of cells
#define FITNESS_EVAL_UPDATE 1 // counts the number of cell updates compared to the overall number of cells

// All possible ways to compute the actual fitness from the number of "fit" cells
// See https://www.desmos.com/calculator/b8daos2dqt for visual explanation.
#define FITNESS_FN_TYPE_ABS 0
#define FITNESS_FN_TYPE_LINEAR 1
#define FITNESS_FN_TYPE_LIKELIHOOD 2

#define CROSSOVER_METHOD_UNIFORM 0 // Each rule has a chance to be used 50/50
#define CROSSOVER_METHOD_SPLICES(N) N // Rulesets are split at N random points and joined across

#define MUTATION_METHOD_UNIFORM 0 // A mutation is determined granularly per-rule -- very expensive
#define MUTATION_METHOD_BINOMIAL_MEMCPY 1 // Number of mutations determined beforehand, then applied via memcpy's
#define MUTATION_METHOD_BINOMIAL_KERNEL 2 // Number of mutations determined beforehand, then applied via kernel

/**************************
 * START OF CONFIGURATION *
 **************************/

// # Grid Geometry

// [uint] Simulation grid width.
#define GRID_WIDTH  32
// [uint] Simulation grid height.
#define GRID_HEIGHT 32
// [enum] The shape of the grid's cells (square for Conway's GoL).
#define GRID_GEOMETRY GRID_GEOMETRY_HEXAGON
// [enum] Which cells are considered in the neighbourhood (vertex for Conway's GoL).
#define CELL_NEIGHBOURHOOD_TYPE CELL_NEIGHBOURHOOD_TYPE_EDGE
// [uchar] Number of states a cell can become (2 for Conway's GoL).
#define CELL_STATES 2

// Conway's GoL:
/* #define GRID_GEOMETRY GRID_GEOMETRY_SQUARE */
/* #define CELL_NEIGHBOURHOOD_TYPE CELL_NEIGHBOURHOOD_TYPE_VERTEX */
/* #define CELL_STATES 2 */


// # Seeking parameters (genetic algorithm)

// [uint] Number of cellular automata to simulate simultaneously.
#define POPULATION_SIZE 64
// [uint] Number of top candidates to keep for the next population, without any adjustments.
#define POPULATION_ELITES 2
// [uint] Number of top candidates to use for crossover and mutation (including elites).
#define POPULATION_SELECTION 16
// [float] The chance that a single rule of a ruleset changes during mutation.
#define MUTATION_CHANCE 0.5
// [bool] Whether to make mutation chance dependent on rank -- lower mutation chance for high fitness candidates.
#define MUTATION_CHANCE_ADAPTIVE_BY_RANK true
// [enum] The method used for the crossover operation.
#define CROSSOVER_METHOD CROSSOVER_METHOD_SPLICES(64)
// [enum] The method used for the mutation operation.
#define MUTATION_METHOD MUTATION_METHOD_BINOMIAL_KERNEL

// ## Fitness function

// [uint; N] N iteration indices at which to evaluate the fitness function.
#define FITNESS_FN_TARGET_ITERATIONS 50, 100, 150, 200, 201, 202, 203
// [uint; N] N target fit cell proportions evaluated at corresponding FITNESS_FN_TARGET_ITERATIONS indices.
#define FITNESS_FN_TARGET_VALUES 0.55, 0.05, 0.55, 0.0, 0.0, 0.0, 0.0

// ### Proportion

// [enum] Which cell evaluation method to use (decide whether a given cell is "fit", or not).
#define FITNESS_EVAL FITNESS_EVAL_UPDATE
// [uint] Which state to count the proportion of.
#define FITNESS_EVAL_STATE_INDEX 0
// [enum] Which way to evaluate the fitness function.
#define FITNESS_FN_TYPE FITNESS_FN_TYPE_LIKELIHOOD


// # Miscellaneous

// [uint] Execution block width and height.
#define BLOCK_LENGTH 16
// [bool] Whether to verify the GPU simulation with an equivalent CPU simulation.
#define CPU_VERIFY false
// [bool] `true` to keep aspect ratio, `false` to stretch to window.
#define KEEP_ASPECT_RATIO true
// [bool] `true` if multisampling should be enabled to fix "jagged" edges, `false` otherwise.
#define USE_MULTISAMPLING true
// [uint] Number of samples per pixel to use when multisampling is enabled.
#define MULTISAMPLING_SAMPLES 16
// [bool] Whether to display 2-state simulations using more colors, based on the cell action.
#define MULTICOLOR_TWO_STATE false
// [float/uint] Number of iterations per second.
#define FRAMERATE 8
// [uint] Number of milliseconds to wait between iterations (overriden by FRAMERATE).
/* #define SLEEP_MS 5000 */
// [uint] Initial window width.
#define WINDOW_WIDTH 1000
// [uint] Initial window height.
#define WINDOW_HEIGHT 1000
// [uint] Max number of frames exported as PNG.
/* #define EXPORT_FRAMES 100 */
#define PROMPT_TO_START false
// [uint] Close the application after this many frames. Useful for profiling. Uncomment to disable.
/* #define EXIT_AFTER_FRAMES 3 */
// [uint] Close the application after this many populations. Useful for profiling. Uncomment to disable.
/* #define EXIT_AFTER_POPULATIONS 5 */

// [bool] Whether to use the kernel with shared memory.
#define USE_SHARED_MEMORY true
// [bool] Whether to use `srand` and `rand` methods for random number sampling (`true`), or a cryptographically secure RNG `arc4random` (`false`).
#define DETERMINISTIC_RANDOMNESS false
// [bool] Whether to enable `debug_synchronous` for all CUB calls to print debug info.
#define CUB_DEBUG_SYNCHRONOUS false
// [uint] Max number of simultaneously running thread blocks performing crossover, each
// with a separate random number generator.
#define CROSSOVER_MUTATE_BLOCKS_MAX 16
// [uint] Number of simultaneously running crossover kernels, each with their own set of
// crossover blocks.
#define CROSSOVER_MUTATE_KERNELS_MAX 32

/************************
 * END OF CONFIGURATION *
 ************************/
