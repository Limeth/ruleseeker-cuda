#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <GL/glut.h>
#include "config.cuh"
#include "util.cuh"
#include "draw.cuh"
#include "math.cuh"

using namespace std;

// Capitalised because they are effectively constant
int CPU_CELL_NEIGHBOURHOOD_COMBINATIONS = -1;
__constant__ int GPU_CELL_NEIGHBOURHOOD_COMBINATIONS = -1;
int CPU_RULESET_SIZE = -1;
__constant__ int GPU_RULESET_SIZE = -1;

__inline__ __host__ __device__ int get_cell_neighbourhood_combinations() {
#ifdef __CUDA_ARCH__
    return GPU_CELL_NEIGHBOURHOOD_COMBINATIONS;
#else
    return CPU_CELL_NEIGHBOURHOOD_COMBINATIONS;
#endif
}

__inline__ __host__ __device__ int get_ruleset_size() {
#ifdef __CUDA_ARCH__
    return GPU_RULESET_SIZE;
#else
    return CPU_RULESET_SIZE;
#endif
}

__inline__ __host__ __device__ void get_neighbours(i32 x, i32 y, i32vec2 neighbours[CELL_NEIGHBOURHOOD_SIZE]) {
#if GRID_GEOMETRY == GRID_GEOMETRY_SQUARE
    #if CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_VERTEX
    neighbours[0] = make_i32vec2(-1, -1);
    neighbours[1] = make_i32vec2( 0, -1);
    neighbours[2] = make_i32vec2( 1, -1);
    neighbours[3] = make_i32vec2(-1,  0);
    neighbours[4] = make_i32vec2( 1,  0);
    neighbours[5] = make_i32vec2(-1,  1);
    neighbours[6] = make_i32vec2( 0,  1);
    neighbours[7] = make_i32vec2( 1,  1);
    #elif CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_EDGE
    neighbours[0] = make_i32vec2(-1,  0);
    neighbours[1] = make_i32vec2( 0, -1);
    neighbours[2] = make_i32vec2( 1,  0);
    neighbours[3] = make_i32vec2( 0,  1);
    #endif
#elif GRID_GEOMETRY == GRID_GEOMETRY_TRIANGLE
    bool pointing_up = (x + y) % 2 == 0;
    #if CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_VERTEX
    if (pointing_up) {
        neighbours[ 0] = make_i32vec2(-1,  0);
        neighbours[ 1] = make_i32vec2(-1, -1);
        neighbours[ 2] = make_i32vec2( 0, -1);
        neighbours[ 3] = make_i32vec2( 1, -1);
        neighbours[ 4] = make_i32vec2( 1,  0);
        neighbours[ 5] = make_i32vec2( 2,  0);
        neighbours[ 6] = make_i32vec2( 2,  1);
        neighbours[ 7] = make_i32vec2( 1,  1);
        neighbours[ 8] = make_i32vec2( 0,  1);
        neighbours[ 9] = make_i32vec2(-1,  1);
        neighbours[10] = make_i32vec2(-2,  1);
        neighbours[11] = make_i32vec2(-2,  0);
    } else {
        neighbours[ 0] = make_i32vec2(-1,  0);
        neighbours[ 1] = make_i32vec2(-2,  0);
        neighbours[ 2] = make_i32vec2(-2, -1);
        neighbours[ 3] = make_i32vec2(-1, -1);
        neighbours[ 4] = make_i32vec2( 0, -1);
        neighbours[ 5] = make_i32vec2( 1, -1);
        neighbours[ 6] = make_i32vec2( 2, -1);
        neighbours[ 7] = make_i32vec2( 2,  0);
        neighbours[ 8] = make_i32vec2( 1,  0);
        neighbours[ 9] = make_i32vec2( 1,  1);
        neighbours[10] = make_i32vec2( 0,  1);
        neighbours[11] = make_i32vec2(-1,  1);
    }
    #elif CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_EDGE
    if (pointing_up) {
        neighbours[0] = make_i32vec2(-1,  0);
        neighbours[1] = make_i32vec2( 1,  0);
        neighbours[2] = make_i32vec2( 0,  1);
    } else {
        neighbours[0] = make_i32vec2(-1,  0);
        neighbours[1] = make_i32vec2( 1,  0);
        neighbours[2] = make_i32vec2( 0, -1);
    }
    #endif
#elif GRID_GEOMETRY == GRID_GEOMETRY_HEXAGON
    bool row_even = y % 2 == 0;

    if (row_even) {
        neighbours[0] = make_i32vec2(-1,  0);
        neighbours[1] = make_i32vec2(-1, -1);
        neighbours[2] = make_i32vec2( 0, -1);
        neighbours[3] = make_i32vec2( 1,  0);
        neighbours[4] = make_i32vec2( 0,  1);
        neighbours[5] = make_i32vec2(-1,  1);
    } else {
        neighbours[0] = make_i32vec2(-1,  0);
        neighbours[1] = make_i32vec2( 0, -1);
        neighbours[2] = make_i32vec2( 1, -1);
        neighbours[3] = make_i32vec2( 1,  0);
        neighbours[4] = make_i32vec2( 1,  1);
        neighbours[5] = make_i32vec2( 0,  1);
    }
#endif
}

void print_configuration() {
    printf("\nConfiguration:\n");
    printf("\tGrid width: %d\n", GRID_WIDTH);
    printf("\tGrid height: %d\n", GRID_HEIGHT);

    if (GRID_GEOMETRY == GRID_GEOMETRY_SQUARE) {
        printf("\tGrid geometry: Square\n");
    } else if (GRID_GEOMETRY == GRID_GEOMETRY_TRIANGLE) {
        printf("\tGrid geometry: Triangle\n");
    } else if (GRID_GEOMETRY == GRID_GEOMETRY_HEXAGON) {
        printf("\tGrid geometry: Hexagon\n");
    } else {
        printf("\tGrid geometry: Invalid, aborting...\n");
        exit(1);
    }

    if (CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_VERTEX) {
        printf("\tCell neighbourhood type: Vertex\n");
    } else if (CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_EDGE) {
        printf("\tCell neighbourhood type: Edge\n");
    } else {
        printf("\tCell neighbourhood type: Invalid, aborting...\n");
        exit(1);
    }

    printf("\tCell neighbourhood size: %d\n", CELL_NEIGHBOURHOOD_SIZE);
    printf("\tCell states: %d\n", CELL_STATES);

    CPU_CELL_NEIGHBOURHOOD_COMBINATIONS = compute_neighbouring_state_combinations(CELL_NEIGHBOURHOOD_SIZE, CELL_STATES);
    cudaMemcpyToSymbol(GPU_CELL_NEIGHBOURHOOD_COMBINATIONS, &CPU_CELL_NEIGHBOURHOOD_COMBINATIONS, sizeof(int));
    printf("\tCell neighbourhood combinations: %d (with a combinatorial number system, %ld with simple indexing)\n", CPU_CELL_NEIGHBOURHOOD_COMBINATIONS, powli(CELL_NEIGHBOURHOOD_SIZE, CELL_STATES));

    CPU_RULESET_SIZE = compute_ruleset_size(CELL_NEIGHBOURHOOD_SIZE, CELL_STATES);
    cudaMemcpyToSymbol(GPU_RULESET_SIZE, &CPU_RULESET_SIZE, sizeof(int));
    printf("\tRuleset size: %d (with a combinatorial number system, %ld with simple indexing)\n", CPU_RULESET_SIZE, CELL_STATES * powli(CELL_NEIGHBOURHOOD_SIZE, CELL_STATES));
    printf("\n");
}

__device__ u8* device_gpu_ruleset;
u8* gpu_ruleset = NULL;
u8* cpu_ruleset = NULL;

__inline__ __host__ __device__ u8* get_ruleset() {
#ifdef __CUDA_ARCH__
    return device_gpu_ruleset;
#else
    return cpu_ruleset;
#endif
}

__device__ u8* gpu_grid_states_1 = NULL;
__device__ u8* gpu_grid_states_2 = NULL;

u8 *cpu_grid_states_1 = NULL;
u8 *cpu_grid_states_2 = NULL;
u8 *cpu_grid_states_tmp = NULL;

// udalosti pro mereni casu v CUDA
cudaEvent_t start, stop;

__inline__ __host__ __device__ int get_cell_index(int width, int height, int x, int y) {
    x = mod(x, width);
    y = mod(y, height);

    return x + y * width;
}

__host__ __device__ u8 get_next_state(u8 current_state, u8* neighbours) {
    u8* ruleset = get_ruleset();

    // In debug mode, validate the `current_state` argument.
    assert(current_state < CELL_STATES);

    // In debug mode, validate the `neighbours` argument.
#ifndef NDEBUG
    {
        u8 total_neighbours = 0;

        for (u8 state = 0; state < CELL_STATES; state++) {
            u8 current_neighbours = neighbours[state];
            total_neighbours += current_neighbours;

            assert(current_neighbours <= CELL_NEIGHBOURHOOD_SIZE);
        }

        assert(total_neighbours == CELL_NEIGHBOURHOOD_SIZE);
    }
#endif

    u32 index = get_rule_index(get_cell_neighbourhood_combinations(), current_state, CELL_STATES, neighbours);

    assert(index < get_ruleset_size());

    return ruleset[index];
}

__host__ __device__ u8 update_cell(u8* in, u8* out, int width, int height, int x, int y) {
    int cellID = get_cell_index(width, height, x, y);
    u8 currentState = in[cellID];
    i32vec2 neighbours[CELL_NEIGHBOURHOOD_SIZE];
    u8 state_count[CELL_STATES] = { 0 };

    get_neighbours(x, y, neighbours);

    for (u32 neighbour_index = 0; neighbour_index < CELL_NEIGHBOURHOOD_SIZE; neighbour_index++) {
        i32vec2 neighbour = neighbours[neighbour_index];
        i32 abs_x = x + neighbour.x;
        i32 abs_y = y + neighbour.y;
        i32 neighbour_cell_index = get_cell_index(width, height, abs_x, abs_y);

        state_count[in[neighbour_cell_index]] += 1;
    }

    /* for (int rel_y = -1; rel_y <= 1; rel_y++) { */
    /*     for (int rel_x = -1; rel_x <= 1; rel_x++) { */
    /*         if (rel_x == 0 && rel_y == 0) { */
    /*             continue; */
    /*         } */

    /*         int abs_x = x + rel_x; */
    /*         int abs_y = y + rel_y; */
    /*         int neighbourID = get_cell_index(width, height, abs_x, abs_y); */

    /*         state_count[in[neighbourID]] += 1; */
    /*     } */
    /* } */

    int nextState = get_next_state(currentState, state_count);
    out[cellID] = nextState;

    return nextState;
}

/* funkce zajistujici aktualizaci simulace - verze pro CPU
 *  in - vstupni simulacni mrizka
 *  out - vystupni simulacni mrizka
 *  width - sirka simulacni mrizky
 *  height - vyska simulacni mrizky
 */
void life_cpu(u8* in, u8* out, int width, int height) {
    int threadID;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            update_cell(in, out, width, height, x, y);
        }
    }
}

// funkce pro zapis barvy pixelu do bitmapy, nova barva je odvozena ze stavu simulace
__inline__ __device__ void stateToColor(u8 oldValue, u8 newValue, uchar4* bitmap, int bitmapId) {
    uchar4 color;

    color.x = (newValue==0 && oldValue==1) ? 255 : 0;
    color.y = (newValue==1 && oldValue==0) ? 255 : 0;
    color.z = (newValue==1 && oldValue==1) ? 255 : 0;
    color.w = 0;

    bitmap[bitmapId] = color;
}

__global__ void life_kernel(uchar4* bitmap, u8* in, u8* out, int width, int height) {
    int x = threadIdx.x + blockIdx.x * BLOCK_LENGTH;
    int y = threadIdx.y + blockIdx.y * BLOCK_LENGTH;
    int threadID = get_cell_index(width, height, x, y);

    if (threadID < width*height) {
        u8 oldValue = in[threadID];
        u8 newValue = update_cell(in, out, width, height, x, y);

        stateToColor(oldValue, newValue, bitmap, threadID);
    }

    /* int combination[3] = { */
    /*     5, 0, 3 */
    /* }; */

    /* if (threadID == 0) { */
    /*     int result = combination_index_with_repetition(3, combination); */
    /*     printf("gpu result: %d\n", result); */
    /* } */
}


// funkce pro spusteni kernelu + priprava potrebnych dat a struktur
void callKernelCUDA(void) {
    // ulozeni pocatecniho casu
    CHECK_ERROR(cudaEventRecord(start, 0));

    // grid and block dimensions
    dim3 blocks(GRID_WIDTH_IN_BLOCKS, GRID_HEIGHT_IN_BLOCKS);
    dim3 threads(BLOCK_LENGTH, BLOCK_LENGTH);

    // aktualizace simulace + vygenerovani bitmapy pro zobrazeni stavu simulace
    life_kernel<<<blocks,threads>>>(bitmap->deviceData, gpu_grid_states_1, gpu_grid_states_2, bitmap->width, bitmap->height);

    // prohozeni ukazatelu (u textur pouzit pouze gpu_grid_states_2)
    swap(gpu_grid_states_1, gpu_grid_states_2);

    // ulozeni casu ukonceni simulace
    CHECK_ERROR(cudaEventRecord(stop, 0));
    CHECK_ERROR(cudaEventSynchronize(stop));

    float elapsedTime;

    // vypis casu simulace
    CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Update: %f ms\n", elapsedTime);

    // kopirovani bitmapy zpet na CPU pro zobrazeni
    CHECK_ERROR(cudaMemcpy(bitmap->pixels, bitmap->deviceData, bitmap->width*bitmap->height*sizeof(uchar4), cudaMemcpyDeviceToHost));

#if CPU_VERIFY
    printf("Verifying on the CPU...\n");
    // krok simulace life game na CPU
    life_cpu(cpu_grid_states_1, cpu_grid_states_2, bitmap->width, bitmap->height);
    swap(cpu_grid_states_1, cpu_grid_states_2);

    cudaMemcpy(cpu_grid_states_tmp, gpu_grid_states_1, bitmap->width*bitmap->height*sizeof(u8), cudaMemcpyDeviceToHost);

    int diffs = 0;

    // porovnani vysledku CPU simulace a GPU simulace
    for(int row=0;row<bitmap->height;row++) {
        for(int col=0; col<bitmap->width; col++) {

            int rowAddr = row * bitmap->width;
            int threadID = rowAddr + col;	// index vlakna

            if(cpu_grid_states_1[threadID] != cpu_grid_states_tmp[threadID])
                diffs++;
        }
    }

    if(diffs != 0)
        std::cout << "CHYBA: " << diffs << " rozdily mezi CPU & GPU simulacni mrizkou" << std::endl;
#endif
}


// inicializace CUDA - alokace potrebnych dat a vygenerovani pocatecniho stavu lifu
void initialize(void) {
    // alokace struktury bitmapy
    bitmap = (bitmap_t*) malloc(sizeof(bitmap));
    bitmap->width = GRID_WIDTH;
    bitmap->height = GRID_HEIGHT;

    cudaHostAlloc((void**) &(bitmap->pixels), bitmap->width*bitmap->height*sizeof(uchar4), cudaHostAllocDefault);


    // alokovani mista pro bitmapu na GPU
    int bitmapSize = bitmap->width*bitmap->height;
    CHECK_ERROR(cudaMalloc((void**) &(gpu_ruleset), 2 * 9 * sizeof(u8)));
    CHECK_ERROR(cudaMalloc((void**) &(bitmap->deviceData), bitmapSize*sizeof(uchar4)));
    CHECK_ERROR(cudaMalloc((void**) &(gpu_grid_states_1), bitmapSize*sizeof(u8)));
    CHECK_ERROR(cudaMalloc((void**) &(gpu_grid_states_2), bitmapSize*sizeof(u8)));

    cudaMemset(bitmap->deviceData, 0, bitmapSize*sizeof(uchar4));

    cpu_ruleset = (u8*) malloc(2 * 9 * sizeof(u8));
    cpu_grid_states_1 = (u8*) malloc(bitmapSize*sizeof(u8));
    cpu_grid_states_2 = (u8*) malloc(bitmapSize*sizeof(u8));
    cpu_grid_states_tmp = (u8*) malloc(bitmapSize*sizeof(u8));

    srand(0);

    // inicializace pocatecniho stavu lifu
    for (int i = 0; i < bitmapSize; i++) {
        cpu_grid_states_1[i] = (u8) (rand() % 2);
    }

    // prekopirovani pocatecniho stavu do GPU
    cudaMemcpy(gpu_grid_states_1, cpu_grid_states_1, bitmapSize*sizeof(u8), cudaMemcpyHostToDevice);

    // nakopirovani tabulky novych stavu do konstantni pameti
    u8 ruleset[2 * 9] = {
        // currentState == 0:
        0, 0, 0, 1, 0, 0, 0, 0, 0,
        // currentState == 1:
        0, 0, 1, 1, 0, 0, 0, 0, 0,
    };

    memcpy(cpu_ruleset, ruleset, 2 * 9 * sizeof(u8));
    CHECK_ERROR(cudaMemcpy(gpu_ruleset, ruleset, 2 * 9 * sizeof(u8), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpyToSymbol(device_gpu_ruleset, &gpu_ruleset, sizeof(u8*)));

    // vytvoreni struktur udalosti pro mereni casu
    CHECK_ERROR(cudaEventCreate( &start ));
    CHECK_ERROR(cudaEventCreate( &stop ));
}

// funkce volana pri ukonceni aplikace, uvolni vsechy prostredky alokovane v CUDA 
void finalize(void) {

    // uvolneni bitmapy - na CPU i GPU
    if (bitmap != NULL) {
        if (bitmap->pixels != NULL) {
            // uvolneni bitmapy na CPU
            cudaFreeHost(bitmap->pixels);
            bitmap->pixels = NULL;
        }
        if (bitmap->deviceData != NULL) {
            // uvolneni bitmapy na GPU
            cudaFree(bitmap->deviceData);
            bitmap->deviceData = NULL;
        }
        cudaFree(gpu_ruleset);
        cudaFree(gpu_grid_states_1);
        cudaFree(gpu_grid_states_2);
        free(bitmap);
    }

    // uvolneni simulacnich mrizek pro CPU variantu lifu
    free(cpu_ruleset);
    free(cpu_grid_states_1);
    free(cpu_grid_states_2);
    free(cpu_grid_states_tmp);

    // zruseni struktur udalosti
    CHECK_ERROR(cudaEventDestroy( start ));
    CHECK_ERROR(cudaEventDestroy( stop ));
}

// called every frame
void idle_func() {
    callKernelCUDA();
    glutPostRedisplay();
}

static void handle_keys(unsigned char key, int x, int y) {
    switch (key) {
        case 27:	// ESC
            finalize();
            exit(0);
    }
}

int main(int argc, char **argv) {
    print_configuration();
    initialize();

    printf("Press Enter to begin simulation.");
    getchar();

    return ui_loop(argc, argv, GRID_WIDTH, GRID_HEIGHT, handle_keys, idle_func);
}
