#include <iostream>
#include <vector>
#include <utility>
#include <fstream>
#include <algorithm>
#include <chrono>

#define LOCAL_QUEUE_SIZE 256
#define PERSONAL_QUEUE_SIZE 128
#define LWS 32

using namespace std;
using Graph = vector<vector<int> >;

__host__ __device__ int round_div_up (int a, int b){
    return (a + b - 1)/b;
}

void cuda_err_check (cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf (stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString (err), file, line);
        exit (EXIT_FAILURE);
    }
}

void single_step (int numNodes, int * nodePtrs, int * nodeNeighbors, int * currLevelNodes, int * nextLevelNodes, bool * visitedNodes, int & numCurrLevelNodes, int & numNextLevelNodes){
    for (int i = 0; i < numCurrLevelNodes; i++){
        int currNode = currLevelNodes[i];
        int start = nodePtrs[currNode];
        int end = nodePtrs[currNode+1];
        for (int j = start; j < end; j++){
            int neighbor = nodeNeighbors[j];
            if (!visitedNodes[neighbor]){
                visitedNodes[neighbor] = true;
                nextLevelNodes[numNextLevelNodes] = neighbor;
                numNextLevelNodes++;
            }
        }
    }
    numCurrLevelNodes = numNextLevelNodes;
    numNextLevelNodes = 0;
}

double sequential_bfs (int numNodes, int * nodePtrs, int * nodeNeighbors){
    double time_spent = 0;
    int * currLevelNodes = (int*)malloc(numNodes * sizeof(int));
    int * nextLevelNodes = (int*)malloc(numNodes * sizeof(int));
    bool * visitedNodes = (bool*)malloc(numNodes * sizeof(bool));
    int numCurrLevelNodes = 1;
    int numNextLevelNodes = 0;
    int startNode = 0;
    currLevelNodes[0] = startNode;
    visitedNodes[startNode] = true;
     while (numCurrLevelNodes > 0){
        numNextLevelNodes = 0;
        auto start = chrono::high_resolution_clock::now();
        single_step(numNodes, nodePtrs, nodeNeighbors, currLevelNodes, nextLevelNodes, visitedNodes, numCurrLevelNodes, numNextLevelNodes);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<float> duration = end - start;
        time_spent += duration.count();
        swap(currLevelNodes, nextLevelNodes);
    }
    free(currLevelNodes);
    free(nextLevelNodes);
    free(visitedNodes);
    return time_spent * 1000;
}

__global__ void global_queue (
    int numNodes, 
    int *d_nodePtrs, int *d_nodeNeighbors, 
    int *d_currLevelNodes, int *d_nodeVisited, const int numCurrLevelNodes,
    int *d_nextLevelNodes, int *numNextLevelNodes
    ){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numCurrLevelNodes){
        int node = d_currLevelNodes[tid];
        int start = d_nodePtrs[node];
        int end = d_nodePtrs[node + 1];
        for (int i = start; i < end; i++){
            int neighbor = d_nodeNeighbors[i];
            if (d_nodeVisited[neighbor] == 0 && (atomicCAS(&d_nodeVisited[neighbor], 0, 1)) == 0){
                int index = atomicAdd(numNextLevelNodes, 1);
                d_nextLevelNodes[index] = neighbor;
            }
        }
    }
}

__global__ void block_queue (
    int numNodes, 
    int *d_nodePtrs, int *d_nodeNeighbors, 
    int *d_currLevelNodes, int *d_nodeVisited, const int numCurrLevelNodes,
    int *d_nextLevelNodes, int *numNextLevelNodes
    ){
    extern __shared__ int4 lmem[];
    int * local_queue = ((int*)lmem) + 1;
    int * local_queue_size = ((int*)lmem);
    if (threadIdx.x == 0) *local_queue_size = 0;
    
    __syncthreads();

    int personal_queue_size = 0;
    int4 personal_queue4[PERSONAL_QUEUE_SIZE/4];

    int *personal_queue = (int*)personal_queue4;
    int4 *d_nextLevelNodes4 = (int4*)d_nextLevelNodes;


    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numCurrLevelNodes){
        int node = d_currLevelNodes[tid];
        int start = d_nodePtrs[node];
        int end = d_nodePtrs[node + 1];
        for (int i = start; i < end; i++){
            int neighbor = d_nodeNeighbors[i];
            if (d_nodeVisited[neighbor] == 0 && (atomicCAS(&d_nodeVisited[neighbor], 0, 1)) == 0){
                int index = -1;
                if (personal_queue_size < PERSONAL_QUEUE_SIZE){
                    personal_queue[personal_queue_size++] = neighbor;
                    continue;
                }
                if (*local_queue_size < LOCAL_QUEUE_SIZE && (index = atomicAdd_block(local_queue_size, 1)) < LOCAL_QUEUE_SIZE){
                    local_queue[index] = neighbor;
                    continue;
                }
                d_nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = neighbor;
            }
        }
    }
    
    // merge personal queue with local queue
    // int local_index = atomicAdd(local_queue_size, personal_queue_size);
    // for (int i = 0; i < personal_queue_size; i++)
    //     local_queue[local_index + i] = personal_queue[i];


    // merge personal queue with global queue
    // int local_index = atomicAdd(numNextLevelNodes, personal_queue_size);
    // for (int i = 0; i < personal_queue_size; i++)
    //     d_nextLevelNodes[local_index + i] = personal_queue[i];

    //merge personal queue with global queue vectorized
    int local_index = atomicAdd(numNextLevelNodes, personal_queue_size);
    if (personal_queue_size > 0){
        int new_personal_queue_size = personal_queue_size;
        int local_offset = local_index%4 == 0 ? 0 : 4-local_index%4;
        if (local_offset > personal_queue_size){
            for (int i = 0; i < personal_queue_size; i++)
                d_nextLevelNodes[local_index + i] = personal_queue[i];
        }
        else{
                // printf("Misaligned local_index: %d\n", local_index);
            for (int i = 0; i < local_offset; i++)
                d_nextLevelNodes[local_index + i] = personal_queue[personal_queue_size-1-i];
            local_index = local_index + local_offset;
            new_personal_queue_size = personal_queue_size - local_offset;
        
            int nquart_personal_queue_size = (new_personal_queue_size - new_personal_queue_size%4)/4;

            for (int i = 0; i < nquart_personal_queue_size; i++)
                d_nextLevelNodes4[local_index/4 + i] = personal_queue4[i];

            for (int i = nquart_personal_queue_size * 4; i < new_personal_queue_size; i++)
                d_nextLevelNodes[local_index + i] = personal_queue[i];
            }
        }

    __syncthreads();
    // merge local queue with global queue
    if (*local_queue_size > 0){
        int number_of_nodes_to_merge_per_thread = *local_queue_size / blockDim.x;
        int start_index = threadIdx.x * number_of_nodes_to_merge_per_thread;
        int end_index = start_index + number_of_nodes_to_merge_per_thread;
        if (threadIdx.x == blockDim.x - 1) end_index = *local_queue_size;
        local_index = atomicAdd(numNextLevelNodes, end_index - start_index);
        for (int i = 0; i < end_index - start_index; i++)
            d_nextLevelNodes[local_index + i] = local_queue[start_index + i];
    }
}

float kernel_launch (
    int numNodes, 
    int *d_nodePtrs, int *d_nodeNeighbors, 
    int *d_currLevelNodes, int * numCurrentLevelNodes, 
    int *d_nodeVisited, int lws,
    bool use_global_queue = false
    ){

    int numBlocks;
    cudaError_t err;

    int4 *d_nextLevelNodes4;
    int *numNextLevelNodes;
    
    int * h_currentLevelNodes;
    err = cudaMallocHost((void**)&h_currentLevelNodes, numNodes * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    int * h_numCurrentLevelNodes = (int*)malloc(sizeof(int));
    err = cudaMemcpy(h_numCurrentLevelNodes, numCurrentLevelNodes, sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

    int nquarts_nodes = round_div_up(numNodes, 4);
    err = cudaMalloc((void**)&numNextLevelNodes, sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&d_nextLevelNodes4, nquarts_nodes * sizeof(int4)); cuda_err_check(err, __FILE__, __LINE__);
    
    err = cudaMemset(numNextLevelNodes, 0, sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    int level = 0;

    cudaEvent_t start, stop;

    err = cudaEventCreate(&start); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventCreate(&stop); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaEventRecord(start); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventRecord(stop); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventSynchronize(stop); cuda_err_check(err, __FILE__, __LINE__);

    float milliseconds = 0;
    err = cudaEventElapsedTime(&milliseconds, start, stop); cuda_err_check(err, __FILE__, __LINE__);

    float total_time = 0;

    int * d_nextLevelNodes = (int*)d_nextLevelNodes4;

    while (*h_numCurrentLevelNodes > 0){

        numBlocks = round_div_up(*h_numCurrentLevelNodes, lws);

        err = cudaMemset(numNextLevelNodes, 0, sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemset(d_nextLevelNodes4, 0, nquarts_nodes * sizeof(int4)); cuda_err_check(err, __FILE__, __LINE__);

        // cout << "Launching kernel with " << numBlocks << " blocks and " << lws << " threads per block" << endl;
        err = cudaEventRecord(start); cuda_err_check(err, __FILE__, __LINE__);
        if (use_global_queue)
            global_queue<<<numBlocks, lws>>>(numNodes, d_nodePtrs, d_nodeNeighbors, d_currLevelNodes, d_nodeVisited, *h_numCurrentLevelNodes, d_nextLevelNodes, numNextLevelNodes);
        else
            block_queue<<<numBlocks, lws, sizeof(int)*(LOCAL_QUEUE_SIZE+1)>>>(numNodes, d_nodePtrs, d_nodeNeighbors, d_currLevelNodes, d_nodeVisited, *h_numCurrentLevelNodes, d_nextLevelNodes, numNextLevelNodes);
        err = cudaEventRecord(stop); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaEventSynchronize(stop); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaMemcpy(numCurrentLevelNodes, numNextLevelNodes, sizeof(int), cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(d_currLevelNodes, d_nextLevelNodes, numNodes * sizeof(int), cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);

        // int old_numCurrentLevelNodes = *h_numCurrentLevelNodes;

        err = cudaMemcpy(h_numCurrentLevelNodes, numCurrentLevelNodes, sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(h_currentLevelNodes, d_currLevelNodes, *h_numCurrentLevelNodes * sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

        sort(h_currentLevelNodes, h_currentLevelNodes + *h_numCurrentLevelNodes);

        float milliseconds = 0;
        err = cudaEventElapsedTime(&milliseconds, start, stop); cuda_err_check(err, __FILE__, __LINE__);
        // cout << "Time taken for kernel execution: " << milliseconds << " ms " << "with " << numBlocks << " blocks and " << lws << " threads per block" << " at level " << level << " with " << old_numCurrentLevelNodes << " nodes" << " generated " << *h_numCurrentLevelNodes << " nodes" << endl;
        level++;
        total_time += milliseconds;

        // cout << "numCurrentLevelNodes on CPU: " << *h_numCurrentLevelNodes << endl;
        // cout << "Level " << level << " on CPU: ";
        // for (int i = 0; i < *h_numCurrentLevelNodes; i++){
        //     cout << h_currentLevelNodes[i] << " ";
        // }
        // cout << endl;
    }

    // cout << "Total time taken for kernel execution: " << total_time << " ms" << endl;

    err = cudaFree(numNextLevelNodes); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_nextLevelNodes4); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFreeHost(h_currentLevelNodes); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaEventDestroy(start); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventDestroy(stop); cuda_err_check(err, __FILE__, __LINE__);

    return total_time;
}

void init_structures (int numNodes, int *d_currLevelNodes, int *numCurrLevelNodes, int *d_nodeVisited){
    cudaError_t err = cudaMemset(d_currLevelNodes, 0, numNodes * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemset(d_nodeVisited, 0, numNodes * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    int number_of_current_level_nodes = 1;
    int startNode = 0;
    char visited = 1;

    err = cudaMemcpy(numCurrLevelNodes, &number_of_current_level_nodes, sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_currLevelNodes, &startNode, sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(&d_nodeVisited[startNode], &visited, sizeof(char), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
}

int main (int argc, char ** argv){
    if (argc < 2){
        cout << "Usage: " << argv[0] << " <graph_file> [-asym]" << endl;
        exit(1);
    }

    bool is_asymm = false;

    if (argc == 3){
        if (strcmp(argv[2], "-asym") == 0){
            is_asymm = true;
        }
    }

    int numNodes, numEdges;
    cudaError_t err;

    int *d_nodePtrs;
    int *d_nodeNeighbors;

    Graph g;
    ifstream infile;
    infile.open(argv[1]);
    // check if file is open
    if (!infile.is_open()){
        cout << "Could not open file " << argv[1] << endl;
        exit(1);
    }
    infile >> numNodes >> numEdges;
    g.resize(numNodes);
    for (int i = 0; i < numEdges; i++){
        int src, dst;
        infile >> src >> dst;
        g[src].push_back(dst);
        if (!is_asymm) g[dst].push_back(src);
    }
    infile.close();

    if (!is_asymm) numEdges *= 2;
    
    err = cudaMalloc((void**)&d_nodePtrs, (numNodes+1) * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&d_nodeNeighbors, numEdges * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    
    int * h_nodePtrs = (int*)malloc((numNodes+1) * sizeof(int));
    int * h_nodeNeighbors = (int*)malloc(numEdges * sizeof(int));

    int ptr = 0;
    for (int i = 0; i < numNodes; i++){
        h_nodePtrs[i] = ptr;
        for (int j = 0; j < g[i].size(); j++){
            h_nodeNeighbors[ptr] = g[i][j];
            ptr++;
        }
    }
    h_nodePtrs[numNodes] = ptr;

    err = cudaMemcpy(d_nodePtrs, h_nodePtrs, (numNodes+1) * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_nodeNeighbors, h_nodeNeighbors, numEdges * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    int *d_currLevelNodes;
    int *numCurrLevelNodes;
    int *d_nodeVisited;

    err = cudaMalloc((void**)&d_currLevelNodes, numNodes * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&d_nodeVisited, numNodes * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&numCurrLevelNodes, sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    
    double sequential_time = 0;
    float global_time = 0;
    float local_time = 0;

    sequential_time = sequential_bfs(numNodes, h_nodePtrs, h_nodeNeighbors);
    init_structures(numNodes, d_currLevelNodes, numCurrLevelNodes, d_nodeVisited);
    global_time = kernel_launch(numNodes, d_nodePtrs, d_nodeNeighbors, d_currLevelNodes, numCurrLevelNodes, d_nodeVisited, LWS, true);
    init_structures(numNodes, d_currLevelNodes, numCurrLevelNodes, d_nodeVisited);
    local_time = kernel_launch(numNodes, d_nodePtrs, d_nodeNeighbors, d_currLevelNodes, numCurrLevelNodes, d_nodeVisited, LWS, false);
    
    cout << "------------------------------------------------------------------" << endl;
    cout << "Sequential time: " << sequential_time << " ms" << endl;
    cout << "Global queue time: " << global_time << " ms" << endl;
    cout << "Local queue time: " << local_time << " ms" << endl;
    cout << "Speedup of global queue over sequential: " << sequential_time/global_time << endl;
    cout << "Speedup of local queue over sequential: " << sequential_time/local_time << endl;
    cout << "Speedup of local queue over global queue: " << global_time/local_time << endl;

    err = cudaFree(d_nodePtrs); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_nodeNeighbors); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_currLevelNodes); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_nodeVisited); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(numCurrLevelNodes); cuda_err_check(err, __FILE__, __LINE__);

    free(h_nodePtrs);
    free(h_nodeNeighbors);

    return 0;
}