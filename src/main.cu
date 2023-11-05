#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <utility>
#include <fstream>
#include <algorithm>

#define LOCAL_QUEUE_SIZE 512
#define PERSONAL_QUEUE_SIZE 128
#define LWS 64

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

__global__ void kernel (
    int numNodes, 
    int *d_nodePtrs, int *d_nodeNeighbors, 
    int *d_currLevelNodes, int *d_nodeVisited, const int numCurrLevelNodes,
    int *d_nextLevelNodes, int *numNextLevelNodes
    ){
    extern __shared__ int lmem[];
    int * local_queue = lmem + 1;
    int * local_queue_size = lmem;
    if (threadIdx.x == 0) *local_queue_size = 0;
    
    __syncthreads();

    int personal_queue_size = 0;
    int personal_queue[PERSONAL_QUEUE_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numCurrLevelNodes){
        int node = d_currLevelNodes[tid];
        int start = d_nodePtrs[node];
        int end = d_nodePtrs[node + 1];
        for (int i = start; i < end; i++){
            int neighbor = d_nodeNeighbors[i];
            if (d_nodeVisited[neighbor] == 0){
                if (d_nodeVisited[neighbor] == 0 && (atomicCAS(&d_nodeVisited[neighbor], 0, 1)) == 0){
                    int index = -1;
                    if (personal_queue_size < PERSONAL_QUEUE_SIZE){
                        index = personal_queue_size++;
                        personal_queue[index] = neighbor;
                    }
                    else if (*local_queue_size < LOCAL_QUEUE_SIZE){
                        if ((index = atomicAdd_block(local_queue_size, 1)) < LOCAL_QUEUE_SIZE){
                            local_queue[index] = neighbor;
                        }
                    }
                    if (index != -1) continue;
                    index = atomicAdd(numNextLevelNodes, 1);
                    d_nextLevelNodes[index] = neighbor;
                }
            }
        }
    }
    
    // merge personal queue with local queue
    // int local_index = atomicAdd(local_queue_size, personal_queue_size);
    // for (int i = 0; i < personal_queue_size; i++)
    //     local_queue[local_index + i] = personal_queue[i];


    // merge personal queue with global queue
    int local_index = atomicAdd(numNextLevelNodes, personal_queue_size);
    for (int i = 0; i < personal_queue_size; i++)
        d_nextLevelNodes[local_index + i] = personal_queue[i];

    __syncthreads();
    // merge local queue with global queue
    int number_of_nodes_to_merge_per_thread = *local_queue_size / blockDim.x;
    int start_index = threadIdx.x * number_of_nodes_to_merge_per_thread;
    int end_index = start_index + number_of_nodes_to_merge_per_thread;
    if (threadIdx.x == blockDim.x - 1) end_index = *local_queue_size;
    local_index = atomicAdd(numNextLevelNodes, end_index - start_index);
    for (int i = 0; i < end_index - start_index; i++)
        d_nextLevelNodes[local_index + i] = local_queue[start_index + i];
    
}

void kernel_launch (
    int numNodes, 
    int *d_nodePtrs, int *d_nodeNeighbors, 
    int *d_currLevelNodes, int * numCurrentLevelNodes, 
    int *d_nodeVisited, int lws = LWS
    ){

    int numBlocks;
    cudaError_t err;

    int *d_nextLevelNodes;
    int *numNextLevelNodes;
    
    int * h_currentLevelNodes;
    err = cudaMallocHost((void**)&h_currentLevelNodes, numNodes * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    int * h_numCurrentLevelNodes = (int*)malloc(sizeof(int));
    err = cudaMemcpy(h_numCurrentLevelNodes, numCurrentLevelNodes, sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMalloc((void**)&numNextLevelNodes, sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&d_nextLevelNodes, numNodes * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    
    err = cudaMemset(numNextLevelNodes, 0, sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    int level = 0;

    cudaEvent_t start, stop;

    err = cudaEventCreate(&start); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventCreate(&stop); cuda_err_check(err, __FILE__, __LINE__);

    float total_time = 0;

    while (*h_numCurrentLevelNodes > 0){

        numBlocks = round_div_up(*h_numCurrentLevelNodes, lws);

        err = cudaMemset(numNextLevelNodes, 0, sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemset(d_nextLevelNodes, 0, numNodes * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaEventRecord(start); cuda_err_check(err, __FILE__, __LINE__);

        // cout << "Launching kernel with " << numBlocks << " blocks and " << lws << " threads per block" << endl;
        kernel<<<numBlocks, lws, sizeof(int)*(LOCAL_QUEUE_SIZE+1)>>>(numNodes, d_nodePtrs, d_nodeNeighbors, d_currLevelNodes, d_nodeVisited, *h_numCurrentLevelNodes, d_nextLevelNodes, numNextLevelNodes);

        err = cudaEventRecord(stop); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaEventSynchronize(stop); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaMemcpy(numCurrentLevelNodes, numNextLevelNodes, sizeof(int), cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(d_currLevelNodes, d_nextLevelNodes, numNodes * sizeof(int), cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);

        int old_numCurrentLevelNodes = *h_numCurrentLevelNodes;

        err = cudaMemcpy(h_numCurrentLevelNodes, numCurrentLevelNodes, sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(h_currentLevelNodes, d_currLevelNodes, *h_numCurrentLevelNodes * sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

        sort(h_currentLevelNodes, h_currentLevelNodes + *h_numCurrentLevelNodes);

        float milliseconds = 0;
        err = cudaEventElapsedTime(&milliseconds, start, stop); cuda_err_check(err, __FILE__, __LINE__);
        cout << "Time taken for kernel execution: " << milliseconds << " ms " << "with " << numBlocks << " blocks and " << lws << " threads per block" << " at level " << level << " with " << old_numCurrentLevelNodes << " nodes" << " generated " << *h_numCurrentLevelNodes << " nodes" << endl;
        level++;
        total_time += milliseconds;

        // cout << "numCurrentLevelNodes on CPU: " << *h_numCurrentLevelNodes << endl;
        // cout << "Level " << level << " on CPU: ";
        // for (int i = 0; i < *h_numCurrentLevelNodes; i++){
        //     cout << h_currentLevelNodes[i] << " ";
        // }
        // cout << endl;
    }

    cout << "Total time taken for kernel execution: " << total_time << " ms" << endl;

    err = cudaFree(numNextLevelNodes); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_nextLevelNodes); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFreeHost(h_currentLevelNodes); cuda_err_check(err, __FILE__, __LINE__);
}

int main (int argc, char ** argv){
    if (argc != 2){
        cout << "Usage: " << argv[0] << " <graph_file>" << endl;
        exit(1);
    }

    int numNodes, numEdges;
    cudaError_t err;

    int *d_nodePtrs;
    int *d_nodeNeighbors;

    {
        Graph g;
        ifstream infile;
        infile.open(argv[1]);
        infile >> numNodes >> numEdges;
        g.resize(numNodes);
        for (int i = 0; i < numEdges; i++){
            int src, dst;
            infile >> src >> dst;
            g[src].push_back(dst);
            g[dst].push_back(src);
        }
        infile.close();

        err = cudaMalloc((void**)&d_nodePtrs, (numNodes+1) * sizeof(int*)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMalloc((void**)&d_nodeNeighbors, numEdges * 2 * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
     
        int ptr = 0;
        for (int i = 0; i < numNodes; i++){
            err = cudaMemcpy(&d_nodePtrs[i], &ptr, sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
            for (int j = 0; j < g[i].size(); j++){
                err = cudaMemcpy(&d_nodeNeighbors[ptr], &g[i][j], sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
                ptr++;
            }
        }
        err = cudaMemcpy(&d_nodePtrs[numNodes], &ptr, sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    }

    int *d_currLevelNodes;
    int *numCurrLevelNodes;
    int *d_nodeVisited;

    err = cudaMalloc((void**)&d_currLevelNodes, numNodes * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&d_nodeVisited, numNodes * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&numCurrLevelNodes, sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    
    err = cudaMemset(d_currLevelNodes, 0, numNodes * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemset(d_nodeVisited, 0, numNodes * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    int number_of_current_level_nodes = 1;
    int startNode = 0;
    char visited = 1;

    err = cudaMemcpy(numCurrLevelNodes, &number_of_current_level_nodes, sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_currLevelNodes, &startNode, sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(&d_nodeVisited[startNode], &visited, sizeof(char), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    kernel_launch(numNodes, d_nodePtrs, d_nodeNeighbors, d_currLevelNodes, numCurrLevelNodes, d_nodeVisited);

    err = cudaFree(d_nodePtrs); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_nodeNeighbors); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_currLevelNodes); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_nodeVisited); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(numCurrLevelNodes); cuda_err_check(err, __FILE__, __LINE__);

    return 0;
}