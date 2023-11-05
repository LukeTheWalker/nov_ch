#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <utility>
#include <fstream>
#include <algorithm>

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
    int *d_currLevelNodes, int *d_nodeVisited, int * numCurrLevelNodes,
    int *d_nextLevelNodes, int *numNextLevelNodes
    ){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < *numCurrLevelNodes){
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

void kernel_launch (
    int numNodes, 
    int *d_nodePtrs, int *d_nodeNeighbors, 
    int *d_currLevelNodes, int * numCurrentLevelNodes, 
    int *d_nodeVisited, int lws = 256
    ){

    int numBlocks;
    cudaError_t err;

    int *d_nextLevelNodes;
    int *numNextLevelNodes;
    char *something_changed;
    
    int * h_currentLevelNodes;
    err = cudaMallocHost((void**)&h_currentLevelNodes, numNodes * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    int * h_numCurrentLevelNodes = (int*)malloc(sizeof(int));
    err = cudaMemcpy(h_numCurrentLevelNodes, numCurrentLevelNodes, sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMalloc((void**)&numNextLevelNodes, sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&d_nextLevelNodes, numNodes * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&something_changed, sizeof(char)); cuda_err_check(err, __FILE__, __LINE__);
    
    err = cudaMemset(numNextLevelNodes, 0, sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    int level = 0;
    cout << "Level " << level << ": 0" << endl;
    level++;

    cudaEvent_t start, stop;

    err = cudaEventCreate(&start); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventCreate(&stop); cuda_err_check(err, __FILE__, __LINE__);

    float total_time = 0;

    while (*h_numCurrentLevelNodes > 0){

        numBlocks = round_div_up(*h_numCurrentLevelNodes, lws);

        err = cudaMemset(numNextLevelNodes, 0, sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemset(d_nextLevelNodes, 0, numNodes * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemset(something_changed, 0, sizeof(char)); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaEventRecord(start); cuda_err_check(err, __FILE__, __LINE__);

        // cout << "Launching kernel with " << numBlocks << " blocks and " << lws << " threads per block" << endl;
        kernel<<<numBlocks, lws>>>(numNodes, d_nodePtrs, d_nodeNeighbors, d_currLevelNodes, d_nodeVisited, numCurrentLevelNodes, d_nextLevelNodes, numNextLevelNodes);

        err = cudaEventRecord(stop); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaEventSynchronize(stop); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaMemcpy(numCurrentLevelNodes, numNextLevelNodes, sizeof(int), cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(d_currLevelNodes, d_nextLevelNodes, numNodes * sizeof(int), cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaMemcpy(h_numCurrentLevelNodes, numCurrentLevelNodes, sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(h_currentLevelNodes, d_currLevelNodes, *h_numCurrentLevelNodes * sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

        sort(h_currentLevelNodes, h_currentLevelNodes + *h_numCurrentLevelNodes);

        // cout << "numCurrentLevelNodes: " << *h_numCurrentLevelNodes << endl;
        // cout << "Level " << level << ": ";
        // for (int i = 0; i < *h_numCurrentLevelNodes; i++){
        //     cout << h_currentLevelNodes[i] << " ";
        // }
        // cout << endl;
        float milliseconds = 0;
        err = cudaEventElapsedTime(&milliseconds, start, stop); cuda_err_check(err, __FILE__, __LINE__);
        cout << "Time taken for kernel execution: " << milliseconds << " ms " << "with " << numBlocks << " blocks and " << lws << " threads per block" << " at level " << level << " with " << *h_numCurrentLevelNodes << " nodes" << endl;
        level++;
        total_time += milliseconds;
    }

    cout << "Total time taken for kernel execution: " << total_time << " ms" << endl;

    err = cudaFree(numNextLevelNodes); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_nextLevelNodes); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(something_changed); cuda_err_check(err, __FILE__, __LINE__);

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
     
        int * h_nodePtrs = (int*)malloc((numNodes+1) * sizeof(int));
        int * h_nodeNeighbors = (int*)malloc(numEdges * 2 * sizeof(int));

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
        err = cudaMemcpy(d_nodeNeighbors, h_nodeNeighbors, numEdges * 2 * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);err = cudaMemcpy(&d_nodePtrs[numNodes], &ptr, sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
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