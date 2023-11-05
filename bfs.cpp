#include <iostream>
#include <fstream>
#include <vector>
#include <string.h>
#include <chrono>

using namespace std;
using Graph = vector<vector<int> >;

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

int main (int argc, char **argv){
    if (argc != 2){
        cout << "Usage: " << argv[0] << " <graph_file>" << endl;
        exit(1);
    }

    int numNodes, numEdges;

    int *d_nodePtrs;
    int *d_nodeNeighbors;

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

    int * nodePtrs = (int*)malloc((numNodes+1) * sizeof(int));
    int * nodeNeighbors = (int*)malloc(numEdges * 2 * sizeof(int));

    int ptr = 0;
    for (int i = 0; i < numNodes; i++){
        nodePtrs[i] = ptr;
        for (int j = 0; j < g[i].size(); j++){
            nodeNeighbors[ptr] = g[i][j];
            ptr++;
        }
    }
    nodePtrs[numNodes] = ptr;

    int * currLevelNodes = (int*)malloc(numNodes * sizeof(int));
    int * nextLevelNodes = (int*)malloc(numNodes * sizeof(int));
    bool * visitedNodes = (bool*)malloc(numNodes * sizeof(bool));
    memset(visitedNodes, false, numNodes * sizeof(bool));
    memset(currLevelNodes, 0, numNodes * sizeof(int));
    memset(nextLevelNodes, 0, numNodes * sizeof(int));
    int numCurrLevelNodes = 1;
    int numNextLevelNodes = 0;

    int level = 0;

    float total_time = 0.0;

    while (numCurrLevelNodes > 0){
        numNextLevelNodes = 0;
        int old_numCurrLevelNodes = numCurrLevelNodes;
        // profile each step
        auto start = chrono::high_resolution_clock::now();
        single_step(numNodes, nodePtrs, nodeNeighbors, currLevelNodes, nextLevelNodes, visitedNodes, numCurrLevelNodes, numNextLevelNodes);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<float> duration = end - start;
        total_time += duration.count();
        swap(currLevelNodes, nextLevelNodes);
        cout << "At level " << level << " with " << old_numCurrLevelNodes << " nodes" << " generated " << numCurrLevelNodes << " nodes" << " in " << duration.count() * 1000 << "ms" << endl;
    }

    cout << "Total time: " << total_time * 1000 << "ms" << endl;

    return 0;
}