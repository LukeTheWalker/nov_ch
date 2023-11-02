#include <iostream>
#include <fstream>
#include <vector>
#include <queue>

using namespace std;
using Graph = vector<vector<int> >;

int main (int argc, char **argv){
    if (argc != 2){
        cout << "Usage: " << argv[0] << " <graph_file>" << endl;
        exit(1);
    }

    int numNodes, numEdges;

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
        }
        infile.close();

        // implement BFS
        queue<int> q;
        vector<bool> visited(numNodes, false);
        vector<int> level(numNodes, 0);

        // start BFS from node 0
        q.push(0);
        visited[0] = true;

        while (!q.empty()) {
            int curr = q.front();
            q.pop();
            for (int neighbor : g[curr]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    level[neighbor] = level[curr] + 1;
                    q.push(neighbor);
                }
            }
        }

        // print all nodes at each level
        for (int i = 0; i < numNodes; i++) {
            bool found = false;
            cout << "Level " << i << ": ";
            for (int j = 0; j < numNodes; j++) {
                if (level[j] == i) {
                    cout << j << " ";
                    found = true;
                }
            }
            cout << endl;
            if (!found) break;
        }
    }

    return 0;
}