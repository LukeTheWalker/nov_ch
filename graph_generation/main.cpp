#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>

using namespace std;

using Graph = vector<vector<int> >;

void fill_barabasi_albert (Graph& g, int n, int m, int m0){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, n-1);
    vector<int> degrees(n, 0);
    for (int i = 0; i < m0; ++i){
        for (int j = i+1; j < m0; ++j){
            g[i].push_back(j);
            g[j].push_back(i);
            degrees[i]++;
            degrees[j]++;
        }
    }
    for (int i = m0; i < n; ++i){
        vector<double> probs(i, 0);
        double sum = 0;
        for (int j = 0; j < i; ++j){
            probs[j] = degrees[j];
            sum += probs[j];
        }
        for (int j = 0; j < i; ++j){
            probs[j] /= sum;
        }
        for (int j = 0; j < m; ++j){
            double r = (double) dis(gen) / dis.max();
            double cumsum = 0;
            for (int k = 0; k < i; ++k){
                cumsum += probs[k];
                if (r < cumsum){
                    g[i].push_back(k);
                    g[k].push_back(i);
                    degrees[i]++;
                    degrees[k]++;
                    break;
                }
            }
        }
        
    }
}

void fill_erdos_renyi_graph(Graph &g, int n, int m){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, n-1);
    for (int i = 0; i < m; ++i){
        int u = dis(gen);
        int v = dis(gen);
        if (u == v) {m--; continue;}
        g[u].push_back(v);
        g[v].push_back(u);
    }
}

void fill_complete_graph(Graph &g, int n){
    for (int i = 0; i < n; ++i){
        for (int j = i+1; j < n; ++j){
            g[i].push_back(j);
            g[j].push_back(i);
        }
    }
}

void write_graph_to_file (Graph &g, string filename){
    ofstream file(filename);
    // write number of vertices and edges
    file << g.size() << " ";
    int edges = 0;
    for (int i = 0; i < g.size(); ++i){
        edges += g[i].size();
    }
    file << edges << "\n";
    for (int i = 0; i < g.size(); ++i){
        for (int j = 0; j < g[i].size(); ++j){
            file << i << " " << g[i][j] << "\n";
        }
    }
    file.close();
}

int main (int argc, char **){
    int n = 1e3;
    int max_edges = 1e4;
    // generate sparse random graph
    Graph g(n);
    fill_erdos_renyi_graph(g, n, max_edges);
    // fill_complete_graph(g, n);
    // fill_barabasi_albert(g, n, max_edges, n/100);
    size_t mem = 0;
    for (int i = 0; i < n; ++i){
        mem += sizeof(g[i]);
        mem += g[i].capacity() * sizeof(int);
    }
    // in megabytes
    cout << "Memory consumption of g: " << mem / 1024 / 1024 << " MB \n";

    // write graph to file
    write_graph_to_file(g, "graph.txt");

}