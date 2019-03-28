// 1. Convert Challenge 9 format to CSR format
// 2. Find the single-source shortest path solutions

#include "common.hh"

#define VTYPE int // hopefully, weights are integers

// int V; int E;
#define V 39
#define E 85

// adjacency list representation for input
// VTYPE *graph; // VxV: adjacency matrix
map<int, vector<edge_info>> graph; // VxV

// CSR representation of the same graph
int *offset;
edge_info *neighbor; // FIXME: it's size?

// for correctness check
int *correct_dist;

bool compareEdges(edge_info a, edge_info b) {
  return (a.dst_id < b.dst_id);
}

void convert_csr_format() {
  int counter=0;
  // offset[0]=0;
  // string str(csr_file);
  // ofstream csr_file1(str.c_str());
  ofstream csr_file1("datasets/short_99");
  cout << "Starting writing csr file\n";
  if(csr_file1.is_open()) {
    for(auto it=graph.begin(); it!=graph.end(); ++it) { // src vertex
      assert(!it->second.empty());
      offset[it->first]=counter;
      // counter+=it->second.size();
      cout << it->first << " OFFSET: " << offset[it->first] << endl;
      // sort(it->second.begin(),it->second.end());
      sort(it->second.begin(),it->second.end(), compareEdges);
      for(auto it2=it->second.begin(); it2!=it->second.end(); ++it2) { // dst vertex
        edge_info t = *it2;
        // cout << "src_id: " << it->first << " dest_id: " << t.dst_id << endl;
        neighbor[counter++] = t;
        csr_file1 << it->first << " " << t.dst_id << endl; // " " << t.wgt << endl;
      }
    }
    offset[V]=E; // check!
  }
  csr_file1.close();
  cout << "Done writing csr file\n";
}

bool check_correctness(vector<VTYPE> dist) {
  for(int i=0; i<V; ++i) {
    if(dist[i]!=correct_dist[i])
      return false;
  }
  return true;

}

void perform_dijkstra() {
  // bool _is_visited[V]; // this is probably needed for work efficiency

  // timestamp, vertex
  priority_queue< iPair, vector <iPair> , greater<iPair> > pq;
  pq.push(make_pair(0, SRC_LOC));
  
  vector<int> dist(V, MAX_TIMESTAMP-1);
  dist[SRC_LOC]=0;

  while(!pq.empty()) {
    int src_id = pq.top().second;
    pq.pop();
    cout << "COMMIT NEW VERTEX: " << src_id << endl;
    // because no is_visited condition
    for(int i=offset[src_id]; i<offset[src_id+1]; ++i) {
      int dst_id = neighbor[i].dst_id;
      int weight = neighbor[i].wgt;

      // cout << "dst_id: " << dst_id << " weight: " << weight << endl;

      // process+relax step (algo specific)
      int temp_dist = dist[src_id]+weight;
      if(dist[dst_id] > temp_dist) {
        dist[dst_id] = temp_dist;
        pq.push(make_pair(dist[dst_id], dst_id));
      }
    }
  }
  /*
  string str(ans_file);
  ofstream ans_file1(str.c_str());
  // ofstream ans_file("datasets/directed_uniform/rome99_ans");
  cout << "Starting writing ans file: " << str << "\n";
  if(ans_file1.is_open()) {
    for(int i=0; i<V; ++i) {
      correct_dist[i] = dist[i];
      // cout << "Distance at i: " << i << " is: " << dist[i] << endl;
      ans_file1 << dist[i] << endl;
    }
  }
  ans_file1.close();
  */
  cout << "Done writing ans file\n";
}

void perform_graphmat() {
 
  vector<int> dist(V, MAX_TIMESTAMP-1);
  vector<int> temp_dist(V, MAX_TIMESTAMP-1);
  dist[SRC_LOC]=0;
  temp_dist[SRC_LOC]=0;
  queue<int> active_vertex;
  active_vertex.push(SRC_LOC);

  while(!active_vertex.empty()) {
    // for(unsigned i=0; i<active_vertex.size(); ++i) {
    while(!active_vertex.empty()) {
      int src_id = active_vertex.front();
      for(int j=offset[src_id]; j<offset[src_id+1]; ++j) {
        int res = dist[src_id] + neighbor[j].wgt;
        int dest_id = neighbor[j].dst_id;
        temp_dist[dest_id] = min(temp_dist[dest_id],res);
        /*if(temp_dist[dest_id] > res) {
          temp_dist[dest_id] = res;
        }*/
      }
      active_vertex.pop();
    }
    for(int i=0; i<V; ++i) {
      if(temp_dist[i]!=dist[i]) {
        // cout << i << " ";
        assert(temp_dist[i]<dist[i]);
        dist[i]=temp_dist[i];
        active_vertex.push(i);
      }
    }
    cout << "Tasks produced by graphmat: " << active_vertex.size() << endl;
  }
  bool flag = check_correctness(dist);
  if(flag) {
    cout << "GRAPHMAT SUCCESSFULLY DONE!\n";
  } else {
    cout << "GRAPHMAT GAVE INCORRECT RESULTS!\n";
    for(int i=0; i<V; ++i) {
      cout << correct_dist[i] << ":" << dist[i] << endl;
    }
  }  
}

void insert_edge(int src_vertex, edge_info e) {
  // vector<int>::iterator it = graph.find(src_vertex);
  // cout << "Insert edge for src_id: " << src_vertex << " and dst_id: " << e.dst_id << endl;
  auto it = graph.find(src_vertex);
  if(it==graph.end()) { // new src vertex
    vector<edge_info> a; a.push_back(e);
    graph.insert(make_pair(src_vertex,a));
  } else { // old src vertex
    it->second.push_back(e);
    // cout << "New size at src_id: " << src_vertex << " is: " << it->second.size() << endl;
  }
}

int main() {
  // string str(input_file);
  // FILE *file = fopen(str.c_str(),"r");
  // FILE *file = fopen("datasets/directed_uniform/rome99","r");
  FILE *file = fopen("datasets/short.mtx","r");
  char linetoread[5000];

  cout << "start reading graph input file!\n";

#if CSR == 1
  int prev_offset=0;
  int e=-1, prev_v=0; // indices start from 1
#endif
 

  while(fgets(linetoread, 5000, file) != NULL) {
    std::string raw(linetoread);
    std::istringstream iss(raw.c_str());
    char ignore;
    string sp;
    int x, y, z;
    float w; // for the graphs which have decimal weights
    iss >> ignore;
    if(ignore == 'c') continue;
    if(ignore == 'p') {
      // iss >> sp >> V >> E;
      iss >> sp >> x >> y;
      // long long x = long(V)*V*4;
      // graph = (VTYPE*)malloc(V*V*sizeof(VTYPE));
      // graph = (VTYPE*)malloc(x);
      // cout << "Allocated to graph\n";
      offset = (int*)malloc((V+1)*sizeof(int));
#if CSR == 1
      offset[0]=0;
#endif
      neighbor = (edge_info*)malloc(E*sizeof(edge_info));
      cout << "Allocated to edge info\n";
      correct_dist = (VTYPE*)malloc(V*sizeof(VTYPE));
      // cout << "Allocated to correct distance\n";
      // memset(graph, 0, V*V*sizeof(VTYPE));
      // memset(graph, 0, x);
      memset(offset, 0, (V+1)*sizeof(int));
      memset(neighbor, 0, E*sizeof(edge_info));
      continue;
    } else {
#if UNDIRECTED == 1 // and unweighted (0-based notation)
      iss >> x >> y;
      --x; --y;
      cout << "x: " << x << " y: " << y << endl;
      if((x>=V)||(y>=V)) {
        cout << "x: " << x << " y: " << y << endl;
      }
      assert(x<V); assert(y<V);
      int z = 1; rand()%256;
      // edge_info e1(y,z);
      // edge_info e2(x,z);
      edge_info e1; e1.dst_id=y; e1.wgt=z;
      edge_info e2; e2.dst_id=x; e2.wgt=z;
      // edge_info e2(x,z);

      insert_edge(x,e1);
      /*if(x!=y) {
        insert_edge(y,e2);
      }*/
#elif CSR == 1 // assume unweighted for now
      // read in original data structures directly: offset, neighbor
      iss >> x >> y;
      // --x; --y; // assuming 1 index
      cout << "IN csr, x: " << x << " y: " << y << endl;
      neighbor[++e].dst_id=y; // dst according to 0 index
      neighbor[e].wgt=1;
      if(x!=prev_v) {
        offset[prev_v+1]=e;
        // cout << (prev_v+1) << " OFFSET: " << e << endl;
        int k=prev_v+1;
        while(offset[--k]==0 && k>0) {
          offset[k]=prev_offset;
        }
        prev_offset=e;
        prev_v=x;
      // cout << "index: " << (src) << " value: " << e << endl;
    }
#else
      // cout << "came to read each edge: " << linetoread << "\n";
      // 1 based-notation
      // iss >> x >> y >> z;
      // iss >> x >> y >> w;
      // z = 1000*(float)w;

      // for web-data graph to generate random weights
      // iss >> x >> y;
      iss >> y >> x;
      z = rand()%256;


      cout << "x: " << x << " y: " << y << " z: " << z << endl;
     
      if((x-1>=V)||(y-1>=V)) {
        cout << "x: " << x << " y: " << y << endl;
      }
      assert(x-1<V); assert(y-1<V);
      if(z<0) z=-z; // for negative edge, dijkstra doesn't work for negative edge
      edge_info e; e.dst_id=y-1; e.wgt=z;
      insert_edge(x-1,e);
      
      
      // if((x>=V)||(y>=V)) {
      //   cout << "x: " << x << " y: " << y << endl;
      // }
      // assert(x<V); assert(y<V);
      // if(z<0) z=-z; // for negative edge, dijkstra doesn't work for negative edge
      // edge_info e; e.dst_id=y; e.wgt=z;
      // insert_edge(x,e);
      
#endif
    }
  }
 
  fclose(file);
  cout << "Done reading graph file!\n";

#if CSR == 1
  offset[V] = E;
  int k=V;
  while(offset[--k]==0 && k>0) { // offset[0] should be 0
    offset[k]=prev_offset;
    // cout << "Setting for k: " << k << endl;
  }
  cout << "Done doing last things for the csr format\n";
#endif
#if CSR == 0
  convert_csr_format();
#endif
  // perform_dijkstra(); 
  // perform_graphmat(); 
}
 
