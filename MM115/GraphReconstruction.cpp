#include "bits/stdc++.h"
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#ifdef _MSC_VER
#include <ppl.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#endif

using namespace std;

#define DUMPOUT cerr
#define dump(...) DUMPOUT<<"  ";DUMPOUT<<#__VA_ARGS__<<" :["<<__LINE__<<":"<<__FUNCTION__<<"]"<<endl;DUMPOUT<<"    ";dump_func(__VA_ARGS__)
#define dumptime() dump(timer.time() - timer.t)

typedef unsigned uint; typedef long long ll; typedef unsigned long long ull; typedef pair<int, int> pii; typedef pair<ll, ll> pll; typedef pair<double, double> pdd; typedef pair<string, string> pss;
template <typename _KTy, typename _Ty> ostream& operator << (ostream& o, const pair<_KTy, _Ty>& m) { o << "{" << m.first << ", " << m.second << "}"; return o; }
template <typename _KTy, typename _Ty> ostream& operator << (ostream& o, const map<_KTy, _Ty>& m) { if (m.empty()) { o << "{ }"; return o; } o << "{" << *m.begin(); for (auto itr = ++m.begin(); itr != m.end(); itr++) { o << ", " << *itr; } o << "}"; return o; }
template <typename _KTy, typename _Ty> ostream& operator << (ostream& o, const unordered_map<_KTy, _Ty>& m) { if (m.empty()) { o << "{ }"; return o; } o << "{" << *m.begin(); for (auto itr = ++m.begin(); itr != m.end(); itr++) { o << ", " << *itr; } o << "}"; return o; }
template <typename _Ty> ostream& operator << (ostream& o, const vector<_Ty>& v) { if (v.empty()) { o << "{ }"; return o; } o << "{" << v.front(); for (auto itr = ++v.begin(); itr != v.end(); itr++) { o << ", " << *itr; } o << "}"; return o; }
template <typename _Ty> ostream& operator << (ostream& o, const set<_Ty>& s) { if (s.empty()) { o << "{ }"; return o; } o << "{" << *(s.begin()); for (auto itr = ++s.begin(); itr != s.end(); itr++) { o << ", " << *itr; } o << "}"; return o; }
template <typename _Ty> ostream& operator << (ostream& o, const unordered_set<_Ty>& s) { if (s.empty()) { o << "{ }"; return o; } o << "{" << *(s.begin()); for (auto itr = ++s.begin(); itr != s.end(); itr++) { o << ", " << *itr; }	o << "}"; return o; }
template <typename _Ty> ostream& operator << (ostream& o, const stack<_Ty>& s) { if (s.empty()) { o << "{ }"; return o; } stack<_Ty> t(s); o << "{" << t.top(); t.pop(); while (!t.empty()) { o << ", " << t.top(); t.pop(); } o << "}";	return o; }
template <typename _Ty> ostream& operator << (ostream& o, const list<_Ty>& l) { if (l.empty()) { o << "{ }"; return o; } o << "{" << l.front(); for (auto itr = ++l.begin(); itr != l.end(); ++itr) { o << ", " << *itr; } o << "}"; return o; }
template <typename _KTy, typename _Ty> istream& operator >> (istream& is, pair<_KTy, _Ty>& m) { is >> m.first >> m.second; return is; }
template <typename _Ty> istream& operator >> (istream& is, vector<_Ty>& v) { for (size_t i = 0; i < v.size(); i++) is >> v[i]; return is; }
namespace aux { // print tuple
  template<typename Ty, unsigned N, unsigned L> struct tp { static void print(ostream& os, const Ty& v) { os << get<N>(v) << ", "; tp<Ty, N + 1, L>::print(os, v); } };
  template<typename Ty, unsigned N> struct tp<Ty, N, N> { static void print(ostream& os, const Ty& v) { os << get<N>(v); } };
}

template<typename... Tys> ostream& operator<<(ostream& os, const tuple<Tys...>& t) { os << "{"; aux::tp<tuple<Tys...>, 0, sizeof...(Tys) - 1>::print(os, t); os << "}"; return os; }

template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T& val) { std::fill((T*)array, (T*)(array + N), val); }

void dump_func() { DUMPOUT << endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPOUT << head; if (sizeof...(Tail) == 0) { DUMPOUT << " "; } else { DUMPOUT << ", "; } dump_func(std::move(tail)...); }

#define PI 3.14159265358979323846
#define EPS 1e-10
#define FOR(i,a,n) for(int i=(a);i<(n);++i)
#define REP(i,n)  FOR(i,0,n)
//#define all(j) (j).begin(), (j).end()
#define SZ(j) ((int)(j).size())
#define fake false

class Timer {
public:
  double t = 0;
  Timer() {}
  static double time() {
#ifdef _MSC_VER
    return __rdtsc() / 2.8e9;
#else
    unsigned long long a, d;
    __asm__ volatile("rdtsc"
      : "=a"(a), "=d"(d));
    return (d << 32 | a) / 2.8e9;
#endif
  }
  void measure() { t = time() - t; }
  double elapsedMs() { return (time() - t) * 1000.0; }
} timer;

struct Xorshift {
  uint64_t x = 88172645463325252LL;
  unsigned nextUInt() {
    x = x ^ (x << 7);
    return x = x ^ (x >> 9);
  }
  unsigned nextUInt(unsigned mod) {
    x = x ^ (x << 7);
    x = x ^ (x >> 9);
    return x % mod;
  }
  unsigned nextUInt(unsigned l, unsigned r) {
    x = x ^ (x << 7);
    x = x ^ (x >> 9);
    return x % (r - l + 1) + l;
  }
  double nextDouble() {
    return double(nextUInt()) / UINT_MAX;
  }
} rnd;

template<typename T>
void shuffleVector(vector<T>& v, Xorshift& rnd) {
  int n = v.size();
  for (int i = n - 1; i >= 1; i--) {
    int r = rnd.nextUInt(i);
    swap(v[i], v[r]);
  }
}

template<typename T>
string to_string(const vector<T>& v) {
  if (v.empty()) return "[]";
  string ret = "[";
  ret += to_string(v[0]);
  for (int i = 1; i < (int)v.size(); i++) ret += "," + to_string(v[i]);
  ret += "]";
  return ret;
}

//#define ORACLE_MODE
#ifdef ORACLE_MODE
#undef NDEBUG
#endif
#ifdef _MSC_VER
namespace Vis {
  cv::Mat_<cv::Vec3b> graphVis(const vector<vector<int>>&);
}
#endif

using VType = bitset<100>;
using DType = uint8_t;
constexpr DType UNKNOWN = 0;
constexpr DType NOT_CONNECTED = 0x7F;
constexpr DType CONNECTED = 1;



struct DisjointSet {
  vector<int> data;
  DisjointSet() {}
  DisjointSet(int size) : data(size, -1) {}
  bool unite(int x, int y) {
    x = root(x); y = root(y);
    if (x != y) {
      if (data[y] < data[x]) swap(x, y);
      data[x] += data[y]; data[y] = x;
    }
    return x != y;
  }
  bool same(int x, int y) {
    return root(x) == root(y);
  }
  int root(int x) {
    return data[x] < 0 ? x : data[x] = root(data[x]);
  }
  int size(int x) {
    return -data[root(x)];
  }
};

map<int, vector<int>> createDisjointMap(DisjointSet& ds) {
  int N = ds.data.size();
  map<int, vector<int>> ret;
  for (int i = 0; i < N; i++) {
    ret[ds.root(i)].push_back(i);
  }
  return ret;
}

struct Path {
  int id, u, v, d;
  vector<int> p;
  Path(int id, const string& line) : id(id) {
    istringstream iss(line);
    int u, v, d;
    iss >> u >> v >> d;
    this->u = u;
    this->v = v;
    this->d = (d == -1 ? NOT_CONNECTED : d);
  }
  Path(int id, int u, int v, int d) : id(id), u(u), v(v), d(d) {}
  bool operator==(const Path& other) const {
    return id == other.id;
  }
  string toString() const {
    return "[id = " + to_string(id) + ", u = " + to_string(u) + ", v = " + to_string(v) + ", d = " + to_string(d) + ", p = " + to_string(p) + "]";
  }
};
ostream& operator<<(ostream& o, const Path& p) {
  o << p.toString();
  return o;
}

// 状態に依存しない情報
int N, K;
double C;
vector<Path> paths;
#ifdef ORACLE_MODE
vector<string> oracle;
#endif

// 高速化用
namespace Tune {
  class FastQueue {
    static constexpr int cap = 1 << 14;
    int front, back;
    DType v[cap];
  public:
    FastQueue() : front(0), back(0) {}
    inline bool empty() { return front == back; }
    inline void push(DType x) { v[front++] = x; }
    inline DType pop() { return v[back++]; }
    inline void reset() { front = back = 0; }
    inline int size() { return front - back; }
  } fqu;
  DType* dist_;
  DType** dist;
  DType* cdist_;
  void setDist() {
    dist_ = new DType[N * N];
    cdist_ = new DType[N * N];
    dist = new DType * [N];
    for (int i = 0; i < N; i++) dist[i] = &dist_[i * N];
  }
  void clearDist(int val) {
    memset(dist_, val, sizeof(DType) * N * N);
  }
}

struct State {
  enum struct TType : uint8_t {
    SWAP = 0, ADD = 1, REMOVE = 2
  };
  int connected;
  int disconnected;
  DisjointSet ds;
  vector<vector<bool>> fixed;
  vector<vector<DType>> adjMat;
  vector<vector<DType>> adjList;
  vector<pii> notFixedEdge;
  vector<pii> notFixedSpace;

  State() {}

  void reset() {
    Tune::setDist();

    connected = 0;
    disconnected = 0;
    ds = DisjointSet(N);
    fixed = vector<vector<bool>>(N, vector<bool>(N, false));
    for (int i = 0; i < N; i++) fixed[i][i] = true;
    adjMat = vector<vector<DType>>(N, vector<DType>(N, UNKNOWN));
    for (int i = 0; i < N; i++) adjMat[i][i] = NOT_CONNECTED;
    adjList = vector<vector<DType>>(N);
  }

  inline void connect(int u, int v) {
    // u-v 間にパスを作る
    // 距離の更新は、u, v を含む部分グラフにのみ行えばよい(要検証)
    if (adjMat[u][v] != CONNECTED) connected++;
    adjMat[u][v] = adjMat[v][u] = CONNECTED;
  }

  inline void disconnect(int u, int v) {
    if (adjMat[u][v] != NOT_CONNECTED) disconnected++;
    adjMat[u][v] = adjMat[v][u] = NOT_CONNECTED;
  }

  inline void addEdge(int u, int v) {
    adjList[u].push_back(v);
    adjList[v].push_back(u);
  }

  inline void removeEdge(int u, int v) {
    adjList[u].erase(find(adjList[u].begin(), adjList[u].end(), v));
    adjList[v].erase(find(adjList[v].begin(), adjList[v].end(), u));
  }

  void fixPaths() {
    // fix された辺の情報を用いてパスを fix
    // connected な辺のみで APSP を解いて、1 より大きい距離のパスについて距離が一致するなら確定
    // TODO: adjMat[i][j] != NOT_CONNECTED として、d=2 のものを探索するような解法
    vector<vector<int>> dist(N, vector<int>(N, NOT_CONNECTED));
    vector<vector<int>> prev(N, vector<int>(N, -1));
    for (DType u = 0; u < N; u++) {
      VType visited;
      Tune::fqu.reset();
      visited[u] = true;
      dist[u][u] = 0;
      Tune::fqu.push(u);
      while (!Tune::fqu.empty()) {
        DType v = Tune::fqu.pop();
        for (DType w : adjList[v]) {
          if (visited[w]) continue;
          visited[w] = true;
          dist[u][w] = dist[u][v] + 1;
          prev[u][w] = v;
          Tune::fqu.push(w);
        }
      }
    }

    for (auto& path : paths) {
      if (path.d == 1) continue;
      if (path.d == NOT_CONNECTED) continue;
      int u = path.u, v = path.v, d = path.d;
      if (dist[u][v] == d) {
        int w = v;
        vector<int> p;
        while (prev[u][w] != -1) {
          p.push_back(w);
          w = prev[u][w];
        }
        p.push_back(u);
        reverse(p.begin(), p.end());
        // path を短くするような辺は NOT_CONNECTED にする
        for (int i = 0; i < (int)p.size() - 2; i++) {
          for (int j = i + 2; j < p.size(); j++) {
            int x = p[i], y = p[j];
            disconnect(x, y);
            fixed[x][y] = fixed[y][x] = true;
#ifdef ORACLE_MODE
            assert(oracle[x][y] == '0');
#endif
          }
        }
        path.p = p;
      }
    }
  }

  void calcInitialState() {
    // 自明な部分を確定させた初期解を求める
    for (auto& path : paths) {
      int u = path.u, v = path.v, d = path.d;
      if (d == NOT_CONNECTED) {
        disconnect(u, v);
        fixed[u][v] = fixed[v][u] = true;
#ifdef ORACLE_MODE
        assert(oracle[u][v] == '0');
#endif
      }
      else {
        ds.unite(u, v);
        // ホップ数 1 なら (u, v) は必ず接続
        if (d == 1) {
          connect(u, v);
          addEdge(u, v);
          fixed[u][v] = fixed[v][u] = true;
          path.p.push_back(u); path.p.push_back(v);
#ifdef ORACLE_MODE
          assert(oracle[u][v] == '1');
#endif
        }
        // ホップ数 2 以上の場合、辺 (u, v) が張られることはない
        else {
          disconnect(u, v);
          fixed[u][v] = fixed[v][u] = true;
#ifdef ORACLE_MODE
          assert(oracle[u][v] == '0');
#endif
        }
      }
    }
    auto dmap = createDisjointMap(ds);
    for (const auto& path : paths) {
      int u = path.u, v = path.v, d = path.d;
      if (d != NOT_CONNECTED) continue;
      // (u, v, -1) -> u の含まれる集合と v の含まれる集合は disjoint
      auto us = dmap[ds.root(u)];
      auto vs = dmap[ds.root(v)];
      for (int u0 : us) for (int v0 : vs) {
        disconnect(u0, v0);
        fixed[u0][v0] = fixed[v0][u0] = true;
#ifdef ORACLE_MODE
        assert(oracle[u0][v0] == '0');
#endif
      }
    }
  }

  void fillUnknownEdges() {
    int total = N * (N - 1) / 2;
    vector<pii> unknownEdges;
    for (int i = 0; i < N; i++) {
      for (int j = i + 1; j < N; j++) {
        if (adjMat[i][j] == UNKNOWN) {
          unknownEdges.emplace_back(i, j);
        }
      }
    }

    if (unknownEdges.empty()) return;

    int r = min((int)unknownEdges.size(), int(C * total - connected));

    // unknownEdges から r 個の辺を選んで張る
    shuffleVector(unknownEdges, rnd);
    for (int i = 0; i < r; i++) {
      int u, v;
      tie(u, v) = unknownEdges.back();
      unknownEdges.pop_back();
      connect(u, v);
      addEdge(u, v);
    }
    // それ以外は張らない
    for (const auto& e : unknownEdges) {
      int u, v;
      tie(u, v) = e;
      disconnect(u, v);
    }

    for (int i = 0; i < N - 1; i++) {
      for (int j = i + 1; j < N; j++) {
        if (!fixed[i][j]) {
          if (adjMat[i][j] == CONNECTED) {
            notFixedEdge.emplace_back(i, j);
          }
          else {
            notFixedSpace.emplace_back(i, j);
          }
        }
      }
    }
  }

  VType enumSubGraphVertices(DType u) {
    VType visited;
    Tune::fqu.reset();
    visited[u] = true;
    Tune::fqu.push(u);
    while (!Tune::fqu.empty()) {
      DType v = Tune::fqu.pop();
      for (DType w : adjList[v]) {
        if (visited[w]) continue;
        visited[w] = true;
        Tune::fqu.push(w);
      }
    }
    return visited;
  }

  inline void sssp(DType u) {
    if (adjList[u].empty()) return;
    VType visited;
    Tune::fqu.reset();
    visited[u] = true;
    Tune::dist[u][u] = 0;
    Tune::fqu.push(u);
    while (!Tune::fqu.empty()) {
      DType v = Tune::fqu.pop();
      for (DType w : adjList[v]) {
        if (visited[w]) continue;
        visited[w] = true;
        Tune::dist[u][w] = Tune::dist[w][u] = Tune::dist[u][v] + 1;
        Tune::fqu.push(w);
      }
    }
  }

  inline void apsp() {
    Tune::clearDist(NOT_CONNECTED);
    for (DType u = 0; u < N; u++) sssp(u);
  }

  vector<vector<int>> getDistanceMatrix(vector<vector<int>> adj) {
    int n = adj.size();
    for (int k = 0; k < n; k++)
      for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
          adj[i][j] = min(adj[i][j], adj[i][k] + adj[k][j]);
    return adj;
  }

  int evaluate() {
    int score = 0;
    int connected = 0;
    for (const auto& path : paths) {
      int u = path.u, v = path.v, d = path.d;
      if (d == NOT_CONNECTED) {
        if (Tune::dist[u][v] != NOT_CONNECTED) score -= 100;
        else score += 1;
      }
      else {
        if (Tune::dist[u][v] == NOT_CONNECTED) score -= 100;
        else {
          int dd = abs(Tune::dist[u][v] - d);
          score += (dd == 0 ? 1 : -dd);
        }
      }
    }
    return score;
  }

  int getPathScore() {
    int score = 0;
    // path score
    for (const auto& path : paths) {
      int u = path.u, v = path.v, d = path.d;
      score += (Tune::dist[u][v] == d ? 1 : 0);
      if (Tune::dist[u][v] != d) {
        dump(u, v, d, int(Tune::dist[u][v]));
      }
    }
    return score;
  }

  void addRandomEdge() {
    if (notFixedSpace.empty()) return;
    int fidx = rnd.nextUInt(notFixedSpace.size());
    auto f = notFixedSpace[fidx];
    swap(notFixedSpace[fidx], notFixedSpace.back());
    notFixedSpace.pop_back();
    notFixedEdge.push_back(f);

    int fu, fv;
    tie(fu, fv) = f;
    connect(fu, fv);
    addEdge(fu, fv);
    // update dist
    auto vs = enumSubGraphVertices(fu);
    for (int u = 0; u < N; u++) if (vs[u]) sssp(u);
  }

  void undoAdd() {
    if (notFixedEdge.empty()) return;
    auto e = notFixedEdge.back();
    notFixedEdge.pop_back();
    notFixedSpace.push_back(e);
    int eu, ev;
    tie(eu, ev) = e;
    disconnect(eu, ev);
    removeEdge(eu, ev);
  }

  void removeRandomEdge() {
    if (notFixedEdge.empty()) return;
    int eidx = rnd.nextUInt(notFixedEdge.size());
    auto e = notFixedEdge[eidx];
    swap(notFixedEdge[eidx], notFixedEdge.back());
    notFixedEdge.pop_back();
    notFixedSpace.push_back(e);

    int eu, ev;
    tie(eu, ev) = e;
    VType vs = enumSubGraphVertices(eu);
    disconnect(eu, ev);
    removeEdge(eu, ev);

    // update dist
    for (DType u = 0; u < N; u++) if (vs[u])
      for (DType v = 0; v < N; v++)
        Tune::dist[u][v] = Tune::dist[v][u] = NOT_CONNECTED;
    for (DType u = 0; u < N; u++) if (vs[u]) sssp(u);
  }

  void undoRemove() {
    if (notFixedSpace.empty()) return;
    auto f = notFixedSpace.back();
    notFixedSpace.pop_back();
    notFixedEdge.push_back(f);
    int fu, fv;
    tie(fu, fv) = f;
    connect(fu, fv);
    addEdge(fu, fv);
  }

  VType swapRandomEdge() {
    if (notFixedEdge.empty() || notFixedSpace.empty()) {
      return VType();
    }
    int eidx = rnd.nextUInt(notFixedEdge.size());
    int fidx = rnd.nextUInt(notFixedSpace.size());
    auto e = notFixedEdge[eidx];
    swap(notFixedEdge[eidx], notFixedEdge.back());
    auto f = notFixedSpace[fidx];
    swap(notFixedSpace[fidx], notFixedSpace.back());

    swap(notFixedEdge.back(), notFixedSpace.back());
    int eu, ev, fu, fv;
    tie(eu, ev) = e;
    tie(fu, fv) = f;
    disconnect(eu, ev);
    removeEdge(eu, ev);
    connect(fu, fv);
    addEdge(fu, fv);
    // update dist
    apsp();
  }

  void undoSwap() {
    if (notFixedEdge.empty() || notFixedSpace.empty()) return;
    auto e = notFixedEdge.back();
    auto f = notFixedSpace.back();
    swap(notFixedEdge.back(), notFixedSpace.back());
    int eu, ev, fu, fv;
    tie(eu, ev) = e;
    tie(fu, fv) = f;
    disconnect(eu, ev);
    removeEdge(eu, ev);
    connect(fu, fv);
    addEdge(fu, fv);
  }

  void undo(TType type) {
    switch (type) {
    case State::TType::ADD:
      undoAdd();
      break;
    case State::TType::REMOVE:
      undoRemove();
      break;
    default:
      break;
    }
    memcpy(Tune::dist_, Tune::cdist_, sizeof(DType) * N * N);
  }

  double getTemp(double startTemp, double endTemp, double t, double T, double deg = 1.0) {
    return endTemp + (startTemp - endTemp) * pow((T - t) / T, deg);
  }

  vector<vector<DType>> solveSA() {
    int bestScore = INT_MIN;
    auto bestAdjMat = adjMat;
    auto bestAdjList = adjList;
    int numLoop = 0;
    double start = timer.elapsedMs();
    double T = 9800 - start;
    const unsigned RR = (1 << 30), mask = (1 << 30) - 1;

    while (timer.elapsedMs() - start < T) {
      int r = rnd.nextUInt(2);
      State::TType type;
      if (r < 1) type = State::TType::ADD;
      else if (r < 2) type = State::TType::REMOVE;
      else type = State::TType::SWAP;

      int prevScore = evaluate();
      transition(type);
      int score = evaluate();

      int diff = score - prevScore;
      double temp = getTemp(3, 0.1, timer.elapsedMs() - start, T);
      double prob = exp(diff / temp);

      if (RR * prob > (rnd.nextUInt() & mask)) {
        if (bestScore < score) {
          bestScore = score;
          bestAdjMat = adjMat;
          bestAdjList = adjList;
          //cerr << bestScore << "/" << paths.size() << " " << (int)timer.elapsedMs() << endl;
        }
      }
      else {
        undo(type);
      }
      numLoop++;
    }

    adjMat = bestAdjMat;
    adjList = bestAdjList;
    apsp();
    dump(numLoop, getPathScore(), paths.size());
    return bestAdjMat;
  }

  void transition(TType type) {
    memcpy(Tune::cdist_, Tune::dist_, sizeof(DType) * N * N);
    VType vs;
    switch (type) {
    case State::TType::ADD:
      addRandomEdge();
      break;
    case State::TType::REMOVE:
      removeRandomEdge();
      break;
    case State::TType::SWAP:
      swapRandomEdge();
      break;
    default:
      break;
    }
  }
};

class GraphReconstruction {
public:
  vector<string> findSolution(int N_, double C_, int K_, vector<string> paths_) {
    timer.measure();
    // initialize
    N = N_;
    C = C_;
    K = K_;

    for (int id = 0; id < paths_.size(); id++) {
      paths.emplace_back(id, paths_[id]);
    }

    State state;
    state.reset();
    state.calcInitialState();
    state.fixPaths();
    state.fillUnknownEdges();
    state.apsp();

    auto adj = state.solveSA();

    vector<string> ans(N, string(N, '0'));
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        if (adj[i][j] == CONNECTED) {
          ans[i][j] = '1';
        }
      }
    }

    return ans;
  }
};

int main() {
  //ifstream ifs("case2.txt");
  //istream& in = ifs;
  istream& in = cin;

  GraphReconstruction prog;
  int N;
  double C;
  int K;
  int NumPaths;
  vector<string> paths;
  string path;
  in >> N;
  //cerr << N << endl;
  in >> C;
  //cerr << C << endl;
  in >> K;
  //cerr << K << endl;
  in >> NumPaths;
  //cerr << NumPaths << endl;
  for (int i = 0; i < NumPaths; i++) {
    getline(in, path);
    if (path.empty()) {
      i--;
      continue;
    }
    paths.push_back(path);
    //cerr << path << endl;
  }
#ifdef ORACLE_MODE
  for (int i = 0; i < N; i++) {
    getline(in, path);
    if (path.empty()) {
      i--;
      continue;
    }
    oracle.push_back(path);
  }
#endif

  vector<string> ret = prog.findSolution(N, C, K, paths);
  cout << ret.size() << endl;
  for (int i = 0; i < (int)ret.size(); ++i)
    cout << ret[i] << endl;
  cout.flush();

  return 0;
}

#ifdef _MSC_VER
namespace Vis {
  constexpr int HEIGHT = 600;
  constexpr int WIDTH = 600;
  constexpr int INF = 1000000;
  cv::Point2i cvtD2I(const cv::Point2d p) {
    int ix = (int)round(p.x + WIDTH * 0.5);
    int iy = (int)round(HEIGHT * 0.5 - p.y);
    return cv::Point2i(ix, iy);
  }
  cv::Mat_<cv::Vec3b> graphVis(const vector<vector<int>>& adjMat) {
    int N = adjMat.size();
    cv::Mat_<cv::Vec3b> img(HEIGHT, WIDTH, cv::Vec3b(255, 255, 255));
    vector<cv::Point2d> pts;
    int r = 250;
    for (int i = 0; i < N; i++) {
      pts.emplace_back(r * cos(-2 * PI * i / N), r * sin(-2 * PI * i / N));
    }
    vector<cv::Point> ipts;
    for (const auto& p : pts) ipts.push_back(cvtD2I(p));
    for (const auto& p : ipts) cv::circle(img, p, 2, cv::Scalar(0, 0, 0), cv::FILLED);
    for (int i = 0; i < N - 1; i++) {
      for (int j = 0; j < N; j++) {
        if (adjMat[i][j] == CONNECTED) {
          cv::line(img, ipts[i], ipts[j], cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }
      }
    }
    return img;
  }
  vector<vector<int>> randomGraph(int n, double c) {
    while (true) {
      int cnt = 0;
      vector<vector<int>> adj(n, vector<int>(n, INF));
      for (int i = 0; i < n; i++) {
        adj[i][i] = 0;
        for (int j = i + 1; j < n; j++) {
          if (rnd.nextDouble() < c) {
            adj[i][j] = adj[j][i] = 1;
            cnt++;
          }
        }
      }
      if (cnt == 0 || cnt == n * (n - 1) / 2) continue;
      return adj;
    }
  }
  vector<vector<int>> getDistanceMatrix(vector<vector<int>> adj) {
    int n = adj.size();
    for (int k = 0; k < n; k++)
      for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
          if (adj[i][k] + adj[k][j] < adj[i][j])
            adj[i][j] = adj[i][k] + adj[k][j];
    return adj;
  }

  cv::Mat_<cv::Vec3b> graphVis2(const vector<vector<int>>& pAdjMat, const vector<vector<int>>& nAdjMat, pii edge) {
    cv::imshow("img", graphVis(pAdjMat));
    cv::waitKey(0);
    return cv::Mat();
  }
  vector<vector<int>> createAdjList(const vector<vector<int>>& adjMat) {
    int n = adjMat.size();
    vector<vector<int>> adjList(n);
    for (int i = 0; i < n - 1; i++) {
      for (int j = i + 1; j < n; j++) {
        if (adjMat[i][j] == 1) {
          adjList[i].push_back(j);
          adjList[j].push_back(i);
        }
      }
    }
    return adjList;
  }
  VType enumSubGraphVertices(const vector<vector<int>>& adjMat, int u) {
    auto adjList = createAdjList(adjMat);
    VType visited;
    queue<int> qu;
    visited[u] = true;
    qu.push(u);
    while (!qu.empty()) {
      int v = qu.front(); qu.pop();
      for (int w : adjList[v]) {
        if (visited[w]) continue;
        visited[w] = true;
        qu.push(w);
      }
    }
    return visited;
  }
  void apspDiff(vector<vector<int>>& dist, const vector<vector<int>>& adj, VType vs) {
    int n = adj.size();
    auto adjList = createAdjList(adj);
    for (int u = 0; u < n; u++) if (vs[u]) {
      if (adjList[u].empty()) continue;
      VType visited;
      queue<int> qu;
      visited[u] = true;
      dist[u][u] = 0;
      qu.push(u);
      while (!qu.empty()) {
        int v = qu.front(); qu.pop();
        for (int w : adjList[v]) {
          if (visited[w]) continue;
          visited[w] = true;
          dist[u][w] = dist[w][u] = dist[u][v] + 1;
          qu.push(w);
        }
      }
    }
  }
  bool verifyDistanceMatrixAddEdge(vector<vector<int>> adj, pii edge) {
    int u, v;
    tie(u, v) = edge;
    assert(adj[u][v] == INF);

    auto prev(graphVis(adj));

    // 辺を削除する前の dist
    auto prevDist = getDistanceMatrix(adj);
    // 辺追加
    adj[u][v] = adj[v][u] = 1;
    // 部分グラフの頂点を列挙
    VType vs = enumSubGraphVertices(adj, u);
    // ナイーブな wf
    auto naiveDist = getDistanceMatrix(adj);

    auto diffDist(prevDist);
    // 差分計算
    apspDiff(diffDist, adj, vs);

    auto now(graphVis(adj));

    // verify
    int n = adj.size();
    bool valid = true;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (naiveDist[i][j] != diffDist[i][j]) {
          dump(i, j, naiveDist[i][j], diffDist[i][j]);
          valid = false;
        }
      }
    }
    if (!valid) {
      dump(edge);
      dump(vs);
      cerr << "adj" << endl;
      for (auto v : adj) cerr << v << endl;
      for (int i = 0; i < n; i++) {
        cerr << i << endl;
        cerr << naiveDist[i] << endl;
        cerr << diffDist[i] << endl;
      }
      cv::imshow("prev", prev);
      cv::imshow("now", now);
      cv::waitKey(0);
    }
    return valid;
  }
  bool verifyDistanceMatrixRemoveEdge(vector<vector<int>> adj, pii edge) {
    int u, v;
    tie(u, v) = edge;
    assert(adj[u][v] == 1);

    auto prev(graphVis(adj));

    // 部分グラフの頂点を列挙
    VType vs = enumSubGraphVertices(adj, u);
    // 辺を削除する前の dist
    auto prevDist = getDistanceMatrix(adj);
    // 辺削除
    adj[u][v] = adj[v][u] = INF;
    // ナイーブな wf
    auto naiveDist = getDistanceMatrix(adj);

    auto diffDist(prevDist);

    int n = diffDist.size();

    // 前処理: vs の情報をクリア
    for (int u = 0; u < n; u++) if (vs[u]) {
      for (int v = 0; v < n; v++) diffDist[u][v] = diffDist[v][u] = INF;
    }

    // 差分計算
    apspDiff(diffDist, adj, vs);

    auto now(graphVis(adj));

    // verify
    bool valid = true;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (i == j) continue;
        if (naiveDist[i][j] != diffDist[i][j]) {
          dump(i, j, naiveDist[i][j], diffDist[i][j]);
          valid = false;
        }
      }
    }
    if (!valid) {
      dump(edge);
      dump(vs);
      cerr << "adj" << endl;
      for (auto v : adj) cerr << v << endl;
      for (int i = 0; i < n; i++) {
        cerr << i << endl;
        cerr << naiveDist[i] << endl;
        cerr << diffDist[i] << endl;
      }
      cv::imshow("prev", prev);
      cv::imshow("now", now);
      cv::waitKey(0);
    }
    return valid;
  }
  bool verifyDistanceMatrixSwapEdge(vector<vector<int>> adj, pii delEdge, pii addEdge) {
    int du, dv, au, av;
    tie(du, dv) = delEdge;
    tie(au, av) = addEdge;
    assert(adj[du][dv] == 1);
    assert(adj[au][av] == INF);

    auto prev = graphVis(adj);

    // 辺削除前の部分グラフの頂点を列挙
    VType vs = enumSubGraphVertices(adj, du);
    // 辺変更前の dist
    auto prevDist = getDistanceMatrix(adj);
    // 辺変更
    adj[du][dv] = adj[dv][du] = INF;
    adj[au][av] = adj[av][au] = 1;
    // 辺変更後の部分グラフの頂点を列挙
    vs |= enumSubGraphVertices(adj, au);
    // ナイーブな wf
    auto naiveDist = getDistanceMatrix(adj);

    auto diffDist(prevDist);
    // 差分計算
    apspDiff(diffDist, adj, vs);

    auto now = graphVis(adj);

    // verify
    int n = adj.size();
    bool valid = true;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (naiveDist[i][j] != diffDist[i][j]) {
          dump(i, j, naiveDist[i][j], diffDist[i][j]);
          valid = false;
        }
      }
    }
    if (!valid) {
      dump(delEdge, addEdge);
      dump(vs);
      for (int i = 0; i < n; i++) {
        cerr << i << endl;
        cerr << naiveDist[i] << endl;
        cerr << diffDist[i] << endl;
      }
      cv::imshow("prev", prev);
      cv::imshow("now", now);
      cv::waitKey(0);
    }
    return valid;
  }
}
int _main() {
  // 辺を追加・削除したときに
  // 1. apsp を計算する
  // 2. 差分更新する
  // の結果を比較する
  using namespace Vis;

  for (int i = 0; i < 1000; i++) {
    dump(i);
    int n = rnd.nextUInt(91) + 10;
    double c = rnd.nextDouble() * 0.9 + 0.1;
    auto g = randomGraph(n, c);

    int du, dv, au, av;
    while (true) {
      //du = rnd.nextUInt(n);
      //dv = rnd.nextUInt(n);
      //if (du == dv) continue;
      //if (g[du][dv] == INF) continue;
      au = rnd.nextUInt(n);
      av = rnd.nextUInt(n);
      if (au == av) continue;
      if (g[au][av] == 1) continue;
      break;
    }

    //assert(verifyDistanceMatrixRemoveEdge(g, pii(du, dv)));
    assert(verifyDistanceMatrixAddEdge(g, pii(au, av)));
  }

  return 0;
}
#endif
