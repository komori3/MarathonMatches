//#define NDEBUG
#include "bits/stdc++.h"
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <array>
#ifdef _MSC_VER
#include <ppl.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#endif

using namespace std;

#define DUMPOUT cerr
#define dump(...) DUMPOUT<<"  ";DUMPOUT<<#__VA_ARGS__<<" :["<<__LINE__<<":"<<__FUNCTION__<<"]"<<endl;DUMPOUT<<"    ";dump_func(__VA_ARGS__)

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

template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T &val) { std::fill((T*)array, (T*)(array + N), val); }

void dump_func() { DUMPOUT << endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPOUT << head; if (sizeof...(Tail) == 0) { DUMPOUT << " "; } else { DUMPOUT << ", "; } dump_func(std::move(tail)...); }

#define PI 3.14159265358979323846
#define EPS 1e-8
#define FOR(i,a,n) for(int i=(a);i<(n);++i)
#define REP(i,n)  FOR(i,0,n)
#define all(j) (j).begin(), (j).end()
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

constexpr int N = 1000;
constexpr int S = 1000;

struct Point {
  int id;
  int x, y, c;
  Point() {}
  Point(int id, int x, int y, int c) : id(id), x(x), y(y), c(c) {}
};

struct Tree {
  int id;
  vector<vector<int>> G;
  Tree() {}
  Tree(int id, const vector<int>& p) : id(id) {
    int K = p.size();
    G.resize(K);
    for (int j = 1; j < K; j++)
      G[p[j]].push_back(j);
  }
};

struct TestCase {
  Point pts[N];
  vector<int> p[S];
  Tree trees[S];

  TestCase(istream& in) {
    int buf;
    in >> buf >> buf;
    for (int i = 0; i < N; i++) {
      int x, y, c;
      in >> x >> y >> c;
      pts[i] = Point(i, x, y, c);
    }
    for (int i = 0; i < S; i++) {
      int K; in >> K;
      p[i].resize(K, -1);
      for (int j = 1; j < K; j++) {
        in >> p[i][j];
        p[i][j]--;
      }
      trees[i] = Tree(i, p[i]);
    }
  }
};

struct State {
  Point pts[N];
  Tree trees[S];
  vector<int> G[N];

  unordered_set<int> edges;
  unordered_set<int> treeE[S];
  vector<int> assigns[S];
private:
  inline int IJ(int i, int j) {
    return i * N + j;
  }
  bitset<N*N> validE;
  inline void setValidE(int i, int j, bool val) {
    validE[IJ(i, j)] = val;
  }
  inline bool isValidE(int i, int j) {
    return validE[IJ(i, j)];
  }

public:
  State() {}
  State(const TestCase& tc) {
    for (int i = 0; i < N; i++)
      pts[i] = tc.pts[i];
    for (int i = 0; i < S; i++)
      trees[i] = tc.trees[i];
    for (int i = 0; i < N - 1; i++) {
      for (int j = i + 1; j < N; j++) {
        int xi = tc.pts[i].x, yi = tc.pts[i].y, ci = tc.pts[i].c;
        int xj = tc.pts[j].x, yj = tc.pts[j].y, cj = tc.pts[j].c;
        if ((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj) <= (ci + cj) * (ci + cj)) {
          G[i].push_back(j);
          G[j].push_back(i);
          setValidE(i, j, true);
          setValidE(j, i, true);
        }
      }
    }
  }

  struct Elem {
    int gp, gu, u;
    Elem(int gp, int gu, int u) : gp(gp), gu(gu), u(u) {}
  };

  struct EmbedReturnValue {
    bool valid;
    int treeId;
    unordered_set<int> usedE;
    vector<int> assign;
    EmbedReturnValue(bool valid) : valid(valid) {}
    EmbedReturnValue(bool valid, int treeId, const unordered_set<int>& usedE, const vector<int>& assign) 
      : valid(valid), treeId(treeId), usedE(usedE), assign(assign) {}
  };

  EmbedReturnValue canEmbed(int treeIdx, int g0) {
    const Tree& t = trees[treeIdx];
    int K = t.G.size();
    bitset<N> usedV;
    usedV[g0] = true;
    unordered_set<int> usedE;
    vector<int> assign(K, -1);
    assign[0] = g0;

    queue<Elem> qu;
    qu.emplace(-1, g0, 0);
    
    bool valid = true;
    while (!qu.empty()) {
      Elem e = qu.front(); qu.pop();
      int gp = e.gp, gu = e.gu, u = e.u;
      vector<int> gvs;
      for (int gv : G[gu]) {
        if (gv == gp || usedV[gv] || usedE.find(IJ(min(gu, gv), max(gu, gv))) != usedE.end() || !isValidE(gu, gv)) continue;
        usedV[gv] = true;
        gvs.push_back(gv);
        if (gvs.size() >= t.G[u].size()) break;
      }
      if (gvs.size() < t.G[u].size()) {
        valid = false;
        break;
      }
      for (int i = 0; i < t.G[u].size(); i++) {
        int v = t.G[u][i], gv = gvs[i];
        int ge = IJ(min(gu, gv), max(gu, gv));
        //usedV[gv] = true;
        usedE.insert(ge);
        assign[v] = gv;
        qu.emplace(gu, gv, v);
      }
    }
    
    return valid ? EmbedReturnValue(valid, t.id, usedE, assign) : EmbedReturnValue(false);
  }

  void embed(const EmbedReturnValue& erv) {
    auto& a = erv.assign;
    int K = a.size();
    auto& usedE = erv.usedE;
    for (int i = 0; i < K; i++) {
      for (int j = 0; j < N; j++) {
        int e = IJ(min(a[i], j), max(a[i], j));
        if (usedE.find(e) != usedE.end()) {
          edges.insert(e);
          treeE[i].insert(e);
        }
        else {
          // ‡—¬‚ð‹–‚³‚È‚¢
          setValidE(j, a[i], false);
        }
      }
    }
    assigns[erv.treeId] = a;
  }

  void output(ostream& out) {
    out << edges.size() << endl;
    for (const auto& e : edges) {
      out << pts[e / N].id + 1 << " " << pts[e % N].id + 1 << endl;
    }
    for (int i = 0; i < S; i++) {
      int K = trees[i].G.size();
      if (assigns[i].empty()) {
        out << 1;
        for (int j = 2; j <= K; j++) out << " " << j;
        out << endl;
      }
      else {
        auto& a = assigns[i];
        out << pts[a[0]].id + 1;
        for (int j = 1; j < K; j++) out << " " << pts[a[j]].id + 1;
        out << endl;
      }
    }
  }

  void solve() {

    sort(trees, trees + S, [&](const Tree& a, const Tree& b) {
      return a.G.size() < b.G.size();
    });

    int cnt = 0;
    for (int i = 0; i < S; i++) {
      for (int j = 0; j < N; j++) {
        EmbedReturnValue ret = canEmbed(i, j);
        if (ret.valid) {
          embed(ret);
          cnt++;
          break;
        }
      }
    }
    dump(cnt);

    sort(trees, trees + S, [&](const Tree& a, const Tree& b) {
      return a.id < b.id;
    });
  }

  int eval() {
    int score = 0;
    for (int i = 0; i < S; i++) {
      auto& a = assigns[i];
      int K = a.size();
      int invalid = 0;
      for (int u = 0; u < K - 1; u++) {
        for (int v = u + 1; v < K; v++) {
          int e = IJ(min(a[u], a[v]), max(a[u], a[v]));
          if (edges.find(e) != edges.end() && treeE[i].find(e) == treeE[i].end())
            invalid++;
        }
      }
      score += invalid == 0 ? 100 : (invalid == 1 ? 10 : (invalid == 2 ? 1 : 0));
    }
    return score;
  }
};



#define DEBUG_MODE

int main() {
  timer.measure();

  cin.tie(0);
  ios::sync_with_stdio(false);

#ifdef DEBUG_MODE
  ifstream in("example2/example_01.txt");
  TestCase tc(in);
#else
  TestCase tc(cin);
#endif

  State state(tc);
  state.solve();

  dump(state.eval());

#ifdef DEBUG_MODE
  ofstream out("example2/out_01.txt");
  state.output(out);
#else
  state.output(cout);
#endif

  return 0;
}
