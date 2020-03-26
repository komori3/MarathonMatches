//#define NDEBUG
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
  double elapsed_ms() { return (time() - t) * 1000.0; }
} timer;

struct Xorshift {
  uint64_t x = 88172645463325252LL;
  unsigned next_int() {
    x = x ^ (x << 7);
    return x = x ^ (x >> 9);
  }
  unsigned next_int(unsigned mod) {
    x = x ^ (x << 7);
    x = x ^ (x >> 9);
    return x % mod;
  }
  unsigned next_int(unsigned l, unsigned r) {
    x = x ^ (x << 7);
    x = x ^ (x >> 9);
    return x % (r - l + 1) + l;
  }
  double next_double() {
    return double(next_int()) / UINT_MAX;
  }
} rnd;

template<typename T>
void shuffle_vector(vector<T>& v, Xorshift& rnd) {
  int n = v.size();
  for (int i = n - 1; i >= 1; i--) {
    int r = rnd.next_int(i);
    swap(v[i], v[r]);
  }
}



struct Board {
  double P;
  int N;
  int H, W;
  vector<string> base;
  vector<pii> offsets;
  vector<vector<string>> layers;
  int T;
  int diff;

  static Board get_random_board(double P, int N, int H, int W, const vector<vector<string>>& grids) {
    Board board;
    board.P = P;
    board.N = N;
    board.H = H;
    board.W = W;
    board.T = 0;
    board.diff = 0;
    for (const auto& grid : grids) {
      int h = grid.size(), w = grid[0].size();
      int r = rnd.next_int(H - h + 1);
      int c = rnd.next_int(W - w + 1);
      board.add_layer(grid, r, c);
    }
    board.set_base();
    return board;
  }

  Board() {}
  Board(double P, int N, int H, int W, const vector<vector<string>>& grids) : P(P), N(N), H(H), W(W), T(0), diff(0) {
    for (const auto& grid : grids) add_layer(grid, 0, 0);
    set_base();
  }
  void add_layer(const vector<string>& grid, int r, int c) {
    int h = grid.size(), w = grid[0].size();
    vector<string> layer(H, string(W, 0));
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        int y = i + r;
        int x = j + c;
        layer[y][x] = grid[i][j];
        T++;
      }
    }
    layers.push_back(layer);
    offsets.emplace_back(r, c);
  }
  void set_base() {
    base.resize(H, string(W, 'A'));
    for (int i = 0; i < H; i++) {
      for (int j = 0; j < W; j++) {
        vector<char> chars;
        for (int n = 0; n < N; n++) {
          if (layers[n][i][j]) chars.push_back(layers[n][i][j]);
        }
        if (!chars.empty()) {
          sort(chars.begin(), chars.end());
          // â‘Î•Î·˜a‚Í’†‰›’l‚ÅÅ¬‰»‚³‚ê‚é
          char med = chars[chars.size() / 2];
          for (char c : chars) diff += abs(c - med);
          base[i][j] = med;
        }
      }
    }
  }
  double get_score() {
    double compression_score = double(H * W) / T;
    double lossiness_score = diff / (12.5 * T);
    return compression_score * P + lossiness_score * (1 - P);
  }
  vector<string> get_ans() {
    vector<string> ans(base);
    for (const auto& offset : offsets) {
      ans.push_back(to_string(offset.first) + " " + to_string(offset.second));
    }
    return ans;
  }
};

class Lossy2dCompression {
public:
  vector<string> findSolution(double P, int N, vector< vector<string> > grids) {
    timer.measure();
    int H = 0;
    int W = 0;
    for (auto grid : grids) {
      H = max(H, (int)grid.size());
      W = max(W, (int)grid[0].size());
    }

    Board best_board(P, N, H, W, grids);
    int interval = 990;
    for (int ext = 1; ext <= 10; ext++) {
      while (timer.elapsed_ms() < interval * ext) {
        Board board = Board::get_random_board(P, N, H + (10 - ext), W + (10 - ext), grids);
        if (board.get_score() < best_board.get_score()) {
          best_board = board;
          //cerr << best_board.get_score() << endl;
        }
      }
    }
    return best_board.get_ans();
  }
};



int main() {
  double P;
  cin >> P;
  int N;
  cin >> N;
  vector< vector<string> > grids;
  for (int i = 0; i < N; i++) {
    int H;
    cin >> H;
    vector<string> grid(H);
    for (int y = 0; y < H; y++) {
      cin >> grid[y];
    }
    grids.push_back(grid);
  }

  Lossy2dCompression prog;
  vector<string> ret = prog.findSolution(P, N, grids);
  cout << ret.size() - N << endl;
  for (int i = 0; i < ret.size(); i++) {
    cout << ret[i] << endl;
  }
  cout.flush();
}
