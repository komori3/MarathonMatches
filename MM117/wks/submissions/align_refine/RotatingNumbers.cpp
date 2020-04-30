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

using uint = unsigned; using ll = long long; using ull = unsigned long long; using pii = pair<int, int>; using pll = pair<ll, ll>; using pdd = pair<double, double>; using pss = pair<string, string>;
template <typename _KTy, typename _Ty> ostream& operator << (ostream& o, const pair<_KTy, _Ty>& m) { o << "{" << m.first << ", " << m.second << "}"; return o; }
template <typename _KTy, typename _Ty> ostream& operator << (ostream& o, const map<_KTy, _Ty>& m) { if (m.empty()) { o << "{ }"; return o; } o << "{" << *m.begin(); for (auto itr = ++m.begin(); itr != m.end(); itr++) { o << ", " << *itr; } o << "}"; return o; }
template <typename _KTy, typename _Ty> ostream& operator << (ostream& o, const unordered_map<_KTy, _Ty>& m) { if (m.empty()) { o << "{ }"; return o; } o << "{" << *m.begin(); for (auto itr = ++m.begin(); itr != m.end(); itr++) { o << ", " << *itr; } o << "}"; return o; }
template <typename _Ty> ostream& operator << (ostream& o, const vector<_Ty>& v) { if (v.empty()) { o << "{ }"; return o; } o << "{" << v.front(); for (auto itr = ++v.begin(); itr != v.end(); itr++) { o << ", " << *itr; } o << "}"; return o; }
template <typename _Ty> ostream& operator << (ostream& o, const deque<_Ty>& v) { if (v.empty()) { o << "{ }"; return o; } o << "{" << v.front(); for (auto itr = ++v.begin(); itr != v.end(); itr++) { o << ", " << *itr; } o << "}"; return o; }
template <typename _Ty> ostream& operator << (ostream& o, const set<_Ty>& s) { if (s.empty()) { o << "{ }"; return o; } o << "{" << *(s.begin()); for (auto itr = ++s.begin(); itr != s.end(); itr++) { o << ", " << *itr; } o << "}"; return o; }
template <typename _Ty> ostream& operator << (ostream& o, const unordered_set<_Ty>& s) { if (s.empty()) { o << "{ }"; return o; } o << "{" << *(s.begin()); for (auto itr = ++s.begin(); itr != s.end(); itr++) { o << ", " << *itr; }	o << "}"; return o; }
template <typename _Ty> ostream& operator << (ostream& o, const stack<_Ty>& s) { if (s.empty()) { o << "{ }"; return o; } stack<_Ty> t(s); o << "{" << t.top(); t.pop(); while (!t.empty()) { o << ", " << t.top(); t.pop(); } o << "}";	return o; }
template <typename _Ty> ostream& operator << (ostream& o, const list<_Ty>& l) { if (l.empty()) { o << "{ }"; return o; } o << "{" << l.front(); for (auto itr = ++l.begin(); itr != l.end(); ++itr) { o << ", " << *itr; } o << "}"; return o; }
template <typename _KTy, typename _Ty> istream& operator >> (istream& is, pair<_KTy, _Ty>& m) { is >> m.first >> m.second; return is; }
template <typename _Ty> istream& operator >> (istream& is, vector<_Ty>& v) { for (size_t t = 0; t < v.size(); t++) is >> v[t]; return is; }
template <typename _Ty> istream& operator >> (istream& is, deque<_Ty>& v) { for (size_t t = 0; t < v.size(); t++) is >> v[t]; return is; }
namespace aux { // print tuple
  template<typename Ty, unsigned N, unsigned L> struct tp { static void print(ostream& os, const Ty& v) { os << get<N>(v) << ", "; tp<Ty, N + 1, L>::print(os, v); } };
  template<typename Ty, unsigned N> struct tp<Ty, N, N> { static void print(ostream& os, const Ty& v) { os << get<N>(v); } };
}

template<typename... Tys> ostream& operator<<(ostream& os, const tuple<Tys...>& t) { os << "{"; aux::tp<tuple<Tys...>, 0, sizeof...(Tys) - 1>::print(os, t); os << "}"; return os; }

template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T& val) { fill((T*)array, (T*)(array + N), val); }

void dump_func() { DUMPOUT << endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPOUT << head; if (sizeof...(Tail) == 0) { DUMPOUT << " "; } else { DUMPOUT << ", "; } dump_func(move(tail)...); }

#define PI 3.14159265358979323846
#define EPS 1e-8
#define rep(t,n) for(int t=0;t<(n);++t)
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

// https://natsugiri.hatenablog.com/entry/2016/10/10/035445
template<class T> struct DEPQ {
  vector<T> d;
  DEPQ() {}
  DEPQ(const vector<T>& d_) : d(d_) { make_heap(); }
  template<class Iter> DEPQ(Iter first, Iter last) : d(first, last) { make_heap(); }
  void push(const T& x) {
    int k = d.size();
    d.push_back(x);
    up(k);
  }
  void pop_min() {
    if (d.size() < 3u) {
      d.pop_back();
    }
    else {
      swap(d[1], d.back()); d.pop_back();
      int k = down(1);
      up(k);
    }
  }
  void pop_max() {
    if (d.size() < 2u) {
      d.pop_back();
    }
    else {
      swap(d[0], d.back()); d.pop_back();
      int k = down(0);
      up(k);
    }
  }
  const T& get_min() const {
    return d.size() < 2u ? d[0] : d[1];
  }
  const T& get_max() const {
    return d[0];
  }
  int size() const { return d.size(); }
  bool empty() const { return d.empty(); }
  void make_heap() {
    for (int i = d.size(); i--; ) {
      if (i & 1 && d[i - 1] < d[i]) swap(d[i - 1], d[i]);
      int k = down(i);
      up(k, i);
    }
  }
  inline int parent(int k) const {
    return ((k >> 1) - 1) & ~1;
  }
  int down(int k) {
    int n = d.size();
    if (k & 1) { // min heap
      while (2 * k + 1 < n) {
        int c = 2 * k + 3;
        if (n <= c || d[c - 2] < d[c]) c -= 2;
        if (c < n && d[c] < d[k]) { swap(d[k], d[c]); k = c; }
        else break;
      }
    }
    else { // max heap
      while (2 * k + 2 < n) {
        int c = 2 * k + 4;
        if (n <= c || d[c] < d[c - 2]) c -= 2;
        if (c < n && d[k] < d[c]) { swap(d[k], d[c]); k = c; }
        else break;
      }
    }
    return k;
  }
  int up(int k, int root = 1) {
    if ((k | 1) < (int)d.size() && d[k & ~1] < d[k | 1]) {
      swap(d[k & ~1], d[k | 1]);
      k ^= 1;
    }
    int p;
    while (root < k && d[p = parent(k)] < d[k]) { // max heap
      swap(d[p], d[k]);
      k = p;
    }
    while (root < k && d[k] < d[p = parent(k) | 1]) { // min heap
      swap(d[p], d[k]);
      k = p;
    }
    return k;
  }
};



int score_table[41];
void init_score_table() {
  for (int s = 2; s <= 40; s++) {
    score_table[s] = int(floor(pow(s - 1, 1.5)));
  }
}

namespace NRP {
  constexpr int N = 4;
  using H = ull;;
  struct Board {
    shared_ptr<Board> parent;
    int g[N][N];
    int move_cost;
    int loc_cost;
    int r, c, s;
    char dir;
    Board(const vector<int>& grid_) : move_cost(0), loc_cost(0) {
      parent = nullptr;
      assert(grid_.size() == N * N);
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          g[i][j] = grid_[i * N + j];
        }
      }
      loc_cost = calc_loc_cost();
      r = c = s = dir = -1;
    }
    inline int get_distance(int i, int j) const {
      int c = g[i][j];
      return abs(i - c / N) + abs(j - c % N);
    }
    int calc_loc_cost() const {
      int cost = 0;
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          cost += get_distance(i, j);
        }
      }
      return cost;
    }
    // R
    void move_cw(int r, int c, int s) {
      int a = (s + 1) >> 1;
      int b = s >> 1;
      for (int y = 0; y < a; y++) {
        for (int x = 0; x < b; x++) {
          g[y + r][x + c] = g[y + r][x + c] ^ g[s - 1 - x + r][y + c];
          g[s - 1 - x + r][y + c] = g[y + r][x + c] ^ g[s - 1 - x + r][y + c];
          g[y + r][x + c] = g[y + r][x + c] ^ g[s - 1 - x + r][y + c];

          g[s - 1 - x + r][y + c] = g[s - 1 - x + r][y + c] ^ g[s - 1 - y + r][s - 1 - x + c];
          g[s - 1 - y + r][s - 1 - x + c] = g[s - 1 - x + r][y + c] ^ g[s - 1 - y + r][s - 1 - x + c];
          g[s - 1 - x + r][y + c] = g[s - 1 - x + r][y + c] ^ g[s - 1 - y + r][s - 1 - x + c];

          g[s - 1 - y + r][s - 1 - x + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[x + r][s - 1 - y + c];
          g[x + r][s - 1 - y + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[x + r][s - 1 - y + c];
          g[s - 1 - y + r][s - 1 - x + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[x + r][s - 1 - y + c];
        }
      }
    }
    // L
    void move_ccw(int r, int c, int s) {
      int a = (s + 1) >> 1;
      int b = s >> 1;
      for (int y = 0; y < a; y++) {
        for (int x = 0; x < b; x++) {
          g[y + r][x + c] = g[y + r][x + c] ^ g[x + r][s - 1 - y + c];
          g[x + r][s - 1 - y + c] = g[y + r][x + c] ^ g[x + r][s - 1 - y + c];
          g[y + r][x + c] = g[y + r][x + c] ^ g[x + r][s - 1 - y + c];

          g[x + r][s - 1 - y + c] = g[x + r][s - 1 - y + c] ^ g[s - 1 - y + r][s - 1 - x + c];
          g[s - 1 - y + r][s - 1 - x + c] = g[x + r][s - 1 - y + c] ^ g[s - 1 - y + r][s - 1 - x + c];
          g[x + r][s - 1 - y + c] = g[x + r][s - 1 - y + c] ^ g[s - 1 - y + r][s - 1 - x + c];

          g[s - 1 - y + r][s - 1 - x + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[s - 1 - x + r][y + c];
          g[s - 1 - x + r][y + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[s - 1 - x + r][y + c];
          g[s - 1 - y + r][s - 1 - x + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[s - 1 - x + r][y + c];
        }
      }
    }

    void move(int r, int c, int s, char dir) {
      dir == 'L' ? move_ccw(r, c, s) : move_cw(r, c, s);
      move_cost += score_table[s];
      loc_cost = calc_loc_cost();
      this->r = r;
      this->c = c;
      this->s = s;
      this->dir = dir;
    }

    void print() const {
      fprintf(stderr, "--------------------\n");
      fprintf(stderr, "move_cost: %d\nloc_cost: %d\nr: %d, c: %d, s: %d, dir: %c\n", move_cost, loc_cost, r, c, s, dir);
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          fprintf(stderr, "%3d ", g[i][j]);
        }
        fprintf(stderr, "\n");
      }
      fprintf(stderr, "--------------------\n");
    }

    inline bool is_inside(int y, int x) const {
      return 0 <= y && y < N && 0 <= x && x < N;
    }

    vector<shared_ptr<Board>> get_all_next_states() const {
      vector<shared_ptr<Board>> next_states;
      for (int s = 2; s <= N; s++) {
        for (int r = 0; r <= N - s; r++) {
          for (int c = 0; c <= N - s; c++) {
            for (char dir : {'L', 'R'}) {
              shared_ptr<Board> nb = make_shared<Board>(*this);
              nb->move(r, c, s, dir);
              next_states.push_back(nb);
            }
          }
        }
      }
      return next_states;
    }

    double get_cost() const {
      return move_cost + loc_cost / 2.0;
    }

    H get_hash() const {
      H hash(0);
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          hash |= H(g[i][j]);
          hash <<= N;
        }
      }
      return hash;
    }
  };
  using BoardPtr = shared_ptr<Board>;

  vector<BoardPtr> solve(const vector<int>& grid) {
    BoardPtr ib = make_shared<Board>(grid);

    auto cmp = [](const BoardPtr& b1, const BoardPtr& b2) {
      return b1->get_cost() > b2->get_cost();
    };
    priority_queue<BoardPtr, vector<BoardPtr>, decltype(cmp)> pq(cmp);
    unordered_map<H, BoardPtr> hashes;
    pq.push(ib);
    hashes[ib->get_hash()] = ib;

    int cnt = 0;
    while (!pq.empty()) {
      BoardPtr b = pq.top(); pq.pop();
      if (b->loc_cost == 0 || timer.elapsedMs() > 9800) {
        vector<BoardPtr> ans;
        while (b != nullptr) {
          ans.push_back(b);
          b = b->parent;
        }
        reverse(all(ans));
        return ans;
      }
      for (const BoardPtr nb : b->get_all_next_states()) {
        H hash = nb->get_hash();
        if (hashes.count(hash) && hashes[hash]->move_cost <= nb->move_cost) continue;
        pq.push(nb);
        hashes[hash] = nb;
        nb->parent = b;
      }
    }
    return vector<BoardPtr>();
  }

}

struct Point {
  int y, x;
  Point() {}
  Point(initializer_list<int> init) : y(*init.begin()), x(*(init.begin() + 1)) {}
  Point(int y, int x) : y(y), x(x) {}
  bool operator==(const Point& rhs) const {
    return this->x == rhs.x && this->y == rhs.y;
  }
  bool operator!=(const Point& rhs) const {
    return !(*this == rhs);
  }
  friend ostream& operator<<(ostream& o, const Point& p) {
    o << "{" << p.y << "," << p.x << "}";
    return o;
  }
};

struct Move {
  int r, c, s;
  char dir;
  int prev_loc_penalty; // 直前の loc_penalty
  int now_penalty;
  Move() {}
  Move(int r, int c, int s, char dir, int prev_loc_penalty, int now_penalty) : r(r), c(c), s(s), dir(dir), prev_loc_penalty(prev_loc_penalty), now_penalty(now_penalty) {}
  string toString() const {
    return to_string(r) + " " + to_string(c) + " " + to_string(s) + " " + dir;
  }
  friend ostream& operator<<(ostream& o, const Move& m) {
    o << m.toString();
    return o;
  }
};

struct State {
  int N, P;
  vector<vector<int>> g;

  vector<Move> moves;
  int move_penalty;
  int loc_penalty;
  Point src, now, dst;
  // taboo は外から受け取る
  State(int N, int P, const vector<vector<int>>& g, int move_penalty, int loc_penalty, const Point& src, const Point& dst)
    : N(N), P(P), g(g), move_penalty(move_penalty), loc_penalty(loc_penalty), src(src), now(src), dst(dst) {}

  inline bool is_inside(int y, int x) const {
    return 0 <= y && y < N && 0 <= x && x < N;
  }

  inline int get_distance(int i, int j) const {
    int c = g[i][j];
    return abs(i - c / N) + abs(j - c % N);
  }

  int calc_loc_penalty(int r, int c, int s) const {
    int loc_penalty = 0;
    for (int y = r; y < r + s; y++) {
      for (int x = c; x < c + s; x++) {
        loc_penalty += get_distance(y, x);
      }
    }
    return loc_penalty;
  }

  int get_penalty() const {
    return move_penalty + loc_penalty * P;
  }

  // R
  void move_cw(int r, int c, int s) {
    int a = (s + 1) >> 1;
    int b = s >> 1;
    for (int y = 0; y < a; y++) {
      for (int x = 0; x < b; x++) {
        g[y + r][x + c] = g[y + r][x + c] ^ g[s - 1 - x + r][y + c];
        g[s - 1 - x + r][y + c] = g[y + r][x + c] ^ g[s - 1 - x + r][y + c];
        g[y + r][x + c] = g[y + r][x + c] ^ g[s - 1 - x + r][y + c];

        g[s - 1 - x + r][y + c] = g[s - 1 - x + r][y + c] ^ g[s - 1 - y + r][s - 1 - x + c];
        g[s - 1 - y + r][s - 1 - x + c] = g[s - 1 - x + r][y + c] ^ g[s - 1 - y + r][s - 1 - x + c];
        g[s - 1 - x + r][y + c] = g[s - 1 - x + r][y + c] ^ g[s - 1 - y + r][s - 1 - x + c];

        g[s - 1 - y + r][s - 1 - x + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[x + r][s - 1 - y + c];
        g[x + r][s - 1 - y + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[x + r][s - 1 - y + c];
        g[s - 1 - y + r][s - 1 - x + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[x + r][s - 1 - y + c];
      }
    }
  }
  // L
  void move_ccw(int r, int c, int s) {
    int a = (s + 1) >> 1;
    int b = s >> 1;
    for (int y = 0; y < a; y++) {
      for (int x = 0; x < b; x++) {
        g[y + r][x + c] = g[y + r][x + c] ^ g[x + r][s - 1 - y + c];
        g[x + r][s - 1 - y + c] = g[y + r][x + c] ^ g[x + r][s - 1 - y + c];
        g[y + r][x + c] = g[y + r][x + c] ^ g[x + r][s - 1 - y + c];

        g[x + r][s - 1 - y + c] = g[x + r][s - 1 - y + c] ^ g[s - 1 - y + r][s - 1 - x + c];
        g[s - 1 - y + r][s - 1 - x + c] = g[x + r][s - 1 - y + c] ^ g[s - 1 - y + r][s - 1 - x + c];
        g[x + r][s - 1 - y + c] = g[x + r][s - 1 - y + c] ^ g[s - 1 - y + r][s - 1 - x + c];

        g[s - 1 - y + r][s - 1 - x + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[s - 1 - x + r][y + c];
        g[s - 1 - x + r][y + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[s - 1 - x + r][y + c];
        g[s - 1 - y + r][s - 1 - x + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[s - 1 - x + r][y + c];
      }
    }
  }
  void move(int r, int c, int s, char dir) {
    //dump(r, c, s, dir);
    int prev_loc_penalty = loc_penalty;
    loc_penalty -= calc_loc_penalty(r, c, s);
    dir == 'L' ? move_ccw(r, c, s) : move_cw(r, c, s);
    loc_penalty += calc_loc_penalty(r, c, s);
    move_penalty += score_table[s];
    moves.emplace_back(r, c, s, dir, prev_loc_penalty, get_penalty());
  }
  void move(const Move& m) {
    move(m.r, m.c, m.s, m.dir);
  }

  void undo() {
    const Move& m = moves.back();
    m.dir == 'L' ? move_cw(m.r, m.c, m.s) : move_ccw(m.r, m.c, m.s);
    loc_penalty = m.prev_loc_penalty;
    move_penalty -= score_table[m.s];
    moves.pop_back();
  }

  void add_next_states(int width, vector<DEPQ<shared_ptr<State>>>& pqs, const vector<vector<bool>>& taboo) {
    int sdist = abs(src.y - dst.y) + abs(src.x - dst.x);
    int dist = abs(now.y - dst.y) + abs(now.x - dst.x);
    auto prev = now;
    for (int s = 2; s <= N; s++) {
      // dy, dx: オフセットからの位置
      for (int dy = 0; dy < s; dy++) {
        for (int dx = 0; dx < s; dx++) {
          // 外周のみ
          if (!(dy == 0 || dy == s - 1) || !(dx == 0 || dx == s - 1)) continue;
          int oy = now.y - dy, ox = now.x - dx;
          // inside
          if (!is_inside(oy, ox) || !is_inside(oy + s - 1, ox + s - 1)) continue;
          // taboo
          bool ok = true;
          for (int y = oy; y < oy + s; y++) {
            for (int x = ox; x < ox + s; x++) {
              if (taboo[y][x]) {
                ok = false;
                break;
              }
            }
            if (!ok) break;
          }
          if (!ok) continue;
          // left move
          int ly = oy + s - dx - 1, lx = ox + dy;
          // dist を s-1 減らす(最も近づく)もののみ許容
          int ndist;
          ndist = abs(dst.y - ly) + abs(dst.x - lx);
          if (ndist == dist - (s - 1)) {
            move(oy, ox, s, 'L');
            now = Point(ly, lx);
            if (pqs[sdist - ndist].size() < width) {
              shared_ptr<State> nstate = make_shared<State>(*this);
              pqs[sdist - ndist].push(nstate);
            }
            else if(get_penalty() < pqs[sdist - ndist].get_min()->get_penalty()) {
              shared_ptr<State> nstate = make_shared<State>(*this);
              pqs[sdist - ndist].push(nstate);
              pqs[sdist - ndist].pop_min();
            }
            now = prev;
            undo();
          }
          int ry = oy + dx, rx = ox + s - dy - 1;
          ndist = abs(dst.y - ry) + abs(dst.x - rx);
          if (ndist == dist - (s - 1)) {
            move(oy, ox, s, 'R');
            now = Point(ry, rx);
            if (pqs[sdist - ndist].size() < width) {
              shared_ptr<State> nstate = make_shared<State>(*this);
              pqs[sdist - ndist].push(nstate);
            }
            else if (get_penalty() < pqs[sdist - ndist].get_min()->get_penalty()) {
              shared_ptr<State> nstate = make_shared<State>(*this);
              pqs[sdist - ndist].push(nstate);
              pqs[sdist - ndist].pop_min();
            }
            now = prev;
            undo();
          }
        }
      }
    }
  }
};
using StatePtr = shared_ptr<State>;

bool operator<(const StatePtr& a, const StatePtr& b) {
  return a->get_penalty() > b->get_penalty();
}

struct Solver {

  int N, P;
  vector<vector<int>> g;

  vector<Move> moves;
  int move_penalty;
  int loc_penalty;

  Solver(int N_, int P_, const vector<int>& grid_) : N(N_), P(P_), move_penalty(0), loc_penalty(0) {
    g.resize(N, vector<int>(N));
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        g[i][j] = grid_[i * N + j] - 1; // to 0-indexed
      }
    }
    loc_penalty += calc_loc_penalty(0, 0, N);
  }

  inline int get_distance(int i, int j) const {
    int c = g[i][j];
    return abs(i - c / N) + abs(j - c % N);
  }
  int calc_loc_penalty(int r, int c, int s) const {
    int loc_penalty = 0;
    for (int y = r; y < r + s; y++) {
      for (int x = c; x < c + s; x++) {
        loc_penalty += get_distance(y, x);
      }
    }
    return loc_penalty;
  }
  int get_penalty() {
    return move_penalty + loc_penalty * P;
  }
  // R
  void move_cw(int r, int c, int s) {
    int a = (s + 1) >> 1;
    int b = s >> 1;
    for (int y = 0; y < a; y++) {
      for (int x = 0; x < b; x++) {
        g[y + r][x + c] = g[y + r][x + c] ^ g[s - 1 - x + r][y + c];
        g[s - 1 - x + r][y + c] = g[y + r][x + c] ^ g[s - 1 - x + r][y + c];
        g[y + r][x + c] = g[y + r][x + c] ^ g[s - 1 - x + r][y + c];

        g[s - 1 - x + r][y + c] = g[s - 1 - x + r][y + c] ^ g[s - 1 - y + r][s - 1 - x + c];
        g[s - 1 - y + r][s - 1 - x + c] = g[s - 1 - x + r][y + c] ^ g[s - 1 - y + r][s - 1 - x + c];
        g[s - 1 - x + r][y + c] = g[s - 1 - x + r][y + c] ^ g[s - 1 - y + r][s - 1 - x + c];

        g[s - 1 - y + r][s - 1 - x + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[x + r][s - 1 - y + c];
        g[x + r][s - 1 - y + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[x + r][s - 1 - y + c];
        g[s - 1 - y + r][s - 1 - x + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[x + r][s - 1 - y + c];
      }
    }
  }
  // L
  void move_ccw(int r, int c, int s) {
    int a = (s + 1) >> 1;
    int b = s >> 1;
    for (int y = 0; y < a; y++) {
      for (int x = 0; x < b; x++) {
        g[y + r][x + c] = g[y + r][x + c] ^ g[x + r][s - 1 - y + c];
        g[x + r][s - 1 - y + c] = g[y + r][x + c] ^ g[x + r][s - 1 - y + c];
        g[y + r][x + c] = g[y + r][x + c] ^ g[x + r][s - 1 - y + c];

        g[x + r][s - 1 - y + c] = g[x + r][s - 1 - y + c] ^ g[s - 1 - y + r][s - 1 - x + c];
        g[s - 1 - y + r][s - 1 - x + c] = g[x + r][s - 1 - y + c] ^ g[s - 1 - y + r][s - 1 - x + c];
        g[x + r][s - 1 - y + c] = g[x + r][s - 1 - y + c] ^ g[s - 1 - y + r][s - 1 - x + c];

        g[s - 1 - y + r][s - 1 - x + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[s - 1 - x + r][y + c];
        g[s - 1 - x + r][y + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[s - 1 - x + r][y + c];
        g[s - 1 - y + r][s - 1 - x + c] = g[s - 1 - y + r][s - 1 - x + c] ^ g[s - 1 - x + r][y + c];
      }
    }
  }
  void move(int r, int c, int s, char dir) {
    //dump(r, c, s, dir);
    //if (!moves.empty() && moves.back().r == r && moves.back().c == c && moves.back().s == s && moves.back().dir != dir) {
    //  undo();
    //  return;
    //}
    int prev_loc_penalty = loc_penalty;
    loc_penalty -= calc_loc_penalty(r, c, s);
    dir == 'L' ? move_ccw(r, c, s) : move_cw(r, c, s);
    loc_penalty += calc_loc_penalty(r, c, s);
    move_penalty += score_table[s];
    moves.emplace_back(r, c, s, dir, prev_loc_penalty, get_penalty());
  }
  void move(const Move& m) {
    move(m.r, m.c, m.s, m.dir);
  }
  void random_move() {
    int r = rnd.next_int(N - 1), c = rnd.next_int(N - 1);
    int th = min(N - r, N - c);
    int s = rnd.next_int(th - 1) + 2;
    char dir = rnd.next_int(2) ? 'L' : 'R';
    move(r, c, s, dir);
  }
  void bruteforce_move() {
    Move best_move;
    int best_diff = INT_MAX;
    for (int s = 2; s <= N; s++) {
      for (int r = 0; r <= N - s; r++) {
        for (int c = 0; c <= N - s; c++) {
          for (char dir : {'L', 'R'}) {
            int prev_loc_penalty = loc_penalty;
            int prev = get_penalty();
            move(r, c, s, dir);
            int now = get_penalty();
            int diff = now - prev;
            if (diff < best_diff) {
              best_diff = diff;
              best_move = Move(r, c, s, dir, prev_loc_penalty, now);
            }
            undo();
          }
        }
      }
    }
    move(best_move);
  }
  void undo() {
    const Move& m = moves.back();
    m.dir == 'L' ? move_cw(m.r, m.c, m.s) : move_ccw(m.r, m.c, m.s);
    loc_penalty = m.prev_loc_penalty;
    move_penalty -= score_table[m.s];
    moves.pop_back();
  }
  vector<string> get_ans() const {
    vector<string> ans;
    for (const auto& m : moves) ans.push_back(m.toString());
    return ans;
  }
  vector<string> get_best_ans() const {
    int min_idx = -1;
    int min_score = INT_MAX;
    for (int i = 0; i < moves.size(); i++) {
      if (moves[i].now_penalty < min_score) {
        min_idx = i;
        min_score = moves[i].now_penalty;
      }
    }
    vector<string> ans;
    for (int i = 0; i <= min_idx; i++) ans.push_back(moves[i].toString());
    return ans;
  }
  Point get_pos(int n) const {
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        if (g[i][j] == n) return { i, j };
    return { -1,-1 };
  }
  Point get_target(int n) const {
    return { n / N, n % N };
  }
  void print() const {
    fprintf(stderr, "--------------------\n");
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        fprintf(stderr, "%4d ", g[i][j]);
      }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "--------------------\n");
  }

  inline bool isInside(int y, int x) const {
    return 0 <= y && y < N && 0 <= x && x < N;
  }

  void moveTo(const Point& dst, const Point& src, const vector<vector<bool>>& taboo) {
    static int bws[41] = {
      0,
      0,0,0,500,500,500,500,500,500,500,
      200,200,200,200,200,100,100,100,100,100,
      80,70,60,50,40,30,20,15,13,11,
      9,8,7,6,5,5,4,4,3,3
    };
    using PQ = DEPQ<StatePtr>;
    int sdist = abs(src.y - dst.y) + abs(src.x - dst.x);
    StatePtr init_state = make_shared<State>(N, P, g, move_penalty, loc_penalty, src, dst);
    vector<PQ> pqs(sdist + 1);
    pqs[0].push(init_state);
    int width = bws[N];
    //int width = 1;
    for (int d = 0; d < sdist; d++) {
      for(int w = 0; w < width && !pqs[d].empty(); w++) {
        const auto& now_state = pqs[d].get_max();
        now_state->add_next_states(width, pqs, taboo);
        pqs[d].pop_max();
      }
    }
    const StatePtr& best_state = pqs[sdist].get_max();
    for (const Move& m : best_state->moves) {
      move(m);
    }
  }

  void align_normal(int n, vector<vector<bool>>& taboo) {
    int i = n / N, j = n % N;
    Point dst(i, j), src(get_pos(n));
    if (dst == src) {
      taboo[i][j] = true;
      return;
    }
    moveTo(dst, src, taboo);
    taboo[i][j] = true;
  }

  void align_corner(int n1, int n2, vector<vector<bool>>& taboo) {
    {
      int i = n1 / N, j = n1 % N;
      if (j == N - 2) j++;
      else if (i == N - 2) i++;
      Point dst(i, j), src(get_pos(n1));
      moveTo(dst, src, taboo);
    }
    if (get_pos(n2) == Point(n1 / N, n1 % N)) {
      if (n1 / N == N - 2) {
        move(n1 / N - 1, n1 % N, 2, 'L');
        move(n1 / N, n1 % N + 1, 2, 'L');
        move(n1 / N - 1, n1 % N, 2, 'R');
        move(n1 / N, n1 % N, 2, 'R');
      }
      else if (n1 % N == N - 2) {
        move(n1 / N, n1 % N - 1, 2, 'R');
        move(n1 / N + 1, n1 % N, 2, 'R');
        move(n1 / N, n1 % N - 1, 2, 'L');
        move(n1 / N, n1 % N, 2, 'L');
      }
      taboo[n1 / N][n1 % N] = taboo[n2 / N][n2 % N] = true;
      return;
    }
    else {
      taboo[n1 / N][n1 % N] = taboo[n2 / N][n2 % N] = true;
      int i = n2 / N, j = n2 % N;
      if (i == N - 1) j++;
      if (j == N - 1) i++;
      Point dst(i, j), src(get_pos(n2));
      moveTo(dst, src, taboo);
      if (n2 / N == N - 1) {
        move(n2 / N - 1, n2 % N, 2, 'R');
      }
      else if (n2 % N == N - 1) {
        move(n2 / N, n2 % N - 1, 2, 'L');
      }
    }
  }

  void align(vector<int> p) {
    vector<vector<bool>> taboo(N, vector<bool>(N, false));
    for(int n : p) {
      int i = n / N, j = n % N;

      if (i < N - 2 && j < N - 2) {
        align_normal(n, taboo);
        continue;
      }
      else if(i == N - 2) {
        int n2 = n + N;
        align_corner(n, n2, taboo);
      }
      else if (j == N - 2) {
        int n2 = n + 1;
        align_corner(n, n2, taboo);
      }

    }

    //dump(timer.elapsedMs());

    // 最後の 4x4 を A* っぽく解く
    map<int, int> mp;
    vector<int> pp;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        mp[(i + N - 4) * N + (j + N - 4)] = i * 4 + j;
      }
    }
    for (int i = N - 4; i < N; i++) {
      for (int j = N - 4; j < N; j++) {
        pp.push_back(mp[g[i][j]]);
      }
    }

    auto ret = NRP::solve(pp);
    for (int i = 1; i < ret.size(); i++) {
      auto r = ret[i];
      move(r->r + N - 4, r->c + N - 4, r->s, r->dir);
    }
  }
};

class RotatingNumbers {
public:
  vector<string> findSolution(int N, int P, vector<int> grid) {
    init_score_table();

    Solver sol(N, P, grid);

    vector<int> p;

    auto line = [&](int si, int sj, int di, int dj) {
      vector<int> ret;
      if (si == di) {
        // row
        for (int j = sj; j < dj; j++) {
          ret.push_back(si * N + j);
        }
      }
      else {
        // col
        for (int i = si; i < di; i++) {
          ret.push_back(i * N + sj);
        }
      }
      return ret;
    };

    for (int n = 0; n < N - 4; n++) {
      for (int x : line(n, n, n, n + 1)) p.push_back(x);
      for (int x : line(n, n + 1, n, N - 1)) p.push_back(x);
      for (int x : line(n + 1, n, N - 1, n)) p.push_back(x);
    }

    sol.align(p);

    //dump(timer.elapsedMs());

    return sol.get_best_ans();
  }
};

int main() {
  timer.measure();

  RotatingNumbers prog;
  int N;
  int P;
  int num;
  vector<int> grid;

  //ifstream ifs("C:\\dev\\TCMM\\problems\\MM117\\seed1.txt");
  //istream& cin = ifs;

  cin >> N;
  //cerr << N << endl;
  cin >> P;
  //cerr << P << endl;
  for (int i = 0; i < N * N; i++) {
    cin >> num;
    //cerr << num << " ";
    grid.push_back(num);
  }
  //cerr << endl;

  vector<string> ret = prog.findSolution(N, P, grid);
  cout << ret.size() << endl;
  for (int i = 0; i < (int)ret.size(); ++i)
    cout << ret[i] << endl;
  cout.flush();

  return 0;
}



int _main() {
  timer.measure();

  init_score_table();

  int N = NRP::N;
  vector<int> p(N*N);
  rep(i, N*N) p[i] = i;

  rep(i, 20) {
    double elapsed = timer.elapsedMs();
    shuffle_vector(p, rnd);
    auto ans = NRP::solve(p);
    ans.back()->print();
    dump(timer.elapsedMs() - elapsed);
  }

  return 0;
}
