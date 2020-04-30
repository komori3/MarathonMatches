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



int score_table[41];
void init_score_table() {
  for (int s = 2; s <= 40; s++) {
    score_table[s] = int(floor(pow(s - 1, 1.5)));
    //dump(s, score_table[s]);
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

  vector<State> get_all_next_states(const vector<vector<bool>>& taboo) const {
    vector<State> states;
    int dist = abs(dst.y - now.y) + abs(dst.x - now.x);
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
            State nstate(*this);
            nstate.move(oy, ox, s, 'L');
            nstate.now = Point(ly, lx);
            states.push_back(nstate);
          }
          int ry = oy + dx, rx = ox + s - dy - 1;
          ndist = abs(dst.y - ry) + abs(dst.x - rx);
          if (ndist == dist - (s - 1)) {
            State nstate(*this);
            nstate.move(oy, ox, s, 'R');
            nstate.now = Point(ry, rx);
            states.push_back(nstate);
          }
        }
      }
    }
    return states;
  }
};

bool operator<(const State& a, const State& b) {
  return a.get_penalty() > b.get_penalty();
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
          for (char dir : "LR") {
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
  bool moveTo(const Point& dst, Point& src, const vector<vector<bool>>& taboo) {
    if (dst == src) return false;
    // offset
    static constexpr int oys[] = { -1,-1,-1,-1,0,0,0,0 };
    static constexpr int oxs[] = { -1,-1,0,0,-1,-1,0,0 };
    // dirs
    static constexpr int dirs[] = { 'L', 'R', 'L', 'R', 'L', 'R', 'L', 'R' };
    // moves
    static constexpr int dys[] = { -1,0,0,-1,0,1,1,0 };
    static constexpr int dxs[] = { 0,-1,1,0,-1,0,0,1 };

    // [sy - 1, sy + 1) * [sx - 1, sx + 1) に対して
    // サイズ 2 回転を(行えるなら)行い、dst に近づくような移動を採用する
    int moy = -1, mox = -1;
    char mdir = -1;
    int mdist = abs(src.y - dst.y) + abs(src.x - dst.x);
    for (int i = 0; i < 8; i++) {
      int oy = src.y + oys[i];
      int ox = src.x + oxs[i];
      if (!isInside(oy, ox) || !isInside(oy + 1, ox + 1)) continue;
      if (taboo[oy][ox] || taboo[oy][ox + 1] || taboo[oy + 1][ox] || taboo[oy + 1][ox + 1]) continue;
      int ny = src.y + dys[i];
      int nx = src.x + dxs[i];
      int dist = abs(ny - dst.y) + abs(nx - dst.x);
      if (dist < mdist) {
        move(oy, ox, 2, dirs[i]);
        src.y += dys[i];
        src.x += dxs[i];
        return true;
      }
    }
    return false;
  }
  void moveTo2(const Point& dst, const Point& src, const vector<vector<bool>>& taboo) {
    using PQ = priority_queue<State>;
    int dist = abs(src.y - dst.y) + abs(src.x - dst.x);
    State init_state(N, P, g, move_penalty, loc_penalty, src, dst);
    vector<PQ> pqs(dist + 1);
    pqs[0].push(init_state);
    int width = 1;
    for (int d = 0; d < dist; d++) {
      for(int w = 0; w < width && !pqs[d].empty(); w++) {
      //while (!pqs[d].empty()) {
        const State& now_state = pqs[d].top();
        const auto& next_states = now_state.get_all_next_states(taboo);
        for (const State& next_state : next_states) {
          int ndist = abs(next_state.now.y - dst.y) + abs(next_state.now.x - dst.x);
          pqs[dist - ndist].push(next_state);
        }
        pqs[d].pop();
      }
    }
    State best_state = pqs[dist].top();
    for (const Move& m : best_state.moves) {
      move(m);
    }
  }

  void align(vector<int> p) {
    vector<vector<bool>> taboo(N, vector<bool>(N, false));
    for(int n : p) {
      //dump(n);
      int i = n / N, j = n % N;
      if (i >= N - 2 && j >= N - 2) break;
      Point dst(n / N, n % N);
      Point src = get_pos(n);
      if (dst == src) {
        taboo[i][j] = true;
        continue;
      }
      if (dst.x == N - 1) dst.y++;
      if (dst.y == N - 1) dst.x++;
      //while (moveTo(dst, src, taboo)) {}
      moveTo2(dst, src, taboo);
      if(dst.x == N - 1) dst.y--;
      if (dst.y == N - 1) dst.x--;
      src = get_pos(n);
      if (dst != src && dst.y < N - 2 && dst.x == N - 1) {
        assert(dst.y + 1 == src.y);
        assert(dst.x == src.x);
        move(src.y, src.x - 1, 2, 'R');
        move(dst.y, dst.x - 1, 2, 'R');
        move(src.y, src.x - 1, 2, 'L');
        move(dst.y, dst.x - 1, 2, 'L');
      }
      if (dst != src && dst.x < N - 2 && dst.y == N - 1) {
        assert(dst.x + 1 == src.x);
        assert(dst.y == src.y);
        move(src.y - 1, src.x, 2, 'L');
        move(dst.y - 1, dst.x, 2, 'L');
        move(src.y - 1, src.x, 2, 'R');
        move(dst.y - 1, dst.x, 2, 'R');
      }
      taboo[n / N][n % N] = true;
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

    for (int n = 0; n < N; n++) {
      for (int x : line(n, n, n, n + 1)) p.push_back(x);
      for (int x : line(n, n + 1, n, N)) p.push_back(x);
      for (int x : line(n + 1, n, N, n)) p.push_back(x);
    }

    sol.align(p);

    //while (timer.elapsedMs() < 9000 && sol.moves.size() < 100000) {
    //  sol.bruteforce_move();
    //}

    //dump(sol.moves.size());

    return sol.get_ans();
  }
};

int main() {
  timer.measure();

  RotatingNumbers prog;
  int N;
  int P;
  int num;
  vector<int> grid;

  //ifstream ifs("C:\\dev\\TCMM\\problems\\MM117\\test.txt");
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
  int N = 10;
  vector<int> g1(N * N);
  rep(i, N * N) g1[i] = i;
  shuffle_vector(g1, rnd);

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

  vector<int> p;
  for (int n = 0; n < N; n++) {
    for (int x : line(n, n, n, n + 1)) p.push_back(x);
    for (int x : line(n, n + 1, n, N)) p.push_back(x);
    for (int x : line(n + 1, n, N, n)) p.push_back(x);
  }

  sort(all(p));

  auto find = [&](int n) {
    for (int i = 0; i < N*N; i++) {
      if (g1[i] == n) return pii(i / N, i % N);
    }
  };

  auto distance = [](const pii& a, const pii& b) {
    return abs(a.first - b.first) + abs(a.second - b.second);
  };

  auto calc = [&](vector<int>& g, int N) {
    int cost = 0;
    for (int i : p) {
      pii src = find(i);
      pii dst(i / N, i % N);
      cost += distance(src, dst);
      swap(g[src.first * N + src.second], g[dst.first * N + dst.second]);
    }
    return cost;
  };

  int numloop = 1000000;
  double sum = 0.0;
  rep(i, numloop) {
    sum += calc(g1, N);
    shuffle_vector(g1, rnd);
  }
  
  dump(sum / numloop);

  return 0;
}