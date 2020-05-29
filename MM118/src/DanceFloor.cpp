//#define NDEBUG
#include "bits/stdc++.h"
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <random>
#ifdef _MSC_VER
#include <ppl.h>
//#include <boost/multiprecision/cpp_dec_float.hpp>
//#include <boost/multiprecision/cpp_int.hpp>
//#include <boost/rational.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#else
//#include <omp.h>
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

template <typename ... Args>
std::string format(const std::string& fmt, Args ... args) {
  size_t len = std::snprintf(nullptr, 0, fmt.c_str(), args ...);
  std::vector<char> buf(len + 1);
  std::snprintf(&buf[0], len + 1, fmt.c_str(), args ...);
  return std::string(&buf[0], &buf[0] + len);
}

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

class FastQueue {
  int front, back;
  int v[1 << 12];
public:
  FastQueue() : front(0), back(0) {}
  inline bool empty() { return front == back; }
  inline void push(int x) { v[front++] = x; }
  inline int pop() { return v[back++]; }
  inline void reset() { front = back = 0; }
  inline int size() { return front - back; }
} fqu;



constexpr int dy[] = { 0, -1, 0, 1 };
constexpr int dx[] = { 1, 0, -1, 0 };
constexpr int dy8[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
constexpr int dx8[] = { 1, 1, 0, -1, -1, -1, 0, 1 };
// 連結成分数差分計算高速化のためのルックアップテーブル
constexpr int lut[] = {
  -1, 0,-1, 0, 0, 1, 0, 0,-1, 0,-1, 0, 0, 1, 0, 0,
   0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0,
  -1, 0,-1, 0, 0, 1, 0, 0,-1, 0,-1, 0, 0, 1, 0, 0,
   0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0,
   0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
   0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
   0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0,
  -1, 0,-1, 0, 0, 1, 0, 0,-1, 0,-1, 0, 0, 1, 0, 0,
   0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0,
  -1, 0,-1, 0, 0, 1, 0, 0,-1, 0,-1, 0, 0, 1, 0, 0,
   0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0,
   0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0,
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
   0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0,
   0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
}; // 0: 連結成分変化なし、-1: 連結成分 1 減少、1: 連結成分要調査

int N, C, D, S;
int tile_colors[52][52][7];
vector<vector<int>> marks;

int debug_count;

void read_input(istream& in) {
  // to 1-indexed
  in >> N >> C >> D >> S;
  for (int y = 1; y <= N; y++) {
    for (int x = 1; x <= N; x++) {
      string cols;
      in >> cols;
      for (int c = 0; c < C; c++) {
        tile_colors[y][x][c] = cols[c] - '0' + 1;
      }
    }
  }
  marks.resize(D);
  for (int i = 0; i < D; i++) {
    int numMarks;
    in >> numMarks;
    marks[i].resize(3 * numMarks);
    for (int j = 0; j < 3 * numMarks; j += 3) {
      in >> marks[i][j] >> marks[i][j + 1] >> marks[i][j + 2];
      marks[i][j]++;
      marks[i][j + 1]++;
    }
  }
}

struct Point {
  int x, y;
  Point() : x(0), y(0) {}
  Point(int x, int y) : x(x), y(y) {}
  int distance(const Point& p) const {
    return abs(x - p.x) + abs(y - p.y);
  }
  bool operator==(const Point& p) const {
    return x == p.x && y == p.y;
  }
  string toString() const {
    return format("(%d,%d)", x, y);
  }
  friend ostream& operator<<(ostream& o, const Point& p) {
    o << p.toString();
    return o;
  }
};

struct Trans {
  // 遷移
  vector<Point> prev; // 変更前のパス
  int ccs[7]; // 変更前の連結成分
  vector<Point> add; // 追加点
  vector<Point> sub; // 削除点
};

struct Path {
  int duration; // 移動時間猶予
  vector<Point> points; // パスの各点
  Path(const Point& from, const Point& to, int duration) : duration(duration) {
    // とりあえず最短で結ぶ
    int x, y, xt, yt;
    x = from.x; y = from.y; xt = to.x; yt = to.y;
    points.emplace_back(x, y);
    while (x != xt) {
      x += (x < xt) ? 1 : -1;
      points.emplace_back(x, y);
    }
    while (y != yt) {
      y += (y < yt) ? 1 : -1;
      points.emplace_back(x, y);
    }
  }
  Trans perturbate() {
    // L 字パスの変形
    static vector<tuple<int, Point, Point>> cands(128);
    cands.clear();
    for (int i = 1; i < (int)points.size() - 1; i++) {
      const Point& p0 = points[i - 1];
      const Point& p1 = points[i];
      const Point& p2 = points[i + 1];
      int d0, d1;
      // 0 -> 1
      if (p0.x != p1.x) {
        d0 = p0.x < p1.x ? 0 : 2;
      }
      else {
        d0 = p0.y < p1.y ? 3 : 1;
      }
      // 1 -> 2
      if (p1.x != p2.x) {
        d1 = p1.x < p2.x ? 0 : 2;
      }
      else {
        d1 = p1.y < p2.y ? 3 : 1;
      }
      if ((d0 ^ d1) & 1) {
        switch ((d0 << 2) | d1) {
        case 3: case 6:
          cands.emplace_back(i, p1, Point(p1.x - 1, p1.y + 1));
          break;
        case 1: case 14:
          cands.emplace_back(i, p1, Point(p1.x - 1, p1.y - 1));
          break;
        case 9: case 12:
          cands.emplace_back(i, p1, Point(p1.x + 1, p1.y - 1));
          break;
        case 4: case 11:
          cands.emplace_back(i, p1, Point(p1.x + 1, p1.y + 1));
          break;
        }
      }
    }
    if (cands.empty()) return Trans();
    Trans trans;
    trans.prev = points;
    auto& selected = cands[rnd.next_int(cands.size())];
    trans.sub.push_back(get<1>(selected));
    trans.add.push_back(get<2>(selected));
    points[get<0>(selected)] = get<2>(selected);
    return trans;
  }
  Trans extend() {
    // パスの伸長: 別の点に行って戻ってくるだけ
    static int dcands[4];
    static int sp;
    if (duration - points.size() + 1 < 2) return Trans();
    Trans trans;
    trans.prev = points;
    int i = rnd.next_int(points.size());
    Point p0(points[i]), p1(points[i]);
    //vector<int> dcands;
    sp = 0;
    for (int d = 0; d < 4; d++) {
      int nx = p0.x + dx[d], ny = p0.y + dy[d];
      if (nx <= 0 || nx > N || ny <= 0 || ny > N) continue;
      dcands[sp++] = d;
    }
    int d = dcands[rnd.next_int(sp)];
    p0.x += dx[d]; p0.y += dy[d];
    points.insert(points.begin() + i + 1, { p0, p1 });
    trans.add.push_back(p0);
    trans.add.push_back(p1);
    return trans;
  }
  inline bool is_inside(int x, int y) const {
    return 1 <= x && x <= N && 1 <= y && y <= N;
  }
  Trans extend2() {
    // パスの伸長: コの字型に変形させる
    static int dcands[2];
    static int sp;
    if (duration - points.size() + 1 < 2) return Trans();
    Trans trans;
    trans.prev = points;
    int i = rnd.next_int(points.size() - 1);
    const Point& p0 = points[i];
    const Point& p3 = points[i + 1];
    int d = p0.y == p3.y ? (p0.x < p3.x ? 0 : 2) : (p0.y < p3.y ? 3 : 1);
    sp = 0;
    int d1 = (d + 1) & 3;
    if (is_inside(p0.x + dx[d1], p0.y + dy[d1])) dcands[sp++] = d1;
    int d2 = (d + 3) & 3;
    if (is_inside(p0.x + dx[d2], p0.y + dy[d2])) dcands[sp++] = d2;
    d = dcands[rnd.next_int(sp)];
    Point p1(p0.x + dx[d], p0.y + dy[d]), p2(p3.x + dx[d], p3.y + dy[d]);
    points.insert(points.begin() + i + 1, { p1, p2 });
    trans.add.push_back(p1);
    trans.add.push_back(p2);
    return trans;
  }
  Trans shrink() {
    // パスの収縮: 行って帰ってくるだけの部分を消す
    static vector<int> cands(128);
    if (points.front().distance(points.back()) == points.size() - 1) return Trans();
    cands.clear();
    for (int i = 2; i < points.size(); i++) {
      if (points[i - 2] == points[i]) {
        cands.push_back(i - 1);
      }
    }
    if (cands.empty()) return Trans();
    Trans trans;
    trans.prev = points;
    int i = cands[rnd.next_int(cands.size())];
    trans.sub.push_back(points[i]);
    trans.sub.push_back(points[i + 1]);
    points.erase(points.begin() + i, points.begin() + i + 2);
    return trans;
  }
  Trans shrink2() {
    // パスの収縮: コの字型を消す
    static vector<int> cands(128);
    // TODO: array
    if (points.front().distance(points.back()) == points.size() - 1) return Trans();
    cands.clear();
    for (int i = 3; i < points.size(); i++) {
      if (points[i - 3].distance(points[i]) == 1) {
        cands.push_back(i - 2);
      }
    }
    if (cands.empty()) return Trans();
    Trans trans;
    trans.prev = points;
    int i = cands[rnd.next_int(cands.size())];
    trans.sub.push_back(points[i]);
    trans.sub.push_back(points[i + 1]);
    points.erase(points.begin() + i, points.begin() + i + 2);
    return trans;
  }
  string toString() const {
    ostringstream oss;
    oss << points;
    return format("max:%d, len:%d, pts:[%s]", duration, (int)points.size() - 1, oss.str().c_str());
  }
  friend ostream& operator<<(ostream& o, const Path& p) {
    o << p.toString();
    return o;
  }
};

struct State {
  int count[52][52];
  int color[52][52];
  // TODO: ull 高速化
  bool palette[52][52]; // for find_contour
  int ccs[7]; // connected components
  int diff; // 隣接マスが異なるものの数
  int not_one; // 1 でない色のセル数
  int sqcsum; // count square sum
  int csum; // count sum
  vector<vector<Path>> paths;
  State() {}
  void init() {
    not_one = sqcsum = csum = 0;
    memset(count, 0, sizeof(int) * 52 * 52);
    memset(color, 0, sizeof(int) * 52 * 52);
    for (int i = 1; i <= N; i++) {
      for (int j = 1; j <= N; j++) {
        color[i][j] = tile_colors[i][j][0];
        not_one += (color[i][j] != 1);
      }
    }
    set_ccs();
    set_diff();
    paths.resize(D);
    for (int d = 0; d < D; d++) {
      for (int i = 3; i < marks[d].size(); i += 3) {
        Point from(marks[d][i - 3], marks[d][i - 2]), to(marks[d][i], marks[d][i + 1]);
        int duration = marks[d][i + 2] - marks[d][i - 1];
        paths[d].emplace_back(from, to, duration);
        add_count(paths[d].back());
      }
    }
  }
  void spread() {
    // 初期状態生成
    for (int d = 0; d < D; d++) {
      for (Path& path : paths[d]) {
        while (true) {
          // 伸ばせるだけ伸ばす
          Trans trans = extend(path);
          if (trans.prev.empty()) break;
        }
      }
    }
    annealing_spread();
  }
  double calc_variance() const {
    return double(sqcsum) / N / N - pow(double(csum) / N / N, 2.0);
  }
  void annealing_spread() {
    double prev_score = calc_variance();
    //dump(prev_score);
    double start = timer.elapsedMs();
    double end = 900;
    double duration = end - start;
    int loop = 0;
    Trans trans;
    while (true) {
      int d = rnd.next_int(D);
      int pid = rnd.next_int(paths[d].size());

      trans = perturbate(paths[d][pid]);
      if (trans.prev.empty()) continue;

      double now_score = calc_variance();

      double diff = now_score - prev_score;
      double temp = get_temp(1e-3, 1e-6, timer.elapsedMs() - start, duration);
      double prob = exp(-diff / temp);

      if (rnd.next_double() < prob) {
        prev_score = now_score;
        //cerr << now_score << endl;
      }
      else {
        undo(paths[d][pid], trans);
      }
      loop++;
      //if (loop % 100000 == 0) {
      //  dump(loop, prev_score, calc_true_score());
      //}
      if (!(loop & 127) && timer.elapsedMs() > end) break;
    }
    //dump(calc_variance());
    //dump(calc_true_score());
    //dump(loop);
  }
  void set_ccs() {
    // 連結成分計算
    static int colors[52][52];
    static ull used[52];
    memset(ccs, 0, sizeof(int) * 7);
    memcpy(colors, color, sizeof(int) * 52 * 52);
    memset(used, 0, sizeof(ull) * 52);
    for (int i = 1; i <= N; i++) {
      for (int j = 1; j <= N; j++) {
        if (used[i] >> j & 1) continue;
        int c = colors[i][j];
        ccs[c]++;
        fqu.reset();
        used[i] |= 1LL << j;
        fqu.push((i << 6) | j);
        while (!fqu.empty()) {
          int p = fqu.pop();
          int ci = p >> 6, cj = p & 0b111111;
          for (int d = 0; d < 4; d++) {
            int ni = ci + dy[d], nj = cj + dx[d];
            if (ni <= 0 || ni > N || nj <= 0 || nj > N || (used[ni] >> nj & 1) || colors[ni][nj] != c) continue;
            used[ni] |= 1LL << nj;
            fqu.push((ni << 6) | nj);
          }
        }
      }
    }
  }
  void set_diff() {
    diff = 0;
    // yoko
    for (int i = 1; i <= N; i++) {
      for (int j = 1; j < N; j++) {
        diff += color[i][j] != color[i][j + 1];
      }
    }
    // tate
    for (int i = 1; i < N; i++) {
      for (int j = 1; j <= N; j++) {
        diff += color[i][j] != color[i + 1][j];
      }
    }
  }
  Trans perturbate(Path& path) {
    Trans trans = path.perturbate();
    memcpy(trans.ccs, ccs, sizeof(int) * 7);
    for (const Point& add : trans.add) add_count(add);
    for (const Point& sub : trans.sub) sub_count(sub);
    return trans;
  }
  Trans extend(Path& path) {
    Trans trans = rnd.next_int(2) ? path.extend2() : path.extend();
    memcpy(trans.ccs, ccs, sizeof(int) * 7);
    for (const Point& add : trans.add) add_count(add);
    for (const Point& sub : trans.sub) sub_count(sub);
    return trans;
  }
  Trans shrink(Path& path) {
    Trans trans = rnd.next_int(2) ? path.shrink2() : path.shrink();
    memcpy(trans.ccs, ccs, sizeof(int) * 7);
    for (const Point& add : trans.add) add_count(add);
    for (const Point& sub : trans.sub) sub_count(sub);
    return trans;
  }
  Trans transition(Path& path) {
    // 各種遷移
    int r = rnd.next_int(3);
    if (r < 1) {
      // L 字型を変形させる
      return perturbate(path);
    }
    else if (r < 2) {
      // パスを伸ばす
      return extend(path);
    }
    else {
      // パスを縮める
      return shrink(path);
    }
  }
  void undo(Path& now, const Trans& trans) {
    for (const Point& sub : trans.sub) add_count_fast(sub);
    for (const Point& add : trans.add) sub_count_fast(add);
    now.points = trans.prev;
    memcpy(ccs, trans.ccs, sizeof(int) * 7);
  }
  double get_temp(double start_temp, double end_temp, double t, double T) {
    return end_temp + (start_temp - end_temp) * (T - t) / T;
  }
  void annealing_single_color() {
    double prev_score = calc_score_single_color();
    double start = timer.elapsedMs();
    double end = 9900;
    double duration = end - start;
    double T = sqrt(N) * 0.1; // 適当
    int loop = 0;
    Trans trans;
    while (true) {
      int d = rnd.next_int(D);
      int pid = rnd.next_int(paths[d].size());

      trans = transition(paths[d][pid]);
      if (trans.prev.empty()) continue;

      double now_score = calc_score_single_color();

      double diff = now_score - prev_score;
      double temp = get_temp(T, 0.01, timer.elapsedMs() - start, duration);
      double prob = exp(-diff / temp);

      if (rnd.next_double() < prob) {
        prev_score = now_score;
        //cerr << now_score << endl;
      }
      else {
        undo(paths[d][pid], trans);
      }
      loop++;
      if (loop % 500000 == 0) {
        dump(loop, prev_score, calc_true_score());
      }
      if (!(loop & 127) && timer.elapsedMs() > end) break;
    }
    dump(calc_true_score());
    dump(loop);
  }
  void annealing() {
    double prev_score = calc_score();
    double start = timer.elapsedMs();
    double end = 14900;
    double duration = end - start;
    double T = sqrt(N) * 0.1; // 適当
    int loop = 0;
    Trans trans;
    while (true) {
      int d = rnd.next_int(D);
      int pid = rnd.next_int(paths[d].size());

      trans = transition(paths[d][pid]);
      if (trans.prev.empty()) continue;

      double now_score = calc_score();

      double diff = now_score - prev_score;
      double temp = get_temp(T, 0.01, timer.elapsedMs() - start, duration);
      double prob = exp(-diff / temp);

      if (rnd.next_double() < prob) {
        prev_score = now_score;
        //cerr << now_score << endl;
      }
      else {
        undo(paths[d][pid], trans);
      }
      loop++;
      if (loop % 500000 == 0) {
        dump(loop, prev_score, calc_true_score());
      }
      if (!(loop & 127) && timer.elapsedMs() > end) break;
    }
    //dump(calc_true_score());
    dump(loop);
  }
  void add_count(const Point& p) {
    // ある点を通ることにする
    int y = p.y, x = p.x;
    int col = color[y][x];
    not_one -= (col != 1);
    for (int d = 0; d < 4; d++) {
      int nc = color[y + dy[d]][x + dx[d]];
      diff -= (nc && nc != col);
    }
    erase_color(x, y);
    int c = ++count[y][x];
    sqcsum += 2 * c - 1;
    csum++;
    int ncol = tile_colors[y][x][c % C];
    not_one += (ncol != 1);
    for (int d = 0; d < 4; d++) {
      int nc = color[y + dy[d]][x + dx[d]];
      diff += (nc && nc != ncol);
    }
    paint_color(x, y, ncol);
  }
  void add_count_fast(const Point& p) {
    // undo 用
    int y = p.y, x = p.x;
    int col = color[y][x];
    not_one -= (col != 1);
    for (int d = 0; d < 4; d++) {
      int nc = color[y + dy[d]][x + dx[d]];
      diff -= (nc && nc != col);
    }
    int c = ++count[y][x];
    sqcsum += 2 * c - 1;
    csum++;
    int ncol = color[y][x] = tile_colors[y][x][c % C];
    not_one += (ncol != 1);
    for (int d = 0; d < 4; d++) {
      int nc = color[y + dy[d]][x + dx[d]];
      diff += (nc && nc != ncol);
    }
  }
  void add_count(const Path& path) {
    const auto& points = path.points;
    for (int i = 1; i < points.size(); i++) {
      add_count(points[i]);
    }
  }
  void sub_count(const Point& p) {
    // ある点を通らないことにする
    int y = p.y, x = p.x;
    int col = color[y][x];
    not_one -= (col != 1);
    for (int d = 0; d < 4; d++) {
      int nc = color[y + dy[d]][x + dx[d]];
      diff -= (nc && nc != col);
    }
    erase_color(x, y);
    int c = --count[y][x];
    sqcsum -= 2 * c + 1;
    csum--;
    int ncol = tile_colors[y][x][c % C];
    not_one += (ncol != 1);
    for (int d = 0; d < 4; d++) {
      int nc = color[y + dy[d]][x + dx[d]];
      diff += (nc && nc != ncol);
    }
    paint_color(x, y, ncol);
  }
  void sub_count_fast(const Point& p) {
    // undo 用
    int y = p.y, x = p.x;
    int col = color[y][x];
    not_one -= (col != 1);
    for (int d = 0; d < 4; d++) {
      int nc = color[y + dy[d]][x + dx[d]];
      diff -= (nc && nc != col);
    }
    int c = --count[y][x];
    sqcsum -= 2 * c + 1;
    csum--;
    int ncol = color[y][x] = tile_colors[y][x][c % C];
    not_one += (ncol != 1);
    for (int d = 0; d < 4; d++) {
      int nc = color[y + dy[d]][x + dx[d]];
      diff += (nc && nc != ncol);
    }
  }
  void sub_count(const Path& path) {
    const auto& points = path.points;
    for (int i = 1; i < points.size(); i++) {
      sub_count(points[i]);
    }
  }
  double calc_score_single_color() const {
    // 連結成分数 + 各色の連結成分数の最大値 + 1ではないグリッドの数
    int score = 0, m = 0;
    for (int i = 1; i < 7; i++) {
      score += ccs[i];
      m = max(m, ccs[i]);
    }
    return score + m + not_one;
  }
  double calc_score() const {
    // 連結成分数 + 各色の連結成分数の最大値 + 色違いの隣接セルのペア数 * 0.1
    int score = 0, mx = 0;
    for (int i = 1; i < 7; i++) {
      score += ccs[i];
      mx = max(mx, ccs[i]);
    }
    return score + mx + diff * 0.1;
  }
  int calc_true_score() {
    // 真のスコア
    int score = 0;
    for (int c : ccs) score += c * c;
    return score;
  }
  void fill_contour(int sx, int sy, int sd) {
    // 輪郭追跡
    // sx, sy, sd: 初期状態
    int s = (sd << 16) | (sx << 8) | sy;
    int x = sx, y = sy, d = sd, c = color[y][x];
    palette[y][x] = 1;
    do {
      d = (d + 1) & 3;
      int ny = y + dy[d], nx = x + dx[d];
      if (color[ny][nx] == c) {
        y = ny; x = nx;
        palette[ny][nx] = 1;
        d = (d + 2) & 3;
      }
    } while (((d << 16) | (x << 8) | y) != s);
  }
  void erase_color(int x, int y) {
    int c = color[y][x];
    // 8 近傍を調べる
    int mask = 0;
    for (int d = 0; d < 8; d++) {
      mask |= (int(c == color[y + dy8[d]][x + dx8[d]]) << d);
    }
    int flag = lut[mask];
    if (flag == -1) {
      // 連結成分 -1
      ccs[c]--;
      color[y][x] = 0;
      return;
    }
    if (flag == 0) {
      // 連結成分変化なし
      color[y][x] = 0;
      return;
    }
    // 4 近傍の連結性を調べないといけない
    // (x, y) の色を削除する
    color[y][x] = 0;
    // 輪郭追跡用配列 4 近傍の初期化
    for (int d = 0; d < 4; d++) {
      // c なら 0 そうでないなら 1
      int ny = y + dy[d], nx = x + dx[d];
      palette[ny][nx] = (c != color[ny][nx]);
    }
    // 輪郭追跡によって 4 近傍の連結性を判定する
    int connection = 0;
    for (int d = 0; d < 4; d++) {
      int ny = y + dy[d], nx = x + dx[d];
      // 1 なら、既に別の場所からの輪郭追跡によって埋められている or 別の色
      if (palette[ny][nx]) continue;
      // 初めて見る連結成分
      connection++;
      // 輪郭追跡
      fill_contour(nx, ny, (d + 2) & 3);
    }
    // 1 つながりの連結成分が connection 個に分裂するので
    ccs[c] += connection - 1;
  }
  void paint_color(int x, int y, int c) {
    // 8 近傍を調べる
    int mask = 0;
    for (int d = 0; d < 8; d++) {
      mask |= (int(c == color[y + dy8[d]][x + dx8[d]]) << d);
    }
    int flag = lut[mask];
    if (flag == -1) {
      // 連結成分 +1
      ccs[c]++;
      color[y][x] = c;
      return;
    }
    if (flag == 0) {
      // 連結成分変化なし
      color[y][x] = c;
      return;
    }
    // 4 近傍の連結性を調べないといけない
    // 輪郭追跡用配列 4 近傍の初期化
    for (int d = 0; d < 4; d++) {
      // c なら 0 そうでないなら 1
      int ny = y + dy[d], nx = x + dx[d];
      palette[ny][nx] = (c != color[ny][nx]);
    }
    // 輪郭追跡によって 4 近傍の連結性を判定する
    int connection = 0;
    for (int d = 0; d < 4; d++) {
      int ny = y + dy[d], nx = x + dx[d];
      // 1 なら、既に別の場所からの輪郭追跡によって埋められている or 別の色
      if (palette[ny][nx]) continue;
      // 初めて見る連結成分
      connection++;
      // 輪郭追跡
      fill_contour(nx, ny, (d + 2) & 3);
    }
    // connection 個の連結成分が 1 つにマージされるので
    ccs[c] -= connection - 1;
    // (x, y) を c で塗る
    color[y][x] = c;
  }
  vector<string> get_ans() const {
    vector<string> dtos(D);
    for (int d = 0; d < D; d++) {
      for (const Path& path : paths[d]) {
        const auto& points = path.points;
        int margin = path.duration - points.size() + 1;
        for (int i = 1; i < points.size(); i++) {
          int dx = points[i].x - points[i - 1].x;
          int dy = points[i].y - points[i - 1].y;
          if (dx < 0) dtos[d].push_back('L');
          else if (dx > 0) dtos[d].push_back('R');
          else if (dy < 0) dtos[d].push_back('U');
          else dtos[d].push_back('D');
        }
        for (int i = 0; i < margin; i++) dtos[d].push_back('-');
      }
    }
    vector<string> ans(S, string(D, ' '));
    for (int d = 0; d < D; d++) {
      for (int s = 0; s < S; s++) {
        ans[s][d] = dtos[d][s];
      }
    }
    return ans;
  }
};

int calc_margin() {
  // M の計算
  int M = 0;
  for (int d = 0; d < D; d++) {
    int sz = marks[d].size();
    for (int i = 3; i < sz - 3; i += 3) {
      int duration = marks[d][i + 2] - marks[d][i - 1];
      int x1, y1, x2, y2;
      x1 = marks[d][i - 3];
      y1 = marks[d][i - 2];
      x2 = marks[d][i];
      y2 = marks[d][i + 1];
      int m = duration - abs(x1 - x2) - abs(y1 - y2);
      M = max(m, M);
    }
  }
  return M;
}

int main() {
  timer.measure();
  ios::sync_with_stdio(false);
  cin.tie(0);

  //ifstream ifs("testcases/41.in");
  //istream& cin = ifs;

  read_input(cin);

  // M を算出
  int M = calc_margin();
  // 被覆率 (D*S)/(N*N*C) を算出
  double coverage = double(D * S) / (N * N * C);
  // パラメータ
  double params[] = { double(N), double(C), double(D), double(S), double(M), coverage, 1.0 };
  // なんちゃって線形分類で求めた重み
  double weight[] = { 0.514006, 0.675051, -0.9004, -0.00899445, -0.0917314, -0.252946, -1 };
  double eval = 0.0;
  for (int i = 0; i < 7; i++) {
    eval += weight[i] * params[i];
  }
  
  // どっちの解法を使うか判定
  bool single_color_mode = (eval < 0.0);

  State state;
  state.init();

  // 初期解はできるだけパスを伸ばし、かつグリッド訪問回数の分散が小さくなるように短時間焼きなまし
  state.spread();

  if (single_color_mode) {
    // できるだけ一つの色に揃えて、1 あるいは 2 点を狙っていく焼きなまし
    state.annealing_single_color();
  }
  else {
    // 問題文と似たスコアで焼きなまし
    // 二乗和より線形和のほうがよくなる傾向にあるが、これはただ単に温度設定が雑すぎるだけかも...
    state.annealing();
  }

  vector<string> ret = state.get_ans();

  cout << ret.size() << endl;
  for (int i = 0; i < ret.size(); i++) {
    cout << ret[i] << endl;
  }
  cout.flush();
  return 0;
}
