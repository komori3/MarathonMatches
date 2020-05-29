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
int N, C, D, S;
vector<vector<string>> tile_colors;
vector<vector<int>> marks;

void read_input(istream& in) {
  in >> N >> C >> D >> S;
  tile_colors.resize(N, vector<string>(N));
  for (int y = 0; y < N; y++) {
    for (int x = 0; x < N; x++) {
      in >> tile_colors[y][x];
    }
  }
  marks.resize(D);
  for (int i = 0; i < D; i++) {
    int numMarks;
    in >> numMarks;
    marks[i].resize(3 * numMarks);
    for (int j = 0; j < 3 * numMarks; j++) {
      in >> marks[i][j];
    }
  }
}

struct Point {
  int x, y;
  Point() : x(0), y(0) {}
  Point(int x, int y) : x(x), y(y) {}
  string toString() const {
    return format("(%d,%d)", x, y);
  }
  friend ostream& operator<<(ostream& o, const Point& p) {
    o << p.toString();
    return o;
  }
};

struct Path {
  int duration;
  vector<Point> points;
  Path(const Point& from, const Point& to, int duration) : duration(duration) {
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
  void rearrange() {
    int x, y, xt, yt;
    x = points.front().x; y = points.front().y; xt = points.back().x; yt = points.back().y;
    points.clear();
    int len = 0;
    points.emplace_back(x, y);
    while (x != xt || y != yt) {
      vector<int> cands;
      for (int d = 0; d < 4; d++) {
        int nx = x + dx[d], ny = y + dy[d];
        if (nx < 0 || nx >= N || ny < 0 || ny >= N) continue;
        int dist = abs(nx - xt) + abs(ny - yt);
        if (len + 1 + dist > duration) continue;
        cands.push_back(d);
      }
      int d = cands[rnd.next_int(cands.size())];
      x += dx[d]; y += dy[d];
      points.emplace_back(x, y);
      len++;
    }
  }
  void perturbate() {
    vector<tuple<int, Point, Point>> cands;
    for (int i = 1; i < (int)points.size() - 1; i++) {
      Point& p0 = points[i - 1];
      Point& p1 = points[i];
      Point& p2 = points[i + 1];
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
        default:
          assert(false);
        }
      }
    }
    if (cands.empty()) {
      //return make_tuple(-1, Point(-1, -1), Point(-1, -1));
      return;
    }
    auto& selected = cands[rnd.next_int(cands.size())];
    points[get<0>(selected)] = get<2>(selected);
    //return selected;
    return;
  }
  void perturbate2() {
    if (duration - points.size() + 1 < 2) return;
    int i = rnd.next_int(points.size());
    Point p0(points[i]), p1(points[i]);
    vector<int> dcands;
    for (int d = 0; d < 4; d++) {
      int nx = p0.x + dx[d], ny = p0.y + dy[d];
      if (nx < 0 || nx >= N || ny < 0 || ny >= N) continue;
      dcands.push_back(d);
    }
    int d = dcands[rnd.next_int(dcands.size())];
    p0.x += dx[d]; p0.y += dy[d];
    points.insert(points.begin() + i + 1, { p0, p1 });
  }
  Point bounding_box() const {
    Point bb(-1, -1);
    for (const Point& p : points) {
      bb.x = max(bb.x, p.x);
      bb.y = max(bb.y, p.y);
    }
    return bb;
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
  vector<vector<int>> count;
  vector<vector<Path>> paths;
  State() {}
  void init() {
    count.resize(N, vector<int>(N, 0));
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
  Path rearrange(Path& path) {
    Path cpath(path);
    sub_count(path);
    path.rearrange();
    add_count(path);
    return cpath;
  }
  Path perturbate(Path& path) {
    Path cpath(path);
    sub_count(path);
    int r = rnd.next_int(100);
    if (r < 80) {
      path.perturbate();
    }
    else if (r < 90) {
      path.perturbate2();
    }
    else {
      path.rearrange();
    }
    add_count(path);
    return cpath;
  }
  void undo(Path& now, const Path& prev) {
    sub_count(now);
    now = prev;
    add_count(now);
  }
  double get_temp(double start_temp, double end_temp, double t, double T) {
    return end_temp + (start_temp - end_temp) * (T - t) / T;
  }
  void annealing() {
    int prev_score = calc_score();
    double start = timer.elapsedMs();
    double end = 9000;
    double duration = end - start;
    int loop = 0;
    while (timer.elapsedMs() < end) {
      int d = rnd.next_int(D);
      int pid = rnd.next_int(paths[d].size());

      Path cpath = perturbate(paths[d][pid]);

      int now_score = calc_score();

      int diff = now_score - prev_score;
      double temp = get_temp(10.0, 0.1, timer.elapsedMs() - start, duration);
      double prob = exp(-diff / temp);

      if (rnd.next_double() < prob) {
        prev_score = now_score;
        //cerr << now_score << endl;
      }
      else {
        undo(paths[d][pid], cpath);
      }
      loop++;
      //if (loop % 10000 == 0) {
      //  dump(loop, prev_score);
      //}
    }
    //dump(calc_score());
    //dump(loop);
  }
  void climbing() {
    int prev_score = calc_score();
    int loop = 0;
    while (timer.elapsedMs() < 9000) {
      int d = rnd.next_int(D);
      int pid = rnd.next_int(paths[d].size());

      Path cpath = rearrange(paths[d][pid]);

      int now_score = calc_score();

      if (now_score > prev_score) {
        undo(paths[d][pid], cpath);
      }
      else {
        prev_score = now_score;
        //cerr << now_score << endl;
      }
      loop++;
    }
    //dump(loop);
  }
  void add_count(const Path& path) {
    const auto& points = path.points;
    for (int i = 1; i < points.size(); i++) {
      count[points[i].y][points[i].x]++;
    }
  }
  void sub_count(const Path& path) {
    const auto& points = path.points;
    for (int i = 1; i < points.size(); i++) {
      count[points[i].y][points[i].x]--;
    }
  }
  int calc_score() const {
    vector<vector<int>> colors(N, vector<int>(N, 0));
    vector<int> ccs(C, 0); // connected components
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        colors[i][j] = tile_colors[i][j][count[i][j] % C] - '0';
      }
    }
    vector<vector<bool>> used(N, vector<bool>(N, false));
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        if (used[i][j]) continue;
        int c = colors[i][j];
        ccs[c]++;
        fqu.reset();
        //queue<pii> qu;
        used[i][j] = true;
        fqu.push((i << 6) | j);
        while (!fqu.empty()) {
          int p = fqu.pop();
          int ci = p >> 6, cj = p & 0b111111;
          for (int d = 0; d < 4; d++) {
            int ni = ci + dy[d], nj = cj + dx[d];
            if (ni < 0 || ni >= N || nj < 0 || nj >= N || used[ni][nj] || colors[ni][nj] != c) continue;
            used[ni][nj] = true;
            fqu.push((ni << 6) | nj);
          }
        }
      }
    }
    int score = 0;
    for (int c : ccs) score += c * c;
    return score;
  }
  vector<string> get_ans() const {
    vector<string> dtos(D);
    for (int d = 0; d < D; d++) {
      for (const Path& path : paths[d]) {
        const auto& points = path.points;
        int margin = path.duration - points.size() + 1;
        assert(margin >= 0);
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

#ifdef _MSC_VER
cv::Mat_<cv::Vec3b> vis_path(const Path& p, int grid_size = 50) {
  Point bb = p.bounding_box();
  int H = (bb.y + 5) * grid_size;
  int W = (bb.x + 5) * grid_size;
  cv::Mat_<cv::Vec3b> img(H, W, cv::Vec3b(255, 255, 255));

  for (int i = 0; i < p.points.size() - 1; i++) {
    const Point& p0 = p.points[i];
    const Point& p1 = p.points[i + 1];
    cv::line(img, cv::Point(p0.x * grid_size, p0.y * grid_size), cv::Point(p1.x * grid_size, p1.y * grid_size), cv::Scalar(0, 0, 0), 2);
  }

  return img;
}
int _main() {
  N = 40;
  Path path(Point(5, 5), Point(14, 14), 28);
  rep(i, 100000) {
    //dump(path);
    if (i % 100 == 0) {
      cv::imshow("img", vis_path(path));
      cv::waitKey(1);
    }
    if (rnd.next_int(10)) path.perturbate();
    else path.perturbate2();
  }
  return 0;
}
#endif

int main() {
  timer.measure();
  ios::sync_with_stdio(false);
  cin.tie(0);

  //ifstream ifs("2.in");
  //istream& cin = ifs;

  read_input(cin);

  State state;
  state.init();

  state.annealing();

  vector<string> ret = state.get_ans();

  cout << ret.size() << endl;
  for (int i = 0; i < ret.size(); i++) {
    cout << ret[i] << endl;
  }
  cout.flush();
  return 0;
}
