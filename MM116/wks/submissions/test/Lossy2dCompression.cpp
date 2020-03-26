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

struct FastMed {
  int hist[26];
  int med, nl, nr, sl, sr, sz;
  FastMed() : med(-1), nl(0), nr(0), sl(0), sr(0), sz(0) {
    memset(hist, 0, sizeof(int) * 26);
  }
  inline int prev() {
    for (int i = med - 1; i >= 0; i--) if (hist[i]) return i;
  }
  inline int next() {
    for (int i = med + 1; i < 26; i++) if (hist[i]) return i;
  }
  inline void add(int x) {
    hist[x]++;
    sz++;
    int h = sz >> 1;
    if (med == -1) med = x;
    else if (x < med) {
      nl++;
      sl += med - x;
      if (h < nl) {
        int pmed = med;
        nr += hist[med];
        med = prev();
        sl -= (pmed - med) * nl;
        sr += (pmed - med) * nr;
        nl -= hist[med];
      }
    }
    else if (x > med) {
      nr++;
      sr += x - med;
      if ((sz >> 1) >= nl + hist[med]) {
        int pmed = med;
        nl += hist[med];
        med = next();
        sl += (med - pmed) * nl;
        sr -= (med - pmed) * nr;
        nr -= hist[med];
      }
    }
  }
  inline void remove_left_shift() {
    int pmed = med;
    nr += hist[med];
    med = prev();
    sl -= (pmed - med) * nl;
    sr += (pmed - med) * nr;
    nl -= hist[med];
  }
  inline void remove_right_shift() {
    int pmed = med;
    nl += hist[med];
    med = next();
    sl += (med - pmed) * nl;
    sr -= (med - pmed) * nr;
    nr -= hist[med];
  }
  inline void remove(int x) {
    sz--;
    hist[x]--;
    if (!sz) {
      med = -1;
      return;
    }
    int h = sz >> 1;
    if (x == med) {
      if (h < nl) remove_left_shift();
      else if (nl + hist[med] <= h) remove_right_shift();
    }
    else if (x < med) {
      nl--;
      sl -= med - x;
      if (nl + hist[med] <= h) remove_right_shift();
    }
    else if (x > med) {
      nr--;
      sr -= x - med;
      if (h < nl) remove_left_shift();
    }
  }
  inline int diff() const { return sl + sr; }
  void verify() {
    vector<int> serial;
    for (int i = 0; i < 26; i++) {
      for (int j = 0; j < hist[i]; j++) {
        serial.push_back(i);
      }
    }
    int n = serial.size();
    assert(serial[n / 2] == med);
    int vsl = 0, vsr = 0;
    for (int x : serial) {
      if (x < med) vsl += med - x;
      if (med < x) vsr += x - med;
    }
    assert(vsl == sl && vsr == sr);
  }
};



struct Rect {
  int id;
  int w, h, x, y;
  Rect() {}
  Rect(int id, int w, int h) : id(id), w(w), h(h), x(-1), y(-1) {}
  Rect(int id, int x, int y, int w, int h) : id(id), x(x), y(y), w(w), h(h) {}
  Rect get_nfp(const Rect& other) const {
    return Rect(-1, x - other.w + 1, y - other.h + 1, w + other.w - 1, h + other.h - 1);
  }
  bool overlap(const Rect& other) const {
    int x0 = x, y0 = y, x1 = x0 + w, y1 = y0 + h;
    int x2 = other.x, y2 = other.y, x3 = x2 + other.w, y3 = y2 + other.h;
    if (x0 > x2) {
      swap(x0, x2); swap(x1, x3);
    }
    bool x_overlap = x1 > x2;
    if (y0 > y2) {
      swap(y0, y2); swap(y1, y3);
    }
    bool y_overlap = y1 > y2;
    return x_overlap && y_overlap;
  }
  string toString() const {
    return "Rect [id=" + to_string(id) + ", w=" + to_string(w) + ", h=" + to_string(h) + ", x=" + to_string(x) + ", y=" + to_string(y) + "]";
  }
};
ostream& operator<<(ostream& o, const Rect& r) {
  o << r.toString();
  return o;
}

// Bottom-Left-Fill
struct BLF {
  int strip_width, strip_height;
  double efficiency;

  vector<Rect> rects;
  vector<Rect> fixed;
  BLF() {}
  BLF(int strip_width, const vector<Rect>& rects) : strip_width(strip_width), rects(rects) {
    int area = 0;
    for (const auto& rect : rects) area += rect.h * rect.w;
    strip_height = area * 3 / strip_width;
  }
  void post_process() {
    strip_height = 0;
    int area = 0;
    for (const auto& rect : fixed) {
      if (!rect.w) continue;
      area += rect.w * rect.h;
      if (rect.y + rect.h > strip_height) {
        strip_height = rect.y + rect.h;
      }
    }
    efficiency = double(area) / (strip_width * strip_height);
  }
  void build() {
    int n = rects.size();
    // fixed[0]: for inner-fit rectangle
    fixed.emplace_back(-1, strip_width, 0, 0, strip_height);

    //vector<vector<int>> overlap(strip_height, vector<int>(strip_width));
    int* overlap_ = new int[(strip_height + 1) * strip_width];
    int** overlap__ = new int* [strip_height + 1];
    for (int i = 0; i <= strip_height; i++) overlap__[i] = &overlap_[i * strip_width];
    int** overlap = &overlap__[1];

    for (const Rect& rect : rects) {
      memset(overlap_, 0, sizeof(int) * (strip_height + 1) * strip_width);
      //vector<vector<int>> overlap(strip_height, vector<int>(strip_width, 0));
      //for (int i = 0; i < strip_height; i++) for (int j = 0; j < strip_width; j++) overlap[i][j] = 0;

      for (const Rect& f : fixed) {
        Rect nfp = f.get_nfp(rect);
        int y0 = max(0, nfp.y);
        int x0 = max(0, nfp.x);
        int y1 = nfp.y + nfp.h;
        int x1 = nfp.x + nfp.w;
        overlap[y0][x0]++;
        if (x1 < strip_width) overlap[y0][x1]--;
        if (y1 < strip_height) {
          overlap[y1][x0]--;
          if (x1 < strip_width) overlap[y1][x1]++;
        }
      }
      // imos2d
      bool flag = false;
      for (int y = 0; y < strip_height; y++) {
        for (int x = 1; x < strip_width; x++) {
          overlap[y][x] += overlap[y][x - 1];
        }
        for (int x = 0; x < strip_width; x++) {
          overlap[y][x] += overlap[y - 1][x];
        }
        for (int x = 0; x < strip_width; x++) {
          if (!overlap[y][x]) {
            fixed.push_back(rect);
            fixed.back().x = x; fixed.back().y = y;
            flag = true;
            break;
          }
        }
        if (flag) break;
      }
    }

    delete[] overlap__;
    delete[] overlap_;

    post_process();
  }
#ifdef _MSC_VER
  cv::Scalar random_color() {
    return cv::Scalar(rnd.next_int(150), rnd.next_int(150), rnd.next_int(150));
  }
  void vis(int delay = 0) {
    cv::Mat_<cv::Vec3b> img(strip_height, strip_width, cv::Vec3b(255, 255, 255));
    for (int i = 1; i <= rects.size(); i++) {
      const auto& r = fixed[i];
      cv::Rect rect(r.x, r.y, r.w, r.h);
      cv::rectangle(img, rect, random_color(), cv::FILLED, cv::INTER_NEAREST);
    }
    int mag = 800 / max(strip_height, strip_width);
    cv::resize(img, img, cv::Size(mag * img.cols, mag * img.rows), 0, 0, cv::INTER_NEAREST);
    cv::imshow("img", img);
    cv::waitKey(delay);
  }
#endif
};

struct Board {
  double P;
  int N;
  int H, W;
  vector<pii> offsets;
  vector<pii> sizes;
  vector<vector<vector<int>>> grids;

  vector<vector<FastMed>> medians;
  int total_grids;
  int total_diff;

  static Board get_random_board(double P, int N, int H, int W, const vector<vector<vector<int>>>& grids) {
    Board board;
    board.P = P;
    board.N = N;
    board.H = H;
    board.W = W;
    board.grids = grids;
    board.total_grids = board.get_total_grids();
    board.medians.resize(H, vector<FastMed>(W));
    board.total_diff = 0;
    for (int i = 0; i < N; i++) {
      const auto& grid = grids[i];
      int h = grid.size(), w = grid[0].size();
      board.sizes.emplace_back(h, w);
      int r = rnd.next_int(H - h + 1);
      int c = rnd.next_int(W - w + 1);
      board.offsets.emplace_back(r, c);
      board.add_layer(i, r, c);
    }
    return board;
  }

  int get_total_grids() const {
    int ret = 0;
    for (const auto& grid : grids) ret += grid.size() * grid[0].size();
    return ret;
  }

  Board() {}
  Board(double P, int N, int H, int W, const vector<vector<vector<int>>>& grids) : P(P), N(N), H(H), W(W), total_grids(0), grids(grids), total_diff(0) {
    total_grids = get_total_grids();
    medians.resize(H, vector<FastMed>(W));
    for (int i = 0; i < N; i++) {
      offsets.emplace_back((int)grids[i].size(), (int)grids[i][0].size());
      add_layer(i, 0, 0);
    }
  }
  Board(double P, int N, const vector<vector<vector<int>>>& grids, BLF& blf) : P(P), N(N), total_grids(0), grids(grids), total_diff(0) {
    H = blf.strip_height;
    W = blf.strip_width;
    total_grids = get_total_grids();
    medians.resize(H, vector<FastMed>(W));
    sort(all(blf.fixed), [](Rect& a, Rect& b) {
      return a.id < b.id;
    });
    for (int i = 1; i < blf.fixed.size(); i++) {
      const auto& rect = blf.fixed[i];
      offsets.emplace_back((int)grids[i - 1].size(), (int)grids[i - 1][0].size());
      add_layer(rect.id, rect.y, rect.x);
    }
  }
  void add_layer(int id, int r, int c) {
    const auto& grid = grids[id];
    int h = grid.size(), w = grid[0].size();
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        int y = i + r;
        int x = j + c;
        total_diff -= medians[y][x].diff();
        medians[y][x].add(grid[i][j]);
        total_diff += medians[y][x].diff();
      }
    }
    offsets[id] = { r, c };
  }
  void remove_layer(int id, int r, int c) {
    const auto& grid = grids[id];
    int h = grid.size(), w = grid[0].size();
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        int y = i + r;
        int x = j + c;
        total_diff -= medians[y][x].diff();
        medians[y][x].remove(grid[i][j]);
        total_diff += medians[y][x].diff();
      }
    }
    offsets[id] = { -1, -1 };
  }
  void move_layer(int id, int nr, int nc) {
    int pr, pc, h, w;
    tie(pr, pc) = offsets[id];
    tie(h, w) = sizes[id];
    // remove
    remove_layer(id, pr, pc);
    // add
    add_layer(id, nr, nc);
    offsets[id] = { nr, nc };
  }
  void adjust_layer(int id) {
    int best_diff = total_diff;
    int best_r, best_c; tie(best_r, best_c) = offsets[id];
    int r_margin = H - grids[id].size();
    int c_margin = W - grids[id][0].size();
    for (int r = 0; r <= r_margin; r++) {
      for (int c = 0; c <= c_margin; c++) {
        move_layer(id, r, c);
        if (total_diff < best_diff) {
          best_diff = total_diff;
          best_r = r; best_c = c;
          //cerr << best_diff << endl;
        }
      }
    }
    move_layer(id, best_r, best_c);
  }
  double get_score() {
    double compression_score = double(H * W) / total_grids;
    double lossiness_score = total_diff / (12.5 * total_grids);
    return compression_score * P + lossiness_score * (1 - P);
  }
  vector<string> get_ans() {
    vector<string> ans(medians.size(), string(medians[0].size(), 0));
    for (int i = 0; i < H; i++) {
      for (int j = 0; j < W; j++) {
        int med = medians[i][j].med;
        ans[i][j] = (med == -1 ? 'A' : (char)med + 'A');
      }
    }
    for (const auto& offset : offsets) {
      ans.push_back(to_string(offset.first) + " " + to_string(offset.second));
    }
    return ans;
  }
};

template<typename T>
void slide(vector<T>& v, int i, int s) {
  int t = i + s;
  int d = s > 0 ? 1 : -1;
  for (int j = i; j != t; j += d) {
    swap(v[j], v[j + d]);
  }
}

double getTemp(double startTemp, double endTemp, double t, double T) {
  return endTemp + (startTemp - endTemp) * (T - t) / T;
}

class Lossy2dCompression {
public:
  vector<string> findSolution(double P, int N, vector< vector<string> > grids) {
    timer.measure();
    //while (timer.elapsed_ms() < 10000) {}

    vector<vector<vector<int>>> igrids;
    for (const auto& grid : grids) {
      int h = grid.size(), w = grid[0].size();
      vector<vector<int>> igrid(h, vector<int>(w));
      for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
          igrid[i][j] = grid[i][j] - 'A';
        }
      }
      igrids.push_back(igrid);
    }

    int H = 0;
    int W = 0;

    int area = 0;
    vector<Rect> rects;
    for (int i = 0; i < igrids.size(); i++) {
      int id = i;
      const auto& grid = igrids[i];
      int h = grid.size(), w = grid[0].size();
      H = max(H, h);
      W = max(W, w);
      area += h * w;
      rects.emplace_back(id, w, h);
    }

    Board best_board(P, N, H, W, igrids);
    double best_score = best_board.get_score();

    // BLF part
    /*if(P < 0.35)*/ {

      sort(all(rects), [](const Rect& a, const Rect& b) {
        return a.h * a.w > b.h* b.w;
      });
      int L = (int)ceil(sqrt(area));

      BLF best_blf;
      double best_efficiency = 0.0;
      int bestL = -1;

      for (int l = W; l <= 2 * L; l++) {
        BLF blf(l, rects);
        blf.build();
        if (best_efficiency < blf.efficiency) {
          best_efficiency = blf.efficiency;
          best_blf = blf;
          bestL = l;
        }
      }
      //dump(best_efficiency, bestL);

      int blf_limit = 1000;
      int loopcnt = 0;

      double start = timer.elapsed_ms();
      double T = blf_limit - start;
      const unsigned RR = (1 << 30), mask = (1 << 30) - 1;
      double prev_efficiency = best_efficiency;
      BLF prev_blf = best_blf;
      while (timer.elapsed_ms() - start < T) {
        int i = rnd.next_int(N), j;
        do {
          j = rnd.next_int(N);
        } while (i == j);
        swap(rects[i], rects[j]);
        BLF blf(bestL, rects);
        blf.build();
        loopcnt++;

        double diff = blf.efficiency - prev_efficiency;
        double temp = getTemp(0.002, 1e-4, timer.elapsed_ms() - start, T);
        double prob = exp(diff / temp);
        if (RR * prob > (rnd.next_int() & mask)) {
          //if(blf.efficiency > prev_efficiency) {
          prev_efficiency = blf.efficiency;
          prev_blf = blf;
          if (prev_efficiency > best_efficiency) {
            best_efficiency = blf.efficiency;
            best_blf = blf;
            //dump(blf.efficiency);
          }
        }
        else swap(rects[i], rects[j]);
      }
      //dump(loopcnt);

      Board board(P, N, igrids, best_blf);
      double score = board.get_score();
      //dump(score);
      if (score < best_score) {
        best_score = score;
        best_board = board;
      }
    }

    // layer moving part
    {
      int num_ext = 20;
      vector<Board> boards(num_ext);
      for (int ext = 0; ext < num_ext; ext++) {
        boards[ext] = Board::get_random_board(P, N, H + ext, W + ext, igrids);
        for (int i = 0; i < N; i++) {
          boards[ext].adjust_layer(i);
        }
        //dump(ext, boards[ext].get_score());
      }
      //dump(timer.elapsed_ms());
      {
        int best_ext = -1;
        double ext_score = DBL_MAX;
        for (int ext = 0; ext < num_ext; ext++) {
          double score = boards[ext].get_score();
          if (score < ext_score) {
            ext_score = score;
            best_ext = ext;
          }
        }
        //dump(best_ext, ext_score);

        int layer_limit = 9800;
        int loopcnt = 0;

        double start = timer.elapsed_ms();
        double T = layer_limit - start;
        const unsigned RR = (1 << 30), mask = (1 << 30) - 1;

        Board board = Board::get_random_board(P, N, H + best_ext, W + best_ext, igrids);
        while (timer.elapsed_ms() < layer_limit) {
          double prev_score = board.get_score();
          int id = rnd.next_int(N);
          int h, w; tie(h, w) = board.sizes[id];
          int pr, pc; tie(pr, pc) = board.offsets[id];
          int nr = rnd.next_int(board.H - h + 1);
          int nc = rnd.next_int(board.W - w + 1);
          //int nr, nc;
          //do {
          //  if (rnd.next_int(2)) {
          //    nr = pr + rnd.next_int(2) ? 1 : -1;
          //    nc = pc;
          //  }
          //  else {
          //    nr = pr;
          //    nc = pc + rnd.next_int(2) ? 1 : -1;
          //  }
          //} while (nr < 0 || board.H - h < nr || nc < 0 || board.W - w < nc);

          board.move_layer(id, nr, nc);
          loopcnt++;

          double now_score = board.get_score();
          double diff = now_score - prev_score;
          double temp = getTemp(0.0002, 1e-6, timer.elapsed_ms() - start, T);
          double prob = exp(-diff / temp);
          if (RR * prob > (rnd.next_int() & mask)) {
            //if(diff < 0) {
            if (now_score < best_score) {
              best_score = now_score;
              best_board = board;
              //dump(best_score);
            }
          }
          else {
            board.move_layer(id, pr, pc);
            loopcnt++;
          }
        }
        //dump(loopcnt);
      }
    }

    return best_board.get_ans();
  }
};



int main() {
  istream& in = cin;
  //ifstream ifs("wks/seed2.txt");
  //istream& in = ifs;
  double P;
  in >> P;
  //cerr << P << endl;
  int N;
  in >> N;
  //cerr << N << endl;
  vector< vector<string> > grids;
  for (int i = 0; i < N; i++) {
    int H;
    in >> H;
    //cerr << H << endl;
    vector<string> grid(H);
    for (int y = 0; y < H; y++) {
      in >> grid[y];
      //cerr << grid[y] << endl;
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
  return 0;
}



struct Med {
  map<int, int> hist;
  int med, nl, nr, sl, sr, sz;
  Med() {
    med = -1;
    nl = nr = sl = sr = sz = 0;
  }
  void add(int x) {
    hist[x]++;
    sz++;
    if (med == -1) {
      med = x;
    }
    else if (x < med) {
      nl++;
      sl += med - x;
      if ((sz >> 1) < nl) {
        // 中央値が左にスライド
        int pmed = med;
        nr += hist[med];
        med = prev(hist.find(med))->first;
        sl -= (pmed - med) * nl;
        sr += (pmed - med) * nr;
        nl -= hist[med];
      }
    }
    else if (x > med) {
      nr++;
      sr += x - med;
      if ((sz >> 1) >= nl + hist[med]) {
        int pmed = med;
        nl += hist[med];
        med = next(hist.find(med))->first;
        sl += (med - pmed) * nl;
        sr -= (med - pmed) * nr;
        nr -= hist[med];
      }
    }
  }
  inline void remove_left_shift() {
    int pmed = med;
    nr += hist[med];
    med = prev(hist.find(med))->first;
    sl -= (pmed - med) * nl;
    sr += (pmed - med) * nr;
    nl -= hist[med];
  }
  inline void remove_right_shift() {
    int pmed = med;
    nl += hist[med];
    med = next(hist.find(med))->first;
    sl += (med - pmed) * nl;
    sr -= (med - pmed) * nr;
    nr -= hist[med];
  }
  void remove(int x) {
    sz--;
    hist[x]--;
    if (!sz) {
      hist.erase(x);
      med = -1;
      return;
    }
    int h = sz >> 1;
    if (x == med) {
      if (h < nl) remove_left_shift();
      else if (nl + hist[med] <= h) remove_right_shift();
    }
    else if (x < med) {
      nl--;
      sl -= med - x;
      if (nl + hist[med] <= h) remove_right_shift();
    }
    else if (x > med) {
      nr--;
      sr -= x - med;
      if (h < nl) remove_left_shift();
    }
    if (!hist[x]) hist.erase(x);
  }
  void verify() {
    vector<int> serial;
    for (const auto& e : hist) {
      for (int i = 0; i < e.second; i++) {
        serial.push_back(e.first);
      }
    }
    int n = serial.size();
    assert(serial[n / 2] == med);
    int vsl = 0, vsr = 0;
    for (int x : serial) {
      if (x < med) vsl += med - x;
      if (med < x) vsr += x - med;
    }
    assert(vsl == sl && vsr == sr);
  }
};

int _main() {

  FastMed med;
  vector<int> serial;
  int N = 1000000;

  srand((unsigned)time(NULL));
  rnd.x = rand();
  REP(i, 100) rnd.next_int();
  for (int i = 0; i < N; i++) serial.push_back(rnd.next_int(26));

  REP(t, 100) {
    if (t % 10 == 0) cerr << t << endl;
    for (int i = 0; i < N; i++) {
      med.add(serial[i]);
      //med.verify();
    }
    shuffle_vector(serial, rnd);
    for (int i = 0; i < N - 1; i++) {
      med.remove(serial[i]);
      //med.verify();
    }
    med.remove(serial.back());
  }

  return 0;
}