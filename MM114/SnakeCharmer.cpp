#undef NDEBUG
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



constexpr int dy[] = { 0, -1, 0, 1 };
constexpr int dx[] = { 1, 0, -1, 0 };

constexpr int RIGHT = 0;
constexpr int UP = 1;
constexpr int LEFT = 2;
constexpr int DOWN = 3;
constexpr int VACANT = -1;

//struct Section {
//  int id, x, y, v;
//  Section() {}
//  Section(int id, int x, int y, int v) : id(id), x(x), y(y), v(v) {}
//  string toString() const {
//    return "{id:" + to_string(id) + ",x:" + to_string(x) + ",y:" + to_string(y) + ",v:" + to_string(v) + "}";
//  }
//#ifdef _MSC_VER
//  cv::Mat_<cv::Vec3b> toImage(int size, cv::Vec3b col) {
//    cv::Mat_<cv::Vec3b> img(size, size, col);
//    cv::putText(img, to_string(v), cv::Point(size / 4, size * 3 / 4), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
//    return img;
//  }
//#endif
//};
//ostream& operator<<(ostream& o, const Section& s) {
//  o << s.toString();
//  return o;
//}
//using SectionPtr = Section*;

int N, V;
int* snake;

inline int XY(int x, int y) {
  return y * N + x;
}

inline bool isInside(int x, int y) {
  return 0 <= x && x < N && 0 <= y && y < N;
}

int countConnectedComponent(vector<vector<bool>> board) {
  // true のシマを数える
  int ret = 0;
  int H = board.size(), W = board[0].size();
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      if (board[i][j]) {
        ret++;
        queue<pii> qu;
        board[i][j] = false;
        qu.emplace(i, j);
        while (!qu.empty()) {
          int ci, cj; tie(ci, cj) = qu.front(); qu.pop();
          for (int d = 0; d < 4; d++) {
            int ni = ci + dy[d], nj = cj + dx[d];
            if (ni < 0 || ni >= H || nj < 0 || nj >= W || !board[ni][nj]) continue;
            board[ni][nj] = false;
            qu.emplace(ni, nj);
          }
        }
      }
    }
  }
  return ret;
}

template <typename T = int>
T ipow(T x, T n) {
  T ret = 1;
  for (T i = 0; i < n; i++) ret *= x;
  return ret;
}



struct State {
  int len;
  int* xs;
  int* ys;
  int* board;
  State() {
    xs = new int[N * N];
    ys = new int[N * N];
    board = new int[N * N];
    fill(board, board + (N*N), -1);
  }
  State clone() {
    State s;
    s.len = len;
    for (int i = 0; i < N * N; i++) {
      s.xs[i] = xs[i];
      s.ys[i] = ys[i];
      s.board[i] = board[i];
    }
    return s;
  }
  void init() {
    len = 0;
    for (int i = 0; i < N * N; i++) {
      xs[i] = ys[i] = board[i] = VACANT;
    }
    xs[0] = ys[0] = N / 2;
    board[XY(N / 2, N / 2)] = 0;
    len++;
  }
  bool isValid(int x, int y) const {
    if (!isInside(x, y)) return false;
    if (board[XY(x, y)] != VACANT) return false;
    return true;
  }
  bool canMove(int dir) const {
    int x = xs[0], y = ys[0];
    int nx = x + dx[dir], ny = y + dy[dir];
    return isValid(nx, ny);
  }
  bool move(int dir) {
    if (!canMove(dir)) return false;
    int nx = xs[0] + dx[dir], ny = ys[0] + dy[dir];
    for (int i = len; i > 0; i--) {
      xs[i] = xs[i - 1]; 
      ys[i] = ys[i - 1]; 
      board[XY(xs[i], ys[i])] = i;
    }
    len++;
    xs[0] = nx;
    ys[0] = ny;
    board[XY(nx, ny)] = 0;
    return true;
  }
  bool alive() const {
    return len != N * N;
  }
  int countSame(int i) const {
    int y = ys[i], x = xs[i];
    if (board[XY(x, y)] == VACANT) return 0;
    int v = snake[i], c = 1;
    for (int d = 0; d < 4; d++) {
      int nx = x + dx[d], ny = y + dy[d];
      if (!isInside(nx, ny)) continue;
      if (board[XY(nx, ny)] == VACANT) continue;
      int nv = snake[board[XY(nx, ny)]];
      if (v == nv) c++;
    }
    return c;
  }
  int evaluate() const {
    int score = 0;
    for (int i = 0; i < len; i++) {
      score += ipow(snake[i], countSame(i));
    }
    return score;
  }
  bool randomMove() {
    int d = rnd.nextUInt(4);
    return move(d);
  }
  int countWall(int x, int y) {
    int w = 0;
    for (int d = 0; d < 4; d++) {
      int nx = x + dx[d], ny = y + dy[d];
      if (!isValid(nx, ny)) w++;
    }
    return w;
  }
  bool simpleMove() {
    // 最も壁の多いマスに移動
    int x = xs[0], y = ys[0];
    int nd, wall = -1;
    for (int d = 0; d < 4; d++) {
      int nx = x + dx[d], ny = y + dy[d];
      if (!canMove(d)) continue;
      int w = countWall(nx, ny);
      if (wall < w) {
        nd = d;
        wall = w;
      }
    }
    return move(nd);
  }

  bool simpleRandomMove() {
    // 最も壁の多いマスに移動(同率ならランダム)
    int x = xs[0], y = ys[0];
    int maxW = -1;
    vector<int> candD;
    for (int d = 0; d < 4; d++) {
      int nx = x + dx[d], ny = y + dy[d];
      if (!canMove(d)) continue;
      int w = countWall(nx, ny);
      if (maxW < w) {
        maxW = w;
        candD.clear();
        candD.push_back(d);
      }
      else if (maxW == w) {
        candD.push_back(d);
      }
    }
    assert(candD.size());
    int n = candD.size();
    int d = candD[rnd.nextUInt(n)];
    return move(d);
  }

  bool isNoDivisionMove(int dir) {
    assert(canMove(dir));
    int nx = xs[0] + dx[dir], ny = ys[0] + dy[dir];
    vector<vector<bool>> b(N, vector<bool>(N, true));
    for (int i = 0; i < len; i++) b[ys[i]][xs[i]] = false;
    b[ny][nx] = false;
    bool ret = countConnectedComponent(b) <= 1;
    return ret;
  }

  bool destructiveMove(int dir) {
    // 既存ルートを組み替えて強制的に移動する
    int x = xs[0], y = ys[0];
    int nx = x + dx[dir], ny = y + dy[dir];
    assert(isInside(nx, ny) && board[XY(nx, ny)] != VACANT);
    int beginIdx = 0;
    int endIdx = board[XY(nx, ny)];

    for (int s = beginIdx, t = endIdx - 1; s < t; s++, t--) {
      swap(board[XY(xs[s], ys[s])], board[XY(xs[t], ys[t])]);
      swap(xs[s], xs[t]);
      swap(ys[s], ys[t]);
    }
    return true;
  }

  bool simpleRandomMove2() {
    // TODO: 分断回避
    int x = xs[0], y = ys[0];
    vector<int> candD, destD;
    for (int d = 0; d < 4; d++) {
      //if (!canMove(d)) continue;
      //if (!isNoDivisionMove(d)) continue; // 分断するような移動は考慮しない
      int nx = x + dx[d], ny = y + dy[d];
      if (!isInside(ny, nx)) continue;
      if (board[XY(nx, ny)] != VACANT) {
        destD.push_back(d);
        continue;
      }
      int w = countWall(nx, ny);
      if (w == 3) {
        // 3 方包囲マスに必ず移動(それ以外ならランダム)
        return move(d);
      }
      else {
        candD.push_back(d);
      }
    }
    int n = candD.size();
    if (n) {
      // 候補がある
      int d = candD[rnd.nextUInt(n)];
      return move(d);
    }
    else {
      int nd = destD.size();
      int d = destD[rnd.nextUInt(nd)];
      return destructiveMove(d);
    }
    return false;
  }

  vector<char> cvt2ans() {
    // tail から辿る
    vector<char> ans;
    for (int i = len - 1; i > 0; i--) {
      int px = xs[i], py = ys[i];
      int nx = xs[i - 1], ny = ys[i - 1];
      int dx = nx - px, dy = ny - py;
      if (dx) ans.push_back(dx > 0 ? 'R' : 'L');
      else ans.push_back(dy > 0 ? 'D' : 'U');
    }
    return ans;
  }

#ifdef _MSC_VER
  cv::Vec3b getGridColor(int i) {
    static const cv::Vec3b cols[6] = {
      cv::Vec3b(150, 150, 150),
      cv::Vec3b(255, 255, 255),
      cv::Vec3b(255, 255, 0),
      cv::Vec3b(0, 255, 0),
      cv::Vec3b(0, 255, 255),
      cv::Vec3b(0, 0, 255)
    };
    return cols[countSame(i)];
  }
  cv::Mat_<cv::Vec3b> cellImage(int i, int size) {
    cv::Mat_<cv::Vec3b> img(size, size, getGridColor(i));
    cv::putText(img, to_string(snake[i]), cv::Point(size / 4, size * 3 / 4), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    return img;
  }
  cv::Mat_<cv::Vec3b> boardImage(int gridSize = 20) {
    int width, height;
    width = height = N * gridSize;
    cv::Mat_<cv::Vec3b> img(height, width, cv::Vec3b(150, 150, 150));
    for (int i = 0; i < len; i++) {
      int x = xs[i] * gridSize;
      int y = ys[i] * gridSize;
      int h = gridSize;
      int w = gridSize;
      cv::Rect roi(x, y, w, h);
      cv::Mat_<cv::Vec3b> img_roi = img(roi);
      cellImage(board[XY(xs[i], ys[i])], gridSize).copyTo(img_roi);
    }
    for (int i = 0; i < len - 1; i++) {
      cv::Point fp((xs[i] + 0.5) * gridSize, (ys[i] + 0.5) * gridSize);
      cv::Point tp((xs[i + 1] + 0.5) * gridSize, (ys[i + 1] + 0.5) * gridSize);
      cv::line(img, fp, tp, cv::Scalar(0, 0, 0));
    }
    return img;
  }
#endif
};

class SnakeCharmer {
public:
  vector<char> findSolution(int N_, int V_, string snake_) {
    timer.measure();

    N = N_;
    V = V_;
    snake = new int[N * N];
    for (int i = 0; i < N * N; i++) snake[i] = snake_[i] - '0';

    State state;
    state.init();

    int bestScore = state.evaluate();
    State bestState = state.clone();

    // spiral
    int n = N * N - 1;
    for (int i = 0, dir = 0, L = 1; i < n; ) {
      for (int k = 0; k < L && i < n; k++, i++) {
        state.move(dir);
        int score = state.evaluate();
        if (bestScore < score) {
          bestScore = score;
          bestState = state.clone();
          cerr << score << endl;
          //cv::imshow("img", state.boardImage());
          //cv::waitKey(0);
        }
      }
      dir = (dir + 1) % 4;
      for (int k = 0; k < L && i < n; k++, i++) {
        state.move(dir);
        int score = state.evaluate();
        if (bestScore < score) {
          bestScore = score;
          bestState = state.clone();
          cerr << score << endl;
          //cv::imshow("img", state.boardImage());
          //cv::waitKey(0);
        }
      }
      dir = (dir + 1) % 4;
      L++;
    }

    dump(state.evaluate());
    exit(1);

    int numRotate = 0;

    while (timer.elapsedMs() < 9000) {
      state.init();
      while (state.alive()) {
        if (!state.simpleRandomMove2()) break;
        int score = state.evaluate();
        if (bestScore < score) {
          bestScore = score;
          bestState = state.clone();
          cerr << timer.elapsedMs() << ": " << bestState.evaluate() << endl;
        }
        //cv::imshow("img", state.boardImage());
        //cv::waitKey(1);
      }
      //cerr << "rotate" << endl;
      numRotate++;
    }

    dump(numRotate);

    return bestState.cvt2ans();
  }
};

namespace NPlayer {
  constexpr int CLEFT = 0x250000;
  constexpr int CUP = 0x260000;
  constexpr int CRIGHT = 0x270000;
  constexpr int CDOWN = 0x280000;
  const cv::Vec3b cols[6] = {
  cv::Vec3b(150, 150, 150),
  cv::Vec3b(255, 255, 255),
  cv::Vec3b(255, 255, 0),
  cv::Vec3b(0, 255, 0),
  cv::Vec3b(0, 255, 255),
  cv::Vec3b(0, 0, 255)
  };

  const cv::Vec3b cols2[6] = {
  cv::Vec3b(150, 150, 150),
  cv::Vec3b(255, 255, 0),
  cv::Vec3b(0, 255, 0),
  cv::Vec3b(0, 255, 255),
  cv::Vec3b(0, 0, 255),
  cv::Vec3b(0, 0, 255)
  };

  struct Player {
    int N, V;
    vector<int> snake;
    
    vector<int> xs;
    vector<int> ys;
    vector<int> board;

    int bestScore;
    int score;
    stack<int> cornerStack;

    Player(int N, int V, string snake_) : N(N), V(V) {
      for (int i = 0; i < N * N; i++) snake.push_back(snake_[i] - '0');
      board.resize(N * N, VACANT);
      score = 0;
      bestScore = 0;
    }

    inline int XY(int x, int y) const {
      return y * N + x;
    }
    inline bool isInside(int x, int y) const {
      return 0 <= x && x < N && 0 <= y && y < N;
    }
    int countSame(int i) const {
      int y = ys[i], x = xs[i];
      if (board[XY(x, y)] == VACANT) return 0;
      int v = snake[i], c = 1;
      for (int d = 0; d < 4; d++) {
        int nx = x + dx[d], ny = y + dy[d];
        if (!isInside(nx, ny)) continue;
        if (board[XY(nx, ny)] == VACANT) continue;
        int nv = snake[board[XY(nx, ny)]];
        if (v == nv) c++;
      }
      return c;
    }
    cv::Vec3b getGridColor(int i) {
      return cols[countSame(i)];
    }
    cv::Mat_<cv::Vec3b> cellImage(int i, int size) {
      cv::Mat_<cv::Vec3b> img(size, size, getGridColor(i));
      cv::putText(img, to_string(snake[i]), cv::Point(size / 4, size * 3 / 4), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
      return img;
    }
    cv::Mat_<cv::Vec3b> toImage(int gridSize = 20) {
      int width, height;
      width = height = N * gridSize;
      cv::Mat_<cv::Vec3b> img(height, width, cv::Vec3b(150, 150, 150));
      for (int i = 0; i < (int)xs.size(); i++) {
        cv::Rect roi(xs[i] * gridSize, ys[i] * gridSize, gridSize, gridSize);
        cv::Mat_<cv::Vec3b> img_roi = img(roi);
        cellImage(i, gridSize).copyTo(img_roi);
      }
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          if (board[XY(j, i)] != VACANT) continue;
          int iy = i * gridSize;
          int ix = j * gridSize;
          int w = gridSize - 1;
          int h = gridSize - 1;
          cv::rectangle(img, cv::Rect(ix + 5, iy + 5, 5, 5), cols2[countWall(j, i)], 2);
        }
      }
      {
        int x = xs.back(), y = ys.back();
        int ix = x * gridSize, iy = y * gridSize;
        cv::rectangle(img, cv::Rect(ix, iy, gridSize, gridSize), cv::Scalar(0, 0, 0), 1);
        x = N / 2; y = N / 2;
        ix = x * gridSize; iy = y * gridSize;
        cv::rectangle(img, cv::Rect(ix, iy, gridSize, gridSize), cv::Scalar(0, 0, 0), 2);
      }
      for (int i = 0; i < (int)xs.size() - 1; i++) {
        cv::Point fp((xs[i] + 0.5) * gridSize, (ys[i] + 0.5) * gridSize);
        cv::Point tp((xs[i + 1] + 0.5) * gridSize, (ys[i + 1] + 0.5) * gridSize);
        cv::line(img, fp, tp, cv::Scalar(0, 0, 0));
      }
      return img;
    }

    void undo() {
      int x = xs.back(), y = ys.back();
      score -= calcCellScore(x, y);
      for (int d = 0; d < 4; d++) score -= calcCellScore(x + dx[d], y + dy[d]);
      board[XY(xs.back(), ys.back())] = VACANT;
      xs.pop_back();
      ys.pop_back();
      score += calcCellScore(x, y);
      for (int d = 0; d < 4; d++) score += calcCellScore(x + dx[d], y + dy[d]);
    }

    int calcCellScore(int x, int y) {
      if (!isInside(x, y) || board[XY(x, y)] == VACANT) return 0;
      int v = snake[board[XY(x, y)]], c = 1;
      for (int d = 0; d < 4; d++) {
        int nx = x + dx[d], ny = y + dy[d];
        if (!isInside(nx, ny) || board[XY(nx, ny)] == VACANT) continue;
        int nv = snake[board[XY(nx, ny)]];
        if (v == nv) c++;
      }
      return ipow(v, c);
    }

    void update(int x, int y) {
      int v = snake[xs.size()];
      score -= calcCellScore(x, y);
      for (int d = 0; d < 4; d++) score -= calcCellScore(x + dx[d], y + dy[d]);
      board[XY(x, y)] = xs.size();
      xs.push_back(x);
      ys.push_back(y);
      score += calcCellScore(x, y);
      for (int d = 0; d < 4; d++) score += calcCellScore(x + dx[d], y + dy[d]);
    }

    void move(int dir) {
      int nx = xs.back() + dx[dir], ny = ys.back() + dy[dir];
      // undo
      if(xs.size() >= 2) {
        int i = xs.size() - 2;
        if (xs[i] == nx && ys[i] == ny) {
          undo();
          return;
        }
      }

      if (!isInside(nx, ny)) return;
      if(board[XY(nx, ny)] != VACANT) return;

      update(nx, ny);
    }

    int countWall(int x, int y) {
      int w = 0;
      for (int d = 0; d < 4; d++) {
        int nx = x + dx[d], ny = y + dy[d];
        if (!isInside(nx, ny) || board[XY(nx, ny)] != VACANT) w++;
      }
      return w;
    }

    bool isValidMove(int dir) {
      int nx = xs.back() + dx[dir], ny = ys.back() + dy[dir];
      if (!isInside(nx, ny) || board[XY(nx, ny)] != VACANT) return false;
      if (nx == N / 2 && ny == N / 2 && xs.size() != N * N - 1) return false;
      if (xs.size() >= 2) {
        int i = xs.size() - 2;
        if (xs[i] == nx && ys[i] == ny) {
          return false;
        }
      }
      
    }

    bool recmove(int dir) {
      int nx = xs.back() + dx[dir], ny = ys.back() + dy[dir];
      if (!isInside(nx, ny)) return false;
      if (board[XY(nx, ny)] != VACANT) return false;
      if (nx == N / 2 && ny == N / 2 && xs.size() != N * N - 1) return false;
      // undo
      if (xs.size() >= 2) {
        int i = xs.size() - 2;
        if (xs[i] == nx && ys[i] == ny) {
          return false;
        }
      }
      // corner 2 つ以上
      update(nx, ny);
      int num3 = 0, corner = -1;
      for (int d = 0; d < 4; d++) {
        int nnx = nx + dx[d], nny = ny + dy[d];
        if (!isInside(nnx, nny) || board[XY(nnx, nny)] != VACANT) continue;
        if (countWall(nnx, nny) == 3) {
          num3++;
          corner = d;
        }
      }
      if (num3 >= 2) {
        undo();
        return false;
      }

      vector<vector<bool>> space(N, vector<bool>(N, true));
      for (int i = 0; i < xs.size(); i++) {
        space[ys[i]][xs[i]] = false;
      }
      if (countConnectedComponent(space) > 1) {
        undo();
        return false;
      }
      
      cornerStack.push(corner);
      return true;
    }

    void rec(int filled) {
      if (filled == 10) {
        if (bestScore < score) {
          bestScore = score;
          cerr << score << endl;
          //cv::imshow("img", toImage(15));
          //cv::waitKeyEx(1);
        }
        return;
      }
      vector<int> ds({ 0, 1, 2, 3 });
      shuffleVector(ds, rnd);
      for (int d : ds) {
        int corner = cornerStack.top();
        if (corner != -1 && corner != d) continue;
        bool success = recmove(d);
        if (!success) continue;
        rec(filled + 1);
        undo();
        cornerStack.pop();
      }
    }

    void solve(int sx, int sy) {
      update(sx, sy);
      cornerStack.push(-1);

      rec(1);
    }

    void play(int sx, int sy) {
      // (sx, sy) からスタート
      update(sx, sy);

      while (1) {
        cv::imshow("img", toImage());
        int key = cv::waitKeyEx(15);
        int dir = -1;
        switch (key) {
        case CRIGHT:
          dir = 0;
          break;
        case CUP:
          dir = 1;
          break;
        case CLEFT:
          dir = 2;
          break;
        case CDOWN:
          dir = 3;
          break;
        case 27:
          return;
        default:
          break;
        }
        if (dir != -1) {
          move(dir);
          dump(score);
        }
      }
    }
  };


}

int main() {
  ifstream in("case1.txt");
  //istream& in = cin;
  SnakeCharmer prog;
  int N;
  int V;
  string snake;
  in >> N;
  in >> V;
  in >> snake;

  NPlayer::Player player(N, V, snake);
  player.play(0, 0);
  //player.solve(0, 0);

  return 0;
}

int _main() {
  ifstream in("case1.txt");
  //istream& in = cin;
  SnakeCharmer prog;
  int N;
  int V;
  string snake;
  in >> N;
  in >> V;
  in >> snake;

  vector<char> ret = prog.findSolution(N, V, snake);
  cout << ret.size() << endl;
  for (int i = 0; i < (int)ret.size(); ++i)
    cout << ret[i] << endl;
  cout.flush();

  return 0;
}

//int main() {
//  int N;
//  int V;
//  string snake;
//  cin >> N;
//  cerr << N << endl;
//  cin >> V;
//  cerr << V << endl;
//  cin >> snake;
//  cerr << snake << endl;
//  cout.flush();
//}