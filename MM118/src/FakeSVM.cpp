#include "bits/stdc++.h"

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

// 焼きなましを使ったなんちゃって線形分類

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

double lrnd[1 << 16];
void init() {
  Xorshift rnd;
  for (int i = 0; i < 100; i++) rnd.nextUInt();
  for (int i = 0; i < (1 << 16); i++) lrnd[i] = log(rnd.nextDouble());
}

vector<string> split(string line) {
  for (char& c : line) if (c == ',') c = ' ';
  istringstream iss(line);
  vector<string> ret;
  string col;
  while (iss >> col) {
    ret.push_back(col);
  }
  return ret;
}

struct Record {
  int seed;
  double N, C, D, S, M, cov, diff;
  Record() {}
  Record(istream& in) {
    in >> seed >> N >> C >> D >> S >> M >> cov >> diff;
  }
};

struct Machine {
  double w[7];
  Machine() {
    w[0] = w[1] = w[2] = w[3] = w[4] = w[5] = w[6] = 0;
  }
  double eval(const Record& r) {
    return w[0] * r.N + w[1] * r.C + w[2] * r.D + w[3] * r.S + w[4] * r.M + w[5] * r.cov + w[6] >= 0 ? r.diff : -r.diff;
  }
  double eval(const vector<Record>& rs) {
    double ret = 0.0;
    for (const auto& r : rs) ret += eval(r);
    return ret;
  }
  void setRandom(Xorshift& rnd) {
    for(int i = 0; i < 6; i++) {
      w[i] = rnd.nextDouble() * 2.0 - 1.0;
    }
    w[6] = rnd.nextDouble() * 10.0 - 5.0;
  }
};

double getTemp(double startTemp, double endTemp, int t, int T) {
  return endTemp + (startTemp - endTemp) * (T - t) / T;
}

double weight[7];

int main() {
  init();

  ifstream ifs("../choose_sol.txt");

  vector<Record> records;

  int num_cases = 1000;
  for(int i = 0; i < num_cases; i++) {
    Record record(ifs);
    records.push_back(record);
  }

  Xorshift rnd;
  Machine best_m, m;
  double best_score = DBL_MIN;

  for (int i = 0; i < 100000; i++) {
    m.setRandom(rnd);
    double score = m.eval(records);
    if (best_score < score) {
      cerr << score << endl;
      dump(vector<double>(m.w, m.w + 7));
      best_score = score;
      best_m = m;
    }
  }

  m = best_m;
  double prevScore = best_score;
  int numLoop = 300000;
  for (int n = 0; n < numLoop; n++) {
    if (n % 100000 == 0) cerr << "n: " << n << endl;
    int idx = rnd.nextUInt(3);
    double pert = (rnd.nextDouble() * 2.0 - 1.0) * (idx == 6 ? 0.05 : 0.01);
    m.w[idx] += pert;
    double nowScore = m.eval(records);

    double diff = nowScore - prevScore;
    double temp = getTemp(3.0, 0.1, n, numLoop);

    if (diff > temp * lrnd[n & 0xFFFF]) {
      prevScore = nowScore;
      if (best_score < nowScore) {
        best_score = nowScore;
        best_m = m;
        cerr << best_score << endl;
        dump(vector<double>(m.w, m.w + 7));
      }
    }
    else {
      m.w[idx] -= pert;
    }
  }

  for (int i = 0; i < 7; i++) {
    best_m.w[i] /= abs(best_m.w[6]);  
    weight[i] = best_m.w[i];
  }
  cout << vector<double>(best_m.w, best_m.w + 7) << endl;
  cout << best_m.eval(records) << endl;

  return 0;
}
