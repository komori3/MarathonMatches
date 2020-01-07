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

using uint = unsigned; using ll = long long; using ull = unsigned long long; using pii = pair<int, int>; using pll = pair<ll, ll>; using pdd = pair<double, double>; using pss = pair<string, string>;
template <typename _KTy, typename _Ty> ostream& operator << (ostream& o, const pair<_KTy, _Ty>& m) { o << "[" << m.first << "," << m.second << "]"; return o; }
template <typename _KTy, typename _Ty> ostream& operator << (ostream& o, const map<_KTy, _Ty>& m) { if (m.empty()) { o << "[]"; return o; } o << "[" << *m.begin(); for (auto itr = ++m.begin(); itr != m.end(); itr++) { o << "," << *itr; } o << "]"; return o; }
template <typename _KTy, typename _Ty> ostream& operator << (ostream& o, const unordered_map<_KTy, _Ty>& m) { if (m.empty()) { o << "[]"; return o; } o << "[" << *m.begin(); for (auto itr = ++m.begin(); itr != m.end(); itr++) { o << "," << *itr; } o << "]"; return o; }
template <typename _Ty> ostream& operator << (ostream& o, const vector<_Ty>& v) { if (v.empty()) { o << "[]"; return o; } o << "[" << v.front(); for (auto itr = ++v.begin(); itr != v.end(); itr++) { o << "," << *itr; } o << "]"; return o; }
template <typename _Ty> ostream& operator << (ostream& o, const set<_Ty>& s) { if (s.empty()) { o << "[]"; return o; } o << "[" << *(s.begin()); for (auto itr = ++s.begin(); itr != s.end(); itr++) { o << "," << *itr; } o << "]"; return o; }
template <typename _Ty> ostream& operator << (ostream& o, const unordered_set<_Ty>& s) { if (s.empty()) { o << "[]"; return o; } o << "[" << *(s.begin()); for (auto itr = ++s.begin(); itr != s.end(); itr++) { o << "," << *itr; }	o << "]"; return o; }
template <typename _Ty> ostream& operator << (ostream& o, const stack<_Ty>& s) { if (s.empty()) { o << "[]"; return o; } stack<_Ty> t(s); o << "[" << t.top(); t.pop(); while (!t.empty()) { o << "," << t.top(); t.pop(); } o << "]";	return o; }
template <typename _Ty> ostream& operator << (ostream& o, const list<_Ty>& l) { if (l.empty()) { o << "[]"; return o; } o << "[" << l.front(); for (auto itr = ++l.begin(); itr != l.end(); ++itr) { o << "," << *itr; } o << "]"; return o; }
template <typename _KTy, typename _Ty> istream& operator >> (istream& is, pair<_KTy, _Ty>& m) { is >> m.first >> m.second; return is; }
template <typename _Ty> istream& operator >> (istream& is, vector<_Ty>& v) { for (size_t t = 0; t < v.size(); t++) is >> v[t]; return is; }
namespace aux { // print tuple
  template<typename Ty, unsigned N, unsigned L> struct tp { static void print(ostream& os, const Ty& v) { os << get<N>(v) << ","; tp<Ty, N + 1, L>::print(os, v); } };
  template<typename Ty, unsigned N> struct tp<Ty, N, N> { static void print(ostream& os, const Ty& v) { os << get<N>(v); } };
}

template<typename... Tys> ostream& operator<<(ostream& os, const tuple<Tys...>& t) { os << "["; aux::tp<tuple<Tys...>, 0, sizeof...(Tys) - 1>::print(os, t); os << "]"; return os; }

template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T &val) { fill((T*)array, (T*)(array + N), val); }

void dump_func() { DUMPOUT << endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPOUT << head; if (sizeof...(Tail) == 0) { DUMPOUT << " "; } else { DUMPOUT << ","; } dump_func(move(tail)...); }

#define PI 3.14159265358979323846
#define EPS 1e-8
#define FOR(t,a,n) for(int t=(a);t<(n);++t)
#define REP(t,n)  FOR(t,0,n)
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



struct Order {
  int r;
  int i;
  int e;
  int d;
  int q;
  ll pr;
  int a;

  Order(int _r, int _i, int _e, int _d, int _q, ll _pr, int _a) :
    r(_r), i(_i), e(_e), d(_d), q(_q), pr(_pr), a(_a) {}
};
using OrderPtr = Order * ;

struct Bom {
  struct MainResources {
    bool need_setup;
    map<int, int> m_to_c;
  };

  struct SubResources {
    bool for_setup = false;
    bool for_manufacturing = false;
    vector<int> ms_list;
  };

  int ss;
  int setup_res_count = 0;
  MainResources main_resources;
  vector<SubResources> s_to_sub_resources; // size: 役割数 - 1
};

struct Operation {
  int r;
  int i;
  int t1;
  int t2;
  int t3;
  bool need_setup = false;
  int main_m = 0;
  vector<int> sub_ms_list;
};



constexpr int MAX_TIME = 86400000;
constexpr ll P_MAX = 1e10;

int I;
int M;
int MM;
int MS;
int BL;
int CL;
int R;
vector<Order> orders;
vector<Bom> boms;
//vector<map<pii, int>> setup_times;
vector<vector<vector<int>>> setup_times;

void read_problem(istream& in) {
  string _str;
  in >> _str >> I >> M >> MM >> BL >> CL >> R;
  MS = M - MM;
  boms.resize(I);
  for (int n = 0; n < I; ++n) {
    int i, ss;
    in >> _str >> i >> ss;
    boms[i].ss = --ss;
    boms[i].s_to_sub_resources.resize(ss);
  }
  for (int n = 0; n < BL; ++n) {
    int i, s, m, c, x, y;
    in >> _str >> i >> s >> m >> c >> x >> y;
    Bom &bom = boms[i];
    if (s == 0) {
      bom.main_resources.need_setup = x == 1;
      bom.main_resources.m_to_c[m] = c;
    }
    else {
      Bom::SubResources &subResource = bom.s_to_sub_resources[s - 1];
      subResource.for_setup = x == 1;
      subResource.for_manufacturing = y == 1;
      subResource.ms_list.push_back(m - MM);
    }
  }

  for (Bom &bom : boms) {
    if (bom.main_resources.need_setup) ++bom.setup_res_count;
    for (Bom::SubResources &sr : bom.s_to_sub_resources) {
      sort(all(sr.ms_list));
      if (sr.for_setup) ++bom.setup_res_count;
    }
  }

  setup_times.resize(MM, vector<vector<int>>(I, vector<int>(I, -1)));
  for (int n = 0; n < CL; ++n) {
    int m, i_pre, i_next, t;
    in >> _str >> m >> i_pre >> i_next >> t;
    //setup_times[m][pii(i_pre, i_next)] = t;
    setup_times[m][i_pre][i_next] = t;
  }

  for (int n = 0; n < R; ++n) {
    int r, i, e, d, q, pr, a;
    in >> _str >> r >> i >> e >> d >> q >> pr >> a;
    orders.emplace_back(r, i, e, d, q, pr, a);
  }
  sort(all(orders), [](const Order &o1, const Order &o2) { return o1.r < o2.r; });
}

struct State {
  vector<Operation> operations;
  State() {}

  void solveGreedy(const vector<int>& perm) {
    operations.resize(R);
    for (int idx = 0; idx < R; ++idx) {
      int r = perm[idx];
      Order &order = orders[r];
      Operation &ope = operations[idx];
      Bom &bom = boms[order.i];
      ope.r = r;
      ope.i = order.i;

      ope.main_m = -1; // place holder
      ope.sub_ms_list.resize(bom.ss, -1); // place holder
    }

    vector<int> mToPreviousI(MM, -1);
    vector<int> mainMToPreviousT3(MM, 0);
    vector<int> subMsToPreviousT3(MS, 0);
    for (int n = 0; n < R; ++n) {
      Operation &ope = operations[n];
      int r = ope.r;
      Order &order = orders[r];
      Bom &bom = boms[order.i];

      Bom::MainResources& mr = bom.main_resources;
      int mainM; // choose greedy
      {
        int earliestOrderCompleteTime = INT_MAX;
        for (const auto& e : bom.main_resources.m_to_c) {
          int mm = e.first;
          int previousI = mToPreviousI[mm];
          int setupT = 0;
          if (previousI >= 0) setupT = setup_times[mm][previousI][ope.i];
          int manuT = order.q * e.second;
          int orderCompTime = max(order.e, mr.need_setup ? mainMToPreviousT3[mm] : mainMToPreviousT3[mm] - setupT) + manuT;
          if (orderCompTime < earliestOrderCompleteTime) {
            earliestOrderCompleteTime = orderCompTime;
            mainM = mm;
          }
        }
      }

      ope.main_m = mainM;

      int previousI = mToPreviousI[mainM];
      int setupT = 0;
      if (previousI >= 0) {
        setupT = setup_times[mainM][previousI][ope.i];
      }
      ope.need_setup = setupT > 0;
      int startT = mr.need_setup ? mainMToPreviousT3[mainM] : mainMToPreviousT3[mainM] - setupT;
      int e = max(order.e, startT);
      for (int s = 0; s < bom.ss; ++s) {
        Bom::SubResources &sr = bom.s_to_sub_resources[s];

        int& subMs = ope.sub_ms_list[s]; // choose greedy
        {
          int earliestUnlockTime = INT_MAX;
          for (int sm : bom.s_to_sub_resources[s].ms_list) {
            int unlockTime = sr.for_setup ? subMsToPreviousT3[sm] : subMsToPreviousT3[sm] - setupT;
            if (unlockTime < earliestUnlockTime) {
              earliestUnlockTime = unlockTime;
              subMs = sm;
            }
          }
        }

        if (!ope.need_setup && !sr.for_manufacturing) {
          ope.sub_ms_list[s] = -1;
          continue;
        }
        startT = sr.for_setup ? subMsToPreviousT3[subMs] : subMsToPreviousT3[subMs] - setupT;
        e = max(e, startT);
      }

      int manuT = order.q * bom.main_resources.m_to_c[mainM];
      ope.t1 = e;
      ope.t2 = ope.t1 + setupT;
      ope.t3 = ope.t2 + manuT;

      mToPreviousI[mainM] = ope.i;
      mainMToPreviousT3[mainM] = ope.t3;
      for (int s = 0; s < bom.ss; ++s) {
        int subMs = ope.sub_ms_list[s];
        if (subMs < 0) continue;
        Bom::SubResources &sr = bom.s_to_sub_resources[s];
        int endT = sr.for_manufacturing ? ope.t3 : ope.t2;
        subMsToPreviousT3[subMs] = endT;
      }
    }
  }

  void solveRandom() {
    operations.resize(R);
    for (int r = 0; r < R; ++r) {
      Order &order = orders[r];
      Operation &ope = operations[r];
      Bom &bom = boms[order.i];
      ope.r = r;
      ope.i = order.i;

      ope.main_m = -1; // place holder
      for (int s = 0; s < bom.ss; ++s) {
        ope.sub_ms_list.push_back(-1); // place holder
      }
    }

    sort(all(operations), [](const Operation &ope1, const Operation &ope2) {
      Order &order1 = orders[ope1.r];
      Order &order2 = orders[ope2.r];
      return order1.e < order2.e;
    });

    vector<int> mToPreviousI(MM, -1);
    vector<int> mainMToPreviousT3(MM, 0);
    vector<int> subMsToPreviousT3(MS, 0);
    for (int n = 0; n < R; ++n) {
      Operation &ope = operations[n];
      int r = ope.r;
      Order &order = orders[r];
      Bom &bom = boms[order.i];

      int mainM;
      {
        int nm = bom.main_resources.m_to_c.size();
        auto it = bom.main_resources.m_to_c.begin();
        mainM = std::next(it, rnd.nextUInt(nm))->first; // random choice
        ope.main_m = mainM;
      }

      Bom::MainResources &mr = bom.main_resources;
      int previousI = mToPreviousI[mainM];
      int setupT = 0;
      if (previousI >= 0) {
        setupT = setup_times[mainM][previousI][ope.i];
      }
      ope.need_setup = setupT > 0;
      int startT = mr.need_setup ? mainMToPreviousT3[mainM] : mainMToPreviousT3[mainM] - setupT;
      int e = max(order.e, startT);
      for (int s = 0; s < bom.ss; ++s) {

        int& subMs = ope.sub_ms_list[s];
        {
          int nm = bom.s_to_sub_resources[s].ms_list.size();
          subMs = bom.s_to_sub_resources[s].ms_list[rnd.nextUInt(nm)]; // random choice
        }

        Bom::SubResources &sr = bom.s_to_sub_resources[s];
        if (!ope.need_setup && !sr.for_manufacturing) {
          ope.sub_ms_list[s] = -1;
          continue;
        }
        startT = sr.for_setup ? subMsToPreviousT3[subMs] : subMsToPreviousT3[subMs] - setupT;
        e = max(e, startT);
      }

      int manuT = order.q * bom.main_resources.m_to_c[mainM];
      ope.t1 = e;
      ope.t2 = ope.t1 + setupT;
      ope.t3 = ope.t2 + manuT;

      mToPreviousI[mainM] = ope.i;
      mainMToPreviousT3[mainM] = ope.t3;
      for (int s = 0; s < bom.ss; ++s) {
        int subMs = ope.sub_ms_list[s];
        if (subMs < 0) continue;
        Bom::SubResources &sr = bom.s_to_sub_resources[s];
        int endT = sr.for_manufacturing ? ope.t3 : ope.t2;
        subMsToPreviousT3[subMs] = endT;
      }
    }
  }

  void write_solution(ostream& out) {
    for (Operation &ope : operations) {
      out << ope.r << '\t' << ope.t1 << '\t' << ope.t2 << '\t' << ope.t3 << '\t' << ope.main_m;
      for (int ms : ope.sub_ms_list) {
        int m = ms < 0 ? -1 : ms + MM;
        out << '\t' << m;
      }
      out << endl;
    }
  }

  ll evaluate() {
    ll V1 = 0, V2 = 0;
    for (int n = 0; n < R; ++n) {
      Operation& ope = operations[n];
      Order& order = orders[ope.r];
      Bom& bom = boms[order.i];
      ll setupT = ope.t2 - ope.t1;
      V1 += setupT * bom.setup_res_count;
      if (ope.t3 <= order.d) {
        V2 += order.pr;
      }
      else {
        ll delay = ope.t3 - order.d;
        V2 += order.pr - (order.pr * delay + order.a - 1) / order.a;
      }
    }
    return -V1 + V2;
  }
};

double getTemp(double startTemp, double endTemp, double t, double T, double deg = 1.0) {
  return endTemp + (startTemp - endTemp) * pow((T - t) / T, deg);
}

void slide(vector<int>& v, int i, int d) {
  assert(d && -i <= d && d < (int)v.size() - i);
  if (d > 0) {
    REP(k, d) {
      int pos = i + k;
      swap(v[pos], v[pos + 1]);
    }
  }
  else {
    REP(k, abs(d)) {
      int pos = i - k;
      swap(v[pos], v[pos - 1]);
    }
  }
}

int main() {
  timer.measure();

  cin.tie(0);
  ios::sync_with_stdio(false);

#ifdef _MSC_VER
  ifstream ifs("asprocon5/practice/practice01.txt");
  ofstream ofs("asprocon5/practice/practice_output01.txt");

  istream& in = ifs;
  ostream& out = ofs;
#else
  istream& in = cin;
  ostream& out = cout;
#endif

  read_problem(in);

  vector<OrderPtr> porders;
  for (Order& o : orders) porders.push_back(&o);
  sort(all(porders), [](const OrderPtr& a, const OrderPtr& b) {
    //return a->i == b->i ? a->e < b->e : a->i < b->i;
    return a->e < b->e;
  });
  vector<int> perm;
  for (const OrderPtr& o : porders) perm.push_back(o->r);

  State bestState;
  bestState.solveGreedy(perm);
  ll bestScore = bestState.evaluate();
  dump(bestScore);

  ll prevScore = bestScore;
  double start = timer.elapsedMs();
  double limit = 1985.0;
  const unsigned RR = (1 << 30), mask = (1 << 30) - 1;
  while(timer.elapsedMs() < limit) {
    //int i = rnd.nextUInt(R), j = rnd.nextUInt(R - 1);
    //if (j >= i) j++;
    //if (i > j) swap(i, j);
    //swap(perm[i], perm[j]);
    //reverse(perm.begin() + i, perm.begin() + j + 1);

    int i = rnd.nextUInt(R), d = rnd.nextUInt(-i, R - i - 1);
    if (!d) continue;
    slide(perm, i, d);

    State state;
    state.solveGreedy(perm);
    ll nowScore = state.evaluate();

    double diff = nowScore - prevScore;
    double temp = getTemp(100000, 1, timer.elapsedMs() - start, limit);
    double prob = exp(diff / temp);

    if(RR * prob > (rnd.nextUInt() & mask)) {
    //if (prevScore < nowScore) {
      prevScore = nowScore;
      if (nowScore > bestScore) {
        bestState = state;
        bestScore = nowScore;
        //dump(timer.elapsedMs(), bestScore);
      }
    }
    else {
      //swap(perm[i], perm[j]);
      //reverse(perm.begin() + i, perm.begin() + j + 1);
      slide(perm, i + d, -d);
    }
  }

  dump(bestScore);

  bestState.write_solution(out);

  dump(timer.elapsedMs());

  return 0;
}
