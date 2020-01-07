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



struct PrimeOrder {
  int r;
  int i;
  int e;
  int d;
  int q;
  ll pr;
  int a;

  PrimeOrder(int _r, int _i, int _e, int _d, int _q, ll _pr, int _a) :
    r(_r), i(_i), e(_e), d(_d), q(_q), pr(_pr), a(_a) {}
};
using PrimeOrderPtr = PrimeOrder * ;

struct PrimeBom {
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
int BL;
int CL;
int R;
vector<PrimeOrder> orders;
vector<PrimeBom> boms;
vector<vector<vector<int>>> setup_times;

void read_problem(istream& in) {
  string _str;
  in >> _str >> I >> M >> MM >> BL >> CL >> R;
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
    PrimeBom &bom = boms[i];
    if (s == 0) {
      bom.main_resources.need_setup = x == 1;
      bom.main_resources.m_to_c[m] = c;
    }
    else {
      PrimeBom::SubResources &subResource = bom.s_to_sub_resources[s - 1];
      subResource.for_setup = x == 1;
      subResource.for_manufacturing = y == 1;
      subResource.ms_list.push_back(m);
    }
  }

  for (PrimeBom &bom : boms) {
    if (bom.main_resources.need_setup) ++bom.setup_res_count;
    for (PrimeBom::SubResources &sr : bom.s_to_sub_resources) {
      sort(all(sr.ms_list));
      if (sr.for_setup) ++bom.setup_res_count;
    }
  }

  setup_times.resize(MM, vector<vector<int>>(I, vector<int>(I, -1)));
  for (int n = 0; n < CL; ++n) {
    int m, i_pre, i_next, t;
    in >> _str >> m >> i_pre >> i_next >> t;
    setup_times[m][i_pre][i_next] = t;
  }

  for (int n = 0; n < R; ++n) {
    int r, i, e, d, q, pr, a;
    in >> _str >> r >> i >> e >> d >> q >> pr >> a;
    orders.emplace_back(r, i, e, d, q, pr, a);
  }
  sort(all(orders), [](const PrimeOrder &o1, const PrimeOrder &o2) { return o1.r < o2.r; });
}

struct Process {
  int m, r, i, t1, t2, t3;
  Process* prev;
  Process* next;
};
using ProcessPtr = Process*;

struct State {
  vector<Operation> operations;
  State() {}

  void solveBlockAssign(const vector<vector<PrimeOrderPtr>>& pBlockOrders) {
    //operations.resize(R);

    vector<int> mToPreviousI(MM, -1);
    vector<int> mToPreviousT3(M, 0);
    for (int b = 0; b < pBlockOrders.size(); ++b) {
      // for each block
      const vector<PrimeOrderPtr>& block = pBlockOrders[b];
      int item = block.front()->i;

      const PrimeBom& bom = boms[item];
      const PrimeBom::MainResources& mr = bom.main_resources;
      int mainM; // choose greedy
      {
        int shortestProcessingTime = INT_MAX;
        for (const auto& entry : bom.main_resources.m_to_c) {
          int mm = entry.first;
          int previousI = mToPreviousI[mm];
          int setupT = 0;
          if (previousI >= 0) setupT = setup_times[mm][previousI][item];
          // 最初のオーダ処理開始時刻
          int start = max(block.front()->e, mr.need_setup ? mToPreviousT3[mm] : mToPreviousT3[mm] - setupT);
          // 最初のオーダ処理終了時刻
          int end = start + block.front()->q * entry.second;
          // 順にオーダを見ていくと最後のオーダを(副資源が潤沢にあるという仮定で)処理し終わる時刻がわかる
          for (int i = 1; i < block.size(); i++) {
            end = max(block[i]->e, end + block[i]->q * entry.second);
          }
          // end - start が最も短い主資源を選択する
          int processingTime = end - start;
          if (processingTime < shortestProcessingTime) {
            shortestProcessingTime = processingTime;
            mainM = mm;
          }
        }
      }

      int previousI = mToPreviousI[mainM];
      int setupT = 0;
      if (previousI >= 0) setupT = setup_times[mainM][previousI][item];
      bool need_setup = setupT > 0;
      int startT = mr.need_setup ? mToPreviousT3[mainM] : mToPreviousT3[mainM] - setupT;
      int e = max(block.front()->e, startT);
      // 使用する副資源選択
      vector<int> sub_ms_list(bom.ss, -1);
      for (int s = 0; s < bom.ss; ++s) {
        // 役割ごとに最も解放時刻の早い副資源を選択する
        const PrimeBom::SubResources& sr = bom.s_to_sub_resources[s];
        int& subM = sub_ms_list[s]; // choose greedy
        {
          int earliestUnlockTime = INT_MAX;
          for (int sm : bom.s_to_sub_resources[s].ms_list) {
            int unlockTime = sr.for_setup ? mToPreviousT3[sm] : mToPreviousT3[sm] - setupT;
            if (unlockTime < earliestUnlockTime) {
              earliestUnlockTime = unlockTime;
              subM = sm;
            }
          }
        }
        if (!need_setup && !sr.for_manufacturing) {
          sub_ms_list[s] = -1;
          continue;
        }
        startT = sr.for_setup ? mToPreviousT3[subM] : mToPreviousT3[subM] - setupT;
        e = max(e, startT);
      }
      // 製造能力
      int ct = bom.main_resources.m_to_c.at(mainM);
      // まとめて割り付け
      for (int i = 0; i < block.size(); i++) {
        e = max(e, block[i]->e);
        const PrimeOrderPtr& porder = block[i];
        int manuT = porder->q * ct;
        Operation ope;
        ope.r = porder->r;
        ope.i = item;
        ope.main_m = mainM;
        ope.sub_ms_list = sub_ms_list;
        ope.need_setup = !i ? need_setup : false;
        ope.t1 = e;
        ope.t2 = ope.t1 + (ope.need_setup ? setupT : 0); // 2 回目以降は段取り 0
        ope.t3 = ope.t2 + manuT;
        mToPreviousT3[mainM] = ope.t3;
        for (int s = 0; s < bom.ss; ++s) {
          int& subM = ope.sub_ms_list[s];
          if (subM < 0) continue;
          const PrimeBom::SubResources& sr = bom.s_to_sub_resources[s];
          if (!ope.need_setup && !sr.for_manufacturing) {
            // subM = -1;  // vs だと通る linux だと死ぬ　参照で未定義踏んでる？
            ope.sub_ms_list[s] = -1;
            continue;
          }
          int endT = sr.for_manufacturing ? ope.t3 : ope.t2;
          mToPreviousT3[subM] = endT;
        }
        operations.push_back(ope);
        e = ope.t3;
      }
      mToPreviousI[mainM] = item;
    }
  }

  void solveGreedy(const vector<PrimeOrderPtr>& porders) {
    operations.resize(R);

    vector<int> mToPreviousI(MM, -1);
    vector<int> mToPreviousT3(M, 0);
    for (int n = 0; n < R; ++n) {
      Operation &ope = operations[n];

      int r = porders[n]->r;
      const PrimeOrder& order = orders[r];
      const PrimeBom& bom = boms[order.i];
      ope.r = r;
      ope.i = order.i;
      ope.main_m = -1; // place_holder
      ope.sub_ms_list.resize(bom.ss, -1); // place holder

      const PrimeBom::MainResources& mr = bom.main_resources;
      int mainM; // choose greedy
      {
        int earliestOrderCompleteTime = INT_MAX;
        for (const auto& e : bom.main_resources.m_to_c) {
          int mm = e.first;
          int previousI = mToPreviousI[mm];
          int setupT = 0;
          if (previousI >= 0) setupT = setup_times[mm][previousI][ope.i];
          int manuT = order.q * e.second;
          int orderCompTime = max(order.e, mr.need_setup ? mToPreviousT3[mm] : mToPreviousT3[mm] - setupT) + manuT;
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
      int startT = mr.need_setup ? mToPreviousT3[mainM] : mToPreviousT3[mainM] - setupT;
      int e = max(order.e, startT);
      for (int s = 0; s < bom.ss; ++s) {
        const PrimeBom::SubResources &sr = bom.s_to_sub_resources[s];

        int& subMs = ope.sub_ms_list[s]; // choose greedy
        {
          int earliestUnlockTime = INT_MAX;
          for (int sm : bom.s_to_sub_resources[s].ms_list) {
            int unlockTime = sr.for_setup ? mToPreviousT3[sm] : mToPreviousT3[sm] - setupT;
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
        startT = sr.for_setup ? mToPreviousT3[subMs] : mToPreviousT3[subMs] - setupT;
        e = max(e, startT);
      }

      int manuT = order.q * bom.main_resources.m_to_c.at(mainM);
      ope.t1 = e;
      ope.t2 = ope.t1 + setupT;
      ope.t3 = ope.t2 + manuT;

      mToPreviousI[mainM] = ope.i;
      mToPreviousT3[mainM] = ope.t3;
      for (int s = 0; s < bom.ss; ++s) {
        int subMs = ope.sub_ms_list[s];
        if (subMs < 0) continue;
        const PrimeBom::SubResources &sr = bom.s_to_sub_resources[s];
        int endT = sr.for_manufacturing ? ope.t3 : ope.t2;
        mToPreviousT3[subMs] = endT;
      }
    }
  }

  void write_solution(ostream& out) {
    for (const Operation &ope : operations) {
      out << ope.r << '\t' << ope.t1 << '\t' << ope.t2 << '\t' << ope.t3 << '\t' << ope.main_m;
      for (int ms : ope.sub_ms_list) {
        int m = ms < 0 ? -1 : ms;
        out << '\t' << m;
      }
      out << endl;
    }
  }

  ll evaluate() const {
    ll V1 = 0, V2 = 0;
    for (int n = 0; n < R; ++n) {
      const Operation& ope = operations[n];
      const PrimeOrder& order = orders[ope.r];
      const PrimeBom& bom = boms[order.i];
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

template<typename T>
void slide(vector<T>& v, int i, int d) {
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

template<typename T>
vector<T> flatten(const vector<vector<T>>& vv) {
  vector<T> ret;
  for (const auto& v : vv) ret.insert(ret.end(), v.begin(), v.end());
  return ret;
}

int main() {
  timer.measure();

  cin.tie(0);
  ios::sync_with_stdio(false);

#ifdef _MSC_VER
  ifstream ifs("workdir/testcase/17/1.txt");
  ofstream ofs("tools/practice/hoge.txt");

  istream& in = ifs;
  ostream& out = ofs;
#else
  istream& in = cin;
  ostream& out = cout;
#endif

  read_problem(in);

  vector<vector<PrimeOrderPtr>> orderBlocks(I);
  // 品目単位でまとめる
  for (PrimeOrder& o : orders) orderBlocks[o.i].push_back(&o);
  // 品目単位で最早開始時刻ソート
  for (auto& v : orderBlocks) {
    sort(all(v), [](const PrimeOrderPtr& a, const PrimeOrderPtr& b) {
      return a->e < b->e;
    });
  }

  vector<PrimeOrderPtr> porders = flatten(orderBlocks);

  State bestState;
  bestState.solveBlockAssign(orderBlocks);
  ll bestScore = bestState.evaluate();
  dump(bestScore);

  //double limit = 1985.0;

  while (timer.elapsedMs() < 200) {
    int i = rnd.nextUInt(I), j = rnd.nextUInt(I - 1);
    if (j >= i) j++;

    swap(orderBlocks[i], orderBlocks[j]);

    State state;
    state.solveBlockAssign(orderBlocks);

    ll nowScore = state.evaluate();

    if (bestScore < nowScore) {
      bestScore = nowScore;
      bestState = state;
      dump(timer.elapsedMs(), bestScore);
    }
    else {
      swap(orderBlocks[i], orderBlocks[j]);
    }
  }

  //ll prevScore = bestScore;
  //double start = timer.elapsedMs();
  //int loopcnt = 0;
  //const unsigned RR = (1 << 30), mask = (1 << 30) - 1;
  //while(timer.elapsedMs() < limit) {
  //  int i = rnd.nextUInt(R), j = rnd.nextUInt(R - 1);
  //  if (j >= i) j++;
  //  if (i > j) swap(i, j);
  //  swap(porders[i], porders[j]);
  //  reverse(porders.begin() + i, porders.begin() + j + 1);

  //  //int i = rnd.nextUInt(R), d = rnd.nextUInt(-i, R - i - 1);
  //  //if (!d) continue;
  //  //slide(porders, i, d);

  //  State state;
  //  state.solveGreedy(porders);
  //  ll nowScore = state.evaluate();

  //  double diff = nowScore - prevScore;
  //  double temp = getTemp(100000, 1, timer.elapsedMs() - start, limit);
  //  double prob = exp(diff / temp);

  //  if(RR * prob > (rnd.nextUInt() & mask)) {
  //  //if (prevScore < nowScore) {
  //    prevScore = nowScore;
  //    if (nowScore > bestScore) {
  //      bestState = state;
  //      bestScore = nowScore;
  //      dump(timer.elapsedMs(), bestScore);
  //    }
  //  }
  //  else {
  //    swap(porders[i], porders[j]);
  //    //reverse(porders.begin() + i, porders.begin() + j + 1);
  //    //slide(porders, i + d, -d);
  //  }
  //  loopcnt++;
  //}
  //dump(loopcnt);

  dump(bestScore);

  bestState.write_solution(out);

  dump(timer.elapsedMs());

  return 0;
}
