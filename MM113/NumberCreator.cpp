#define NDEBUG
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

template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T &val) { std::fill((T*)array, (T*)(array + N), val); }

void dump_func() { DUMPOUT << endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPOUT << head; if (sizeof...(Tail) == 0) { DUMPOUT << " "; } else { DUMPOUT << ", "; } dump_func(std::move(tail)...); }

#define PI 3.14159265358979323846
#define EPS 1e-8
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



struct BigInt {
  static constexpr int N = 512;
  using Bit = bitset<N>;
private:
  size_t msb; // 1-indexed

  string sdiv2(const string& s) { string ret; int cur = 0; bool lz = false; for (int ptr = 0; ptr < s.size(); ptr++) { cur = cur * 10 + s[ptr] - '0'; if (cur >> 1)lz = true; if (lz)ret.push_back((cur >> 1) + '0'); cur &= 1; }return ret == "" ? "0" : ret; }
  Bit to_bit(string s) { Bit bit; int ptr = 0; while (s[0] != '0') { int t = s.back() - '0'; if (t & 1)bit[ptr] = true; ptr++; s = sdiv2(s); }return bit; }
  vector<string> split(string s, char delim) const {
    for (char& c : s) if (c == delim) c = ' ';
    istringstream iss(s);
    vector<string> ret;
    string buf;
    while (iss >> buf) {
      ret.push_back(buf);
    }
    return ret;
  }
public:
  Bit bit;
  BigInt() : msb(0) {}
  inline size_t size() const {
    return msb;
  }
  BigInt(uint64_t n) : bit(n) {
    msb = 0;
    for (int i = 63; i >= 0; i--) if ((n >> i) & 1) {
      msb = i + 1;
      return;
    }
  }
  BigInt(const string& s) : bit(to_bit(s)) {
    msb = 0;
    for (int i = N - 1; i >= 0; i--) if (bit[i]) {
      msb = i + 1;
      return;
    }
  }
  inline bool operator!() const {
    return !msb;
  }
  inline BigInt& operator<<=(size_t n) {
    msb += n;
    bit <<= n;
    return *this;
  }
  inline BigInt operator<<(size_t n) const {
    return BigInt(*this) <<= n;
  }
  inline BigInt& operator>>=(size_t n) {
    msb = max(0, (int)msb - (int)n);
    bit >>= n;
    return *this;
  }
  inline BigInt operator>>(size_t n) const {
    return BigInt(*this) >>= n;
  }
  inline bool operator==(const BigInt& b) const {
    size_t msb = std::max(this->size(), b.size());
    for (size_t i = 0; i < msb; i++) if (this->bit[i] ^ b.bit[i]) return false;
    return true;
  }
  inline bool operator!=(const BigInt& b) const {
    return !(*this == b);
  }
  inline bool operator<(const BigInt& b) const {
    if (*this == b) return false;
    size_t msb1 = this->size(), msb2 = b.size();
    if (msb1 == msb2) {
      for (int i = (int)msb1 - 1; i >= 0; i--) {
        if (this->bit[i] ^ b.bit[i]) return b.bit[i];
      }
    }
    return msb1 < msb2;
  }
  inline bool operator<=(const BigInt& b) const {
    return *this == b || *this < b;
  }
  inline bool operator>(const BigInt& b) const {
    return !(*this <= b);
  }
  inline bool operator>=(const BigInt& b) const {
    return !(*this < b);
  }
  inline BigInt& operator+=(const BigInt& b) {
    Bit& b1 = this->bit;
    const Bit& b2 = b.bit;
    size_t msb = max(this->msb, b.msb);
    if (!msb) return *this;
    bool c = false;
    for (int i = 0; i < msb; i++) {
      bool s = b1[i] ^ b2[i] ^ c;
      c = (b1[i] & b2[i]) | (c & (b1[i] | b2[i]));
      b1[i] = s;
    }
    this->msb = msb;
    if (c) this->msb++, b1[msb] = true;
    return *this;
  }
  inline BigInt operator+(const BigInt& b) const {
    return BigInt(*this) += b;
  }
  inline BigInt& operator-=(const BigInt& b) {
    assert(*this >= b);
    Bit& b1 = this->bit;
    const Bit& b2 = b.bit;
    size_t msb = max(this->msb, b.msb);
    if (!msb) return *this;
    bool c = false;
    for (int i = 0; i < msb; i++) {
      bool s = b1[i] ^ b2[i] ^ c;
      c = (!b1[i] & (c | b2[i])) | (c & b2[i]);
      b1[i] = s;
    }
    this->msb = msb;
    if (!b1[msb - 1]) this->msb--;
    return *this;
  }
  inline BigInt operator-(const BigInt& b) const {
    return BigInt(*this) -= b;
  }
  inline BigInt& operator*=(const BigInt& b) {
    BigInt ret;
    if (!(*this) || !b) return *this = ret;
    int msb = this->msb;
    Bit& b1 = this->bit;
    for (int i = 0; i < msb; i++) if (b1[i]) {
      ret += b << i;
    }
    return *this = ret;
  }
  inline BigInt operator*(const BigInt& b) const {
    return BigInt(*this) *= b;
  }
  inline BigInt& operator/=(const BigInt& b) {
    assert(b.msb);
    if (*this < b) return *this = BigInt();
    BigInt a(*this), c(b), q(0), sq(1), ori_c(c);
    while (a >= c) {
      c <<= 1;
      if (a >= c) sq <<= 1;
      else c >>= 1, a -= c, q += sq, sq = BigInt(1), c = ori_c;
    }
    return *this = q;
  }
  inline BigInt operator/(const BigInt& b) const {
    return BigInt(*this) /= b;
  }
  string to_string() const {
    string sbit(bit.to_string());
    int i;
    for (i = 0; i < sbit.size(); i++) if (sbit[i] == '1') break;
    return i == sbit.size() ? "0" : string(sbit.begin() + i, sbit.end());
  }
  uint64_t to_ullong() const {
    return bit.to_ullong();
  }
  vector<int> getRLE() const {
    string sbit = bit.to_string();
    vector<string> ones = split(sbit, '0');
    vector<string> zeros = split(sbit, '1');
    zeros.erase(zeros.begin());
    vector<int> rle;
    for (int i = 0; i < ones.size() - 1; i++) {
      rle.push_back((int)ones[i].size());
      rle.push_back((int)zeros[i].size());
    }
    rle.push_back((int)ones[ones.size() - 1].size());
    if (ones.size() == zeros.size()) rle.push_back((int)zeros[zeros.size() - 1].size());
    return rle;
  }
};
ostream& operator<<(ostream& o, const BigInt& b) {
  o << b.to_string();
  return o;
}



namespace NBinarySolver {

  BigInt target;
  vector<int> rle; // run length encoding

  struct Cmd {
  private:
    int pack;
  public:
    Cmd() {}
    Cmd(int id1, char op, int id2) : pack((id1 << 20) + (int(op) << 10) + id2) {}
    inline int getId1() const {
      return pack >> 20;
    }
    inline char getOp() const {
      return (pack >> 10) & 0b1111111111;
    }
    inline int getId2() const {
      return pack & 0b1111111111;
    }
    string toString() const {
      return to_string(getId1()) + " " + getOp() + " " + to_string(getId2());
    }
  };
  ostream& operator<<(ostream& o, const Cmd& c) {
    o << c.toString();
    return o;
  }

  struct State {
    vector<BigInt> nums;
    vector<Cmd> cmds;
    unordered_map<int, int> e2;
    unordered_map<int, int> r1;

    int ptrId, ptrRLE;

    State(int num0, int num1) {
      nums.emplace_back(to_string(num0));
      nums.emplace_back(to_string(num1));
      ptrId = ptrRLE = 0;
    }
    bool isCompleted() const {
      return ptrRLE == rle.size();
    }
    int create1() {
      operate(0, '/', 0);
      e2[0] = nums.size() - 1;
      r1[1] = nums.size() - 1;
      return nums.size() - 1;
    }
    int getOrCreateExp2(int e) {
      if (e2.count(e)) return e2[e];
      if (e & 1) {
        if (!e2.count(e - 1)) {
          operate_mul(getOrCreateExp2(e >> 1), getOrCreateExp2(e >> 1), e >> 1);
          e2[e - 1] = nums.size() - 1;
        }
        operate_mul(getOrCreateExp2(e - 1), getOrCreateExp2(1), 1);
        return e2[e] = nums.size() - 1;
      }
      operate_mul(getOrCreateExp2(e >> 1), getOrCreateExp2(e >> 1), e >> 1);
      return e2[e] = nums.size() - 1;
    }
    int getOrCreateRep1(int rep) {
      if (r1.count(rep)) return r1[rep];
      int e = getOrCreateExp2(rep);
      operate(e, '-', e2[0]);
      return r1[rep] = nums.size() - 1;
    }
    void preProcess() {
      int id1 = create1();
      operate(id1, '+', id1);
      e2[1] = nums.size() - 1;
      int flen = rle[0];
      ptrId = getOrCreateRep1(flen);
      ptrRLE = 1;
    }
    void postProcess() {
      if (!(rle.size() & 1)) {
        int len = rle[rle.size() - 1];
        operate_mul(nums.size() - 1, getOrCreateExp2(len), len);
      }
      ptrRLE++;
    }
    void operate_mul(int id1, int id2, int shift) {
      nums.push_back(nums[id1] << shift);
      cmds.emplace_back(id1, '*', id2);
    }
    void operate(int id1, char op, int id2) {
      switch (op) {
      case '+': nums.push_back(nums[id1] + nums[id2]); break;
      case '-': nums.push_back(nums[id1] - nums[id2]); break;
      case '*': nums.push_back(nums[id1] * nums[id2]); break;
      case '/': nums.push_back(nums[id1] / nums[id2]); break;
      }
      cmds.emplace_back(id1, op, id2);
    }
    int eval() const {
      return cmds.size() * 1000 - e2.size() - r1.size();
    }
    vector<string> getAns() const {
      vector<string> ans;
      for (const auto& c : cmds) ans.push_back(c.toString());
      return ans;
    }
    shared_ptr<State> getNextState() const {
      assert(!isCompleted());
      shared_ptr<State> ns(new State(*this));
      if (!ns->ptrRLE) {
        ns->preProcess();
        return ns;
      }
      if (ns->ptrRLE == rle.size() - 1) {
        ns->postProcess();
        return ns;
      }

      int len0 = rle[ns->ptrRLE];
      int len1 = rle[ns->ptrRLE + 1];
      int len01 = len0 + len1;

      int id1, id2;
      id1 = ns->ptrId; id2 = ns->getOrCreateExp2(len01);
      ns->operate_mul(id1, id2, len01);
      id1 = ns->nums.size() - 1; id2 = ns->getOrCreateRep1(len1);
      ns->operate(id1, '+', id2);
      ns->ptrId = ns->nums.size() - 1;
      ns->ptrRLE += 2;
      return ns;
    }

    bool canGetSub0State(int depth) const {
      int end = ptrRLE + 2 * (depth + 1);
      if (end >= rle.size()) return false;
      for (int i = 1; i <= depth; i++) {
        int p = ptrRLE + 2 * i;
        if (rle[p] != 1) return false;
      }
      return true;
    }
    shared_ptr<State> getSub0State(int depth) const {
      shared_ptr<State> s(new State(*this));
      int end = ptrRLE + 2 * (depth + 1);
      int e = 0;
      vector<int> subtractExp2s;
      for (int i = end - 1; i >= ptrRLE; i--) {
        e += rle[i];
        if (i > ptrRLE && (i & 1)) subtractExp2s.push_back(e - 1);
      }
      int r = e - rle[ptrRLE];

      int id1, id2;

      id1 = s->ptrId; id2 = s->getOrCreateExp2(e);
      s->operate_mul(id1, id2, e);
      s->ptrId = s->nums.size() - 1;

      id1 = s->ptrId; id2 = s->getOrCreateRep1(r);
      s->operate(id1, '+', id2);
      s->ptrId = s->nums.size() - 1;

      for (int sube : subtractExp2s) {
        id1 = s->ptrId; id2 = s->getOrCreateExp2(sube);
        s->operate(id1, '-', id2);
        s->ptrId = s->nums.size() - 1;
      }

      s->ptrRLE = end;
      return s;
    }

    bool canGetAdd1State(int depth) const {
      int end = ptrRLE + 2 * (depth + 1);
      if (end >= rle.size()) return false;
      for (int i = 1; i <= depth; i++) {
        int p = ptrRLE + 2 * i - 1;
        if (rle[p] != 1) return false;
      }
      return true;
    }
    shared_ptr<State> getAdd1State(int depth) const {
      shared_ptr<State> s(new State(*this));
      int end = ptrRLE + 2 * (depth + 1);
      int e = 0;
      vector<int> addExp2s;
      for (int i = end - 1; i >= ptrRLE; i--) {
        e += rle[i];
        int j = end - i;
        if (!(j & 1) && i != ptrRLE) addExp2s.push_back(e);
      }
      int r = rle[end - 1];

      int id1, id2;
      id1 = s->ptrId; id2 = s->getOrCreateExp2(e);
      s->operate_mul(id1, id2, e);
      s->ptrId = s->nums.size() - 1;

      id1 = s->ptrId; id2 = s->getOrCreateRep1(r);
      s->operate(id1, '+', id2);
      s->ptrId = s->nums.size() - 1;

      for (int adde : addExp2s) {
        id1 = s->ptrId; id2 = s->getOrCreateExp2(adde);
        s->operate(id1, '+', id2);
        s->ptrId = s->nums.size() - 1;
      }

      s->ptrRLE = end;
      return s;
    }


    vector<shared_ptr<State>> getAllNextStates() {
      assert(!isCompleted());

      vector<shared_ptr<State>> states;

      if (ptrRLE == 0) {
        shared_ptr<State> ns(new State(*this));
        ns->preProcess();
        states.push_back(ns);
        return states;
      }

      if (ptrRLE == rle.size() - 1) {
        shared_ptr<State> ns(new State(*this));
        ns->postProcess();
        states.push_back(ns);
        return states;
      }

      // default
      states.push_back(getNextState());

      for (int depth = 1; canGetSub0State(depth); depth++) states.push_back(getSub0State(depth));

      for (int depth = 1; canGetAdd1State(depth); depth++) states.push_back(getAdd1State(depth));

      return states;
    }
  };
  using StatePtr = shared_ptr<State>;
  bool operator==(const StatePtr& s1, const StatePtr& s2) {
    return s1->eval() == s2->eval();
  }
  bool operator<(const StatePtr& s1, const StatePtr& s2) {
    return s1->eval() < s2->eval();
  }
  bool operator>(const StatePtr& s1, const StatePtr& s2) {
    return s1->eval() > s2->eval();
  }
}

namespace NDecimalSolver {
  struct TestCase {
    const int num0;
    const int num1;
    const string target;
    TestCase(int num0, int num1, string target) : num0(num0), num1(num1), target(target) {}
  };
  using TestCasePtr = shared_ptr<TestCase>;

  struct ScoreTable {
    vector<string> smallTargetStringList;
    vector<int> smallTargetList;
    unordered_set<int> smallTargetSet;

    ScoreTable(const TestCasePtr& tc) {
      string target = tc->target;
      for (int p = 0; p < target.size(); p += 4) {
        int end = min(p + 4, (int)target.size());
        string sub = target.substr(p, end - p);
        smallTargetStringList.push_back(sub);
        int isub = stoi(sub);
        smallTargetList.push_back(isub);
        if (isub) smallTargetSet.insert(isub);
      }
    }
  };
  using ScoreTablePtr = shared_ptr<ScoreTable>;

  struct State {
    static constexpr int THRESH = 20000;

    ScoreTablePtr table;
    int match;
    unordered_map<int, int> numMap;
    vector<int> cmds;

    State(const TestCasePtr& tc, const ScoreTablePtr& table) : table(table), match(0) {
      push(tc->num0);
      push(tc->num1);
    }

    bool isCompleted() {
      return match == table->smallTargetSet.size();
    }

    int push(int num) {
      assert(num);
      int id = numMap.size();
      numMap[id] = num;
      if (table->smallTargetSet.count(num)) match++;
      return id;
    }

    inline int in_range(int c) {
      return 0 <= c && c < THRESH;
    }

    inline int eval_in(int c, bitset<THRESH>& target) {
      int e = 0;
      if (in_range(c) && target[c]) e += 100;
      if (in_range(c >> 1) && target[c >> 1]) e++;
      if (in_range(c << 1) && target[c << 1]) e++;
      if (in_range(c - 1) && target[c - 1]) e++;
      if (in_range(c + 1) && target[c + 1]) e++;
      if (in_range(c >> 2) && target[c >> 2]) e++;
      if (in_range(c << 2) && target[c << 2]) e++;
      if (in_range(c - 2) && target[c - 2]) e++;
      if (in_range(c + 2) && target[c + 2]) e++;
      if (in_range(c - 4) && target[c - 4]) e++;
      if (in_range(c + 4) && target[c + 4]) e++;
      return e;
    }

    int eval() {
      static bitset<THRESH> visited;
      static bitset<THRESH> target;
      visited.reset();
      target.reset();

      int e = 0;
      vector<int> numList(numMap.size());
      {
        int i = 0;
        for (const auto& e : numMap) {
          assert(e.second);
          numList[i++] = e.second;
        }
      }
      for (int num : table->smallTargetSet) target[num] = true;
      for (int num : numList) target[num] = false;

      for (int id0 = 0; id0 < numList.size(); id0++) {
        int n0 = numList[id0];
        for (int id1 = id0; id1 < numList.size(); id1++) {
          int n1 = numList[id1];

          int c;
          c = n0 + n1;
          if (c < THRESH && !visited[c]) {
            visited[c] = true;
            e += eval_in(c, target);
          }
          c = abs(n0 - n1);
          if (!visited[c]) {
            visited[c] = true;
            e += eval_in(c, target);
          }
          c = n0 * n0;
          if (c < THRESH && !visited[c]) {
            visited[c] = true;
            e += eval_in(c, target);
          }
          c = n0 * n1;
          if (c < THRESH && !visited[c]) {
            visited[c] = true;
            e += eval_in(c, target);
          }
          c = n1 * n1;
          if (c < THRESH && !visited[c]) {
            visited[c] = true;
            e += eval_in(c, target);
          }
          c = max(n0, n1) / min(n0, n1);
          if (!visited[c]) {
            visited[c] = true;
            e += eval_in(c, target);
          }
        }
      }
      return match * 10000 + e;
    }

    int packCmd(int id1, char op, int id2) {
      return (id1 << 20) + (int(op) << 10) + id2;
    }

    void operate(int id0, char op, int id1) {
      int n0 = numMap[id0];
      int n1 = numMap[id1];
      int n = -1;
      switch (op) {
      case '+':
        n = n0 + n1;
        break;
      case '-':
        n = n0 - n1;
        break;
      case '*':
        n = n0 * n1;
        break;
      case '/':
        n = n0 / n1;
        break;
      }
      if (!n) {
        dump(n0, n1);
      }
      push(n);
      cmds.push_back(packCmd(id0, op, id1));
    }

    void undo() {
      int i = numMap.size() - 1;
      int num = numMap[i];
      numMap.erase(i);
      cmds.pop_back();
      if (table->smallTargetSet.count(num)) match--;
    }

    shared_ptr<State> getBestNextState() {
      State bestState(*this);
      int bestScore = INT_MIN;

      unordered_set<int> numSet;
      for (const auto& e : numMap) numSet.insert(e.second);

      for (int id0 = 0; id0 < numMap.size(); id0++) {
        for (int id1 = id0; id1 < numMap.size(); id1++) {
          int n0 = numMap[id0], n1 = numMap[id1];
          int n;

          n = n0 + n1;
          if (n < THRESH && !numSet.count(n)) {
            operate(id0, '+', id1);
            int e = eval();
            if (bestScore < e) {
              bestScore = e;
              bestState = *this;
            }
            undo();
          }

          n = abs(n0 - n1);
          if (n != 0 && !numSet.count(n)) {
            if (n0 <= n1)
              operate(id1, '-', id0);
            else
              operate(id0, '-', id1);
            int e = eval();
            if (bestScore < e) {
              bestScore = e;
              bestState = *this;
            }
            undo();
          }

          n = n0 * n0;
          if (n < THRESH && !numSet.count(n)) {
            operate(id0, '*', id0);
            int e = eval();
            if (bestScore < e) {
              bestScore = e;
              bestState = *this;
            }
            undo();
          }

          n = n0 * n1;
          if (n < THRESH && !numSet.count(n)) {
            operate(id0, '*', id1);
            int e = eval();
            if (bestScore < e) {
              bestScore = e;
              bestState = *this;
            }
            undo();
          }

          n = n1 * n1;
          if (n < THRESH && !numSet.count(n)) {
            operate(id1, '*', id1);
            int e = eval();
            if (bestScore < e) {
              bestScore = e;
              bestState = *this;
            }
            undo();
          }

          n = max(n0, n1) / min(n0, n1);
          if (n != 0 && !numSet.count(n)) {
            if (n0 <= n1)
              operate(id1, '/', id0);
            else
              operate(id0, '/', id1);
            int e = eval();
            if (bestScore < e) {
              bestScore = e;
              bestState = *this;
            }
            undo();
          }
        }
      }
      return make_shared<State>(bestState);
    }

    void preProcess() {
      operate(0, '/', 0);
      operate(2, '+', 2);
      operate(3, '+', 3);
      operate(3, '+', 4);
      operate(4, '+', 5);
      operate(6, '*', 6);
      operate(7, '*', 7);
    }

    int findId(int num) {
      for (const auto& entry : numMap) {
        if (num == entry.second)
          return entry.first;
      }
      return -1;
    }

    void postProcess() {
      int e4 = findId(10000);

      int pos = findId(table->smallTargetList[0]);
      for (int i = 1; i < table->smallTargetList.size() - 1; i++) {
        operate(pos, '*', e4);
        pos = numMap.size() - 1;
        if (table->smallTargetList[i] != 0) {
          int tid = findId(table->smallTargetList[i]);
          operate(pos, '+', tid);
          pos = numMap.size() - 1;
        }
      }

      if (table->smallTargetList.size() > 1) {
        int i = table->smallTargetList.size() - 1;
        int len = table->smallTargetStringList[i].length();
        int ep = 1;
        for (int j = 0; j < len; j++)
          ep *= 10;
        int eid = findId(ep);
        if (eid == -1) {
          int e1 = findId(10);
          operate(e4, '/', e1);
          eid = findId(1000);
        }
        operate(pos, '*', eid);
        pos = numMap.size() - 1;
        if (table->smallTargetList[i] != 0) {
          int tid = findId(table->smallTargetList[i]);
          operate(pos, '+', tid);
        }
      }
    }

    string getCmdString(int pack) {
      int id1 = pack >> 20;
      char op = (char)((pack >> 10) & 0b1111111111);
      int id2 = pack & 0b1111111111;
      return to_string(id1) + " " + op + " " + to_string(id2);
    }

    vector<string> getAns() {
      vector<string> ans;
      for (int i = 0; i < cmds.size(); i++) {
        ans.push_back(getCmdString(cmds[i]));
      }
      return ans;
    }
  };
  using StatePtr = shared_ptr<State>;
}

class NumberCreator {
public:
  vector<string> findSolution(int num0, int num1, string T) {

    vector<string> bsol, dsol;

    {
      using namespace NBinarySolver;

      target = BigInt(T);
      rle = target.getRLE();

      StatePtr firstState(new State(num0, num1));

      int maxTurn = rle.size();
      using Q = priority_queue<StatePtr, vector<StatePtr>, greater<StatePtr>>;
      vector<Q> states(maxTurn + 1);

      states[0].push(firstState);

      int width = 1000;
      for (int t = 0; t < maxTurn; t++) {
        for (int w = 0; w < width; w++) {
          if (states[t].empty()) break;
          StatePtr nowState = states[t].top(); states[t].pop();
          for (StatePtr nextState : nowState->getAllNextStates()) {
            int nt = nextState->ptrRLE;
            states[nt].push(nextState);
          }
        }
        states[t] = Q();
      }

      bsol = states[maxTurn].top()->getAns();
    }

    {
      using namespace NDecimalSolver;

      TestCasePtr tc(new TestCase(num0, num1, T));
      ScoreTablePtr stp(new ScoreTable(tc));

      StatePtr firstState(new State(tc, stp));

      firstState->preProcess();

      StatePtr nowState = firstState;
      int t = 0;
      while (!nowState->isCompleted()) {
        nowState = nowState->getBestNextState();
        t++;
        dump(t, nowState->eval());
      }

      nowState->postProcess();

      dsol = nowState->getAns();
    }

    return bsol.size() < dsol.size() ? bsol : dsol;
  }

};

int main() {
  NumberCreator nc;
  int Num0;
  int Num1;
  string T;

  cin >> Num0;
  cin >> Num1;
  cin >> T;

  //Num0 = 325;
  //Num1 = 175;
  //T = "3294684590760609785571660722402890009467790483189244099871289896781377090183241049692668665134173673";

  vector<string> ret = nc.findSolution(Num0, Num1, T);
  cout << ret.size() << endl;
  for (int i = 0; i < (int)ret.size(); ++i)
    cout << ret[i] << endl;
  cout.flush();
}