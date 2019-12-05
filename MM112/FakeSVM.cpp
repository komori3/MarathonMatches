#include "bits/stdc++.h"

using namespace std;
template <typename _Ty> ostream& operator << (ostream& o, const vector<_Ty>& v) { if (v.empty()) { o << "{ }"; return o; } o << "{" << v.front(); for (auto itr = ++v.begin(); itr != v.end(); itr++) { o << ", " << *itr; } o << "}"; return o; }

// 焼きなましを使ったなんちゃって線形分類
// 入力まんま vs ソルバ で 20000 ケース回した結果を使って 2 クラス分類

// (提出は間に合わなかったが) D, K ごとにモデルを作る
// N, C の重みとバイアスを決める

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
  int seed, s1, s2, sMax;
  double s1rel, s2rel, eval;
  int N, C, D, K;
  Record() {}
  Record(const string& line) {
    auto cols = split(line);
    seed = stoi(cols[0]);
    s1 = stoi(cols[1]);
    s2 = stoi(cols[2]);
    sMax = max(s1, s2);
    s1rel = double(s1) / sMax;
    s2rel = double(s2) / sMax;
    eval = s2rel - s1rel;
    N = stoi(cols[3]);
    C = stoi(cols[4]);
    D = stoi(cols[5]);
    K = stoi(cols[6]);
  }
};

struct Machine {
  double w[3];
  Machine() {
    w[0] = w[1] = w[2] = 0;
  }
  double eval(const Record& r) {
    return w[0] * r.N + w[1] * r.C + w[2] >= 0 ? 1.0 * r.eval : -1.0 * r.eval;
  }
  double eval(const vector<Record>& rs) {
    double ret = 0.0;
    for (const auto& r : rs) ret += eval(r);
    return ret;
  }
  void setRandom(Xorshift& rnd) {
    w[0] = rnd.nextDouble() * 2.0 - 1.0;
    w[1] = rnd.nextDouble() * 2.0 - 1.0;
    w[2] = rnd.nextDouble() * 10.0 - 5.0;
  }
};

double getTemp(double startTemp, double endTemp, int t, int T) {
  return endTemp + (startTemp - endTemp) * (T - t) / T;
}

double weight[6][11][3];

bool useSolver(int N, int C, int D, int K) {
  return weight[D][K][0] * N + weight[D][K][1] * C + weight[D][K][2] >= 0;
}

int main() {
  init();

  ifstream ifs("submissionStats.csv");

  vector<Record> allRecords;

  string line;
  ifs >> line;
  while (ifs >> line) {
    Record record(line);
    allRecords.push_back(record);
  }

  double total = 0;

  for (int D = 1; D <= 5; D++) {
    for (int K = 1; K <= 10; K++) {
      vector<Record> records;
      for (auto& r : allRecords) if (r.D == D && r.K == K) records.push_back(r);

      Xorshift rnd;
      Machine bestM, m;
      double bestScore = DBL_MIN;

      for (int i = 0; i < 100000; i++) {
        m.setRandom(rnd);
        double score = m.eval(records);
        if (bestScore < score) {
          cerr << score << endl;
          bestScore = score;
          bestM = m;
        }
      }

      m = bestM;
      double prevScore = bestScore;
      int numLoop = 100000;
      for (int n = 0; n < numLoop; n++) {
        if (n % 100000 == 0) cerr << "n: " << n << endl;
        int idx = rnd.nextUInt(3);
        double pert = (rnd.nextDouble() * 2.0 - 1.0) * (idx == 2 ? 0.05 : 0.01);
        m.w[idx] += pert;
        double nowScore = m.eval(records);

        double diff = nowScore - prevScore;
        double temp = getTemp(3.0, 0.01, n, numLoop);

        if (diff > temp * lrnd[n & 0xFFFF]) {
          prevScore = nowScore;
          if (bestScore < nowScore) {
            bestScore = nowScore;
            bestM = m;
            cerr << bestScore << endl;
          }
        }
        else {
          m.w[idx] -= pert;
        }
      }

      for (int i = 0; i < 3; i++) {
        bestM.w[i] /= abs(bestM.w[2]);
        weight[D][K][i] = bestM.w[i];
      }
      cout << vector<double>(bestM.w, bestM.w + 3) << endl;
      cout << bestM.eval(records) << endl;
      total += bestM.eval(records);
    }
  }

  cerr << total << endl;
  for (int D = 1; D <= 5; D++) {
    for (int K = 1; K <= 10; K++) {
      for (int i = 0; i < 3; i++) {
        printf("w[%d][%d][%d]=%.10f;", D, K, i, weight[D][K][i]);
      }
      printf("\n");
    }
  }

  return 0;
}
