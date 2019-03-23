#define _CRT_SECURE_NO_WARNINGS
#define NDEBUG
#include "bits/stdc++.h"
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#ifdef _MSC_VER
#include <ppl.h>
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

template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T &val) { std::fill((T*)array, (T*)(array + N), val); }

void dump_func() { DUMPOUT << endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPOUT << head; if (sizeof...(Tail) == 0) { DUMPOUT << " "; } else { DUMPOUT << ", "; } dump_func(std::move(tail)...); }

#define PI 3.14159265358979323846
#define EPS 1e-10
#define FOR(i,a,n) for(int i=(a);i<(n);++i)
#define REP(i,n)  FOR(i,0,n)
#define all(j) (j).begin(), (j).end()
#define SZ(j) ((int)(j).size())
#define fake false

#define X(xyz) ((xyz) & 0b1111111111)
#define Y(xyz) (((xyz) >> 10) & 0b1111111111)
#define Z(xyz) ((xyz) >> 20)
#define XYZ(x, y, z) (((z) << 20) | ((y) << 10) | (x))
#define IJ(x, y) (((y) << 16) | (x))


class timer {
	vector<timer> timers;
	int n = 0;
public:
#ifdef _MSC_VER
	double limit = 19.9;
#else
	double limit = 19.9;
#endif
	double t = 0;
	timer() {}
	timer(int size) : timers(size) {}
	bool elapses() const {
		return time() - t > limit;
	}
	void measure() {
		t = time() - t;
		++n;
	}
	void measure(char id) {
		timers[id].measure();
	}
	void print() {
		if (n % 2)
			measure();
		for (int i = 0; i < 128; ++i) {
			if (timers[i].n)
				cerr << (char)i << ' ' << timers[i].t << 's' << endl;
		}
		cerr << "  " << t << 's' << endl;
	}
	static double time() {
#ifdef _MSC_VER
		return __rdtsc() / 2.6e9;
#else
		unsigned long long a, d;
		__asm__ volatile ("rdtsc" : "=a" (a), "=d" (d));
		return (d << 32 | a) / 2.8e9;
#endif
	}
} timer(128);

struct SXor128 {
	unsigned x, y, z, w;
	SXor128() { x = 123456789; y = 362436069; z = 521288629; w = 88675123; }
	SXor128(int _w) { x = 123456789; y = 362436069; z = 521288629; w = _w; }
	void setSeed() { x = 123456789; y = 362436069; z = 521288629; w = 88675123; }
	void setSeed(int _w) { x = 123456789; y = 362436069; z = 521288629; w = _w; }
	unsigned nextUInt() {
		unsigned t = (x ^ (x << 11));
		x = y; y = z; z = w;
		return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
	}
	unsigned nextUInt(unsigned mod) {
		unsigned t = (x ^ (x << 11));
		x = y; y = z; z = w;
		w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
		return w % mod;
	}
	unsigned nextUInt(unsigned l, unsigned r) {
		unsigned t = (x ^ (x << 11));
		x = y; y = z; z = w;
		w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
		return w % (r - l + 1) + l;
	}
	double nextDouble() {
		return double(nextUInt()) / UINT_MAX;
	}
} rnd;



constexpr int L = 1000;
constexpr int N = 1000;
constexpr int M = 100000;
vector<int> R, P, A, B, C, D;

struct Ball {
	int id;
	int r, p;
	int bonusSum = 0; //  bonus の総和
	Ball() {}
	Ball(int id, int r, int p) : id(id), r(r), p(p) {}
};
vector<Ball> balls;
vector<vector<vector<int>>> hm;

inline int dist2(int xyz1, int xyz2) {
	int dx = X(xyz1) - X(xyz2), dy = Y(xyz1) - Y(xyz2), dz = Z(xyz1) - Z(xyz2);
	return dx * dx + dy * dy + dz * dz;
}
inline bool isContained(int r, int xyz) {
	int x = X(xyz), y = Y(xyz), z = Z(xyz);
	return !(x - r < 0 || y - r < 0 || z - r < 0 || x + r > L || y + r > L || z + r > L);
}

struct State {
	vector<int> pos; // ball[i] の位置
	set<int> placed; // 配置済み
	int score;

	State() : pos(N, -1), score(0) {}

	// ボーナス計算
	int bonus(int i, int j) {
		if (i > j) swap(i, j);
		int ij = IJ(i, j);
		if (hm[i][j].empty()) return 0;
		int ret = 0;
		int d2 = dist2(pos[i], pos[j]);
		for (const auto& e : hm[i][j]) {
			int p = e >> 10;
			int d = e & 0b1111111111;
			if (d2 <= d * d)
				ret += p;
		}
		return ret;
	}

	// 配置
	bool place(int id, int xyz) {
		Ball& b = balls[id];
		if (!isContained(b.r, xyz)) return false;
		for(int i : placed){
			int R = balls[id].r + balls[i].r;
			if (dist2(xyz, pos[i]) < R * R) return false;
		}
		pos[id] = xyz;
		score += b.p;
		for (int j : placed)
			score += bonus(id, j);
		placed.insert(id);
		return true;
	}

	// 除去
	void remove(int id) {
		Ball& b = balls[id];
		score -= b.p;
		placed.erase(id);
		for (int j : placed)
			score -= bonus(id, j);
		pos[id] = -1;
	}

	// swap チェック
	bool check(int u, int v) {
		// 場外からもってくる
		if (pos[v] == -1) {
			if (balls[u].r >= balls[v].r)
				return true;
			else {
				if (!isContained(balls[v].r, pos[u]))
					return false;
				// u 以外の placed との距離を測る
				for (int w : placed) {
					if (u == w) continue;
					int R = balls[v].r + balls[w].r;
					if (dist2(pos[u], pos[w]) < R * R)
						return false;
				}
				return true;
			}
		}
		// 立方体内に存在する 2 球を交換する
		else {
			if (balls[u].r == balls[v].r)
				return true;
			else {
				if (!isContained(balls[v].r, pos[u]) || !isContained(balls[u].r, pos[v]))
					return false;
				// u, v 以外の placed との距離を測る
				for (int w : placed) {
					if (w == u || w == v) continue;
					int R = balls[v].r + balls[w].r;
					if (dist2(pos[u], pos[w]) < R * R)
						return false;
					R = balls[u].r + balls[w].r;
					if (dist2(pos[v], pos[w]) < R * R)
						return false;
				}
				return true;
			}
		}
	}

	// 交換処理
	int swapBall(int u, int v) {
		int uxyz = pos[u], vxyz = pos[v];

		int prevScore = score;
		if (pos[v] == -1) {
			remove(u);
			place(v, uxyz);
			return score - prevScore;
		}
		else {
			remove(u);
			remove(v);
			place(v, uxyz);
			place(u, vxyz);
			return score - prevScore;
		}
		return INT_MIN / 8;
	}

};

// 最初の 8 つをゴリっと配置するやつ
State solveGreedy() {
	vector<Ball*> pBalls(N, nullptr);
	REP(i, N) pBalls[i] = &balls[i];

	// たくさんボーナスが入りそうなのを優先
	sort(all(pBalls), [&](const Ball* a, const Ball* b) {
		return a->bonusSum > b->bonusSum;
	});

	State bestState;
	// 雑乱択
	while (timer.time() - timer.t < 0.3) {
		int r1 = rnd.nextUInt(32), r2;
		do {
			r2 = rnd.nextUInt(32);
		} while (r1 == r2);
		swap(pBalls[r1], pBalls[r2]);

		State state;

		for (int d = 0; d < 8; d++) {
			auto& pb = pBalls[d];
			int r = pb->r;
			int point[3];
			for (int b = 0; b < 3; b++) point[b] = ((d >> b) & 1) ? 500 - r : 500 + r;
			state.place(pb->id, XYZ(point[0], point[1], point[2]));
		}

		if (bestState.score < state.score) {
			bestState = state;
		}
		else {
			swap(pBalls[r1], pBalls[r2]);
		}
	}

	// 残りをガチャっと詰める
	int pos = 8;
	for (int pos = 8; pos < pBalls.size(); pos++) {
		for (int n = 0; n < 10000; n++) {
			int x = rnd.nextUInt(999) + 1;
			int y = rnd.nextUInt(999) + 1;
			int z = rnd.nextUInt(999) + 1;
			if (bestState.place(pBalls[pos]->id, XYZ(x, y, z))) {
				break;
			}
		}
	}

	return bestState;
}

double getTemp(double startTemp, double endTemp, double t, double T, double deg = 1.0) {
	return endTemp + (startTemp - endTemp) * pow((T - t) / T, deg);
}

// サイズの差の絶対値が margin 以下の 2 球の位置を swap する焼きなまし
void solveSA(State& state) {
	int margin = 4;

	// サイズの近い球の matrix
	vector<vector<int>> simBalls(201);
	REP(i, N) {
		int br = balls[i].r;
		int rmin = max(1, br - margin);
		int rmax = min(200, br + margin);
		for (int r = rmin; r <= rmax; r++)
			simBalls[r].push_back(i);
	}

	int loopcnt = 0;
	double maxTemp = 10;
	double S = timer.time() - timer.t;
	double T = 2.97;
	const unsigned RR = (1 << 30), mask = (1 << 30) - 1;
	while (timer.time() - timer.t < T) {
		loopcnt++;

		// placed から 1 つ選ぶ (TODO: 高速化)
		vector<int> cand;
		REP(i, N) if (state.pos[i] != -1) cand.push_back(i);
		int u = cand[rnd.nextUInt(cand.size())];

		// v = near(u)
		auto& sim = simBalls[balls[u].r];
		if (sim.empty()) continue;
		int v = sim[rnd.nextUInt(sim.size())];
		if (u == v) continue;

		// 交換可能かチェック
		if (!state.check(u, v)) continue;

		int diff = state.swapBall(u, v);
		double temp = getTemp(maxTemp, 0.1, timer.time() - timer.t - S, T - S);
		double prob = exp(diff / temp);

		if (RR * prob <= (rnd.nextUInt() & mask)) {
			state.swapBall(v, u);
		}

		if (loopcnt % 10000 == 0) {
			dump(loopcnt, state.score);
		}
	}
	dump(loopcnt);
}


int main(int argc, char** argv) {
	timer.measure();
#ifdef _MSC_VER
	FILE* fp = fopen("in.txt", "r");
#else
	FILE* fp = stdin;
#endif

	{ int buf; fscanf(fp, "%d%d%d", &buf, &buf, &buf); }

	R = vector<int>(N);
	P = vector<int>(N);
	A = vector<int>(M);
	B = vector<int>(M);
	C = vector<int>(M);
	D = vector<int>(M);

	balls.reserve(N);
	REP(i, N) {
		fscanf(fp, "%d%d", &R[i], &P[i]);
		balls.emplace_back(i, R[i], P[i]);
	}

	hm = vector<vector<vector<int>>>(N, vector<vector<int>>(N));
	REP(i, M) {
		fscanf(fp, "%d%d%d%d", &A[i], &B[i], &C[i], &D[i]);
		A[i]--; B[i]--;
		hm[A[i]][B[i]].emplace_back((D[i] << 10) | C[i]);
		balls[A[i]].bonusSum += D[i];
		balls[B[i]].bonusSum += D[i];
	}

	State state = solveGreedy();
	solveSA(state);

	vector<int> x(N, -1), y(N, -1), z(N, -1);
	REP(id, N) {
		int xyz = state.pos[id];
		if (xyz == -1) continue;
		x[id] = X(xyz);
		y[id] = Y(xyz);
		z[id] = Z(xyz);
	}
	REP(i, N) printf("%d %d %d\n", x[i], y[i], z[i]);

	return 0;
}