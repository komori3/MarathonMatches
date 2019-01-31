#define NDEBUG
#include "bits/stdc++.h"
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#ifdef _MSC_VER
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
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

#define X(xy) ((xy) & 0xFF)
#define Y(xy) ((xy) >> 8)
#define XY(x, y) (((y) << 8) | (x))



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



class NaiveMat2D {
public:
	int H, W;
	int data[64][64];
	NaiveMat2D(int H, int W) : H(H), W(W) {
		memset(data, 0, sizeof(data));
	}

	inline int at(int i, int j) {
		return data[i][j];
	}

	void add(int i1, int j1, int i2, int j2, int x) {
		for (int i = i1; i < i2; i++) for (int j = j1; j < j2; j++) {
			data[i][j] += x;
		}
	}

	int maxval(int i1, int j1, int i2, int j2) {
		int ret = INT_MIN;
		for (int i = i1; i < i2; i++) for (int j = j1; j < j2; j++) {
			ret = max(ret, data[i][j]);
		}
		return ret;
	}

	int minval(int i1, int j1, int i2, int j2) {
		int ret = INT_MAX;
		for (int i = i1; i < i2; i++) for (int j = j1; j < j2; j++) {
			ret = min(ret, data[i][j]);
		}
		return ret;
	}
};

struct SegmentTree {
	static const int n = 1 << 6;
	int segmin[2 * n - 1], segmax[2 * n - 1], segadd[2 * n - 1];
	SegmentTree() {
		REP(i, 2 * n - 1) {
			segmin[i] = 0;
			segmax[i] = 0;
			segadd[i] = 0;
		}
	}

	void add(int a, int b, int x, int k = 0, int l = 0, int r = n) {
		if (r <= a || b <= l) return;
		if (a <= l && r <= b) {
			segadd[k] += x;
			return;
		}
		add(a, b, x, k * 2 + 1, l, (l + r) / 2);
		add(a, b, x, k * 2 + 2, (l + r) / 2, r);

		segmin[k] = min(segmin[k * 2 + 1] + segadd[k * 2 + 1], segmin[k * 2 + 2] + segadd[k * 2 + 2]);
		segmax[k] = max(segmax[k * 2 + 1] + segadd[k * 2 + 1], segmax[k * 2 + 2] + segadd[k * 2 + 2]);
	}

	int minval(int a, int b, int k = 0, int l = 0, int r = n) {
		if (r <= a || b <= l) return INT_MAX;
		if (a <= l && r <= b) return segmin[k] + segadd[k];
		int vl = minval(a, b, k * 2 + 1, l, (l + r) / 2);
		int vr = minval(a, b, k * 2 + 2, (l + r) / 2, r);

		return min(vl, vr) + segadd[k];
	}
	int maxval(int a, int b, int k = 0, int l = 0, int r = n) {
		if (r <= a || b <= l) return INT_MIN;
		if (a <= l && r <= b) return segmax[k] + segadd[k];
		int vl = maxval(a, b, k * 2 + 1, l, (l + r) / 2);
		int vr = maxval(a, b, k * 2 + 2, (l + r) / 2, r);

		return max(vl, vr) + segadd[k];
	}
	int operator[](const int &k) {
		return minval(k, k + 1);
	}
};

struct Segtree2D {
	int H, W;
	SegmentTree data[64];
	Segtree2D(int H, int W) : H(H), W(W) {}
	inline int at(int i, int j) {
		return data[i][j];
	}
	void add(int i1, int j1, int i2, int j2, int x) {
		for (int i = i1; i < i2; i++)
			data[i].add(j1, j2, x);
	}
	int maxval(int i1, int j1, int i2, int j2) {
		int ret = INT_MIN;
		for (int i = i1; i < i2; i++)
			ret = max(ret, data[i].maxval(j1, j2));
		return ret;
	}
	int minval(int i1, int j1, int i2, int j2) {
		int ret = INT_MAX;
		for (int i = i1; i < i2; i++)
			ret = min(ret, data[i].minval(j1, j2));
		return ret;
	}
};

class SqrtDecomp2D {
	int H, W;
	int sqH, sqW, bcH, bcW;
	int data[64][64];
	int bcAdd[12][12], bcMin[12][12], bcMax[12][12];
	int bis[4][64], bjs[4][64];
public:
	SqrtDecomp2D(int H, int W) : H(H), W(W) {
		memset(data, 0, sizeof(data));
		memset(bcAdd, 0, sizeof(bcAdd));
		memset(bcMin, 0, sizeof(bcMin));
		memset(bcMax, 0, sizeof(bcMax));
		memset(bis, 0, sizeof(bis));
		memset(bjs, 0, sizeof(bjs));

		sqH = (int)ceil(sqrt(H)); sqW = (int)ceil(sqrt(H));
		bcH = (H + sqH - 1) / sqH; bcW = (W + sqW - 1) / sqW;

		for (int bi = 0; bi < bcH; bi++) for (int bj = 0; bj < bcW; bj++) {
			int bmin = INT_MAX, bmax = INT_MIN;
			int si = bi * sqH, ei = min(H, si + sqH);
			int sj = bj * sqW, ej = min(W, sj + sqW);
			for (int i = si; i < ei; i++) for (int j = sj; j < ej; j++) {
				bmin = min(bmin, data[i][j]);
				bmax = max(bmax, data[i][j]);
			}
			bcMin[bi][bj] = bmin;
			bcMax[bi][bj] = bmax;
		}

		REP(i, H + 1) {
			bis[1][i] = (i + sqH - 1) / sqH;
			bis[0][i] = (i % sqH) ? bis[1][i] - 1 : bis[1][i];
			bis[2][i] = i / sqH;
			bis[3][i] = (i % sqH) ? bis[2][i] + 1 : bis[2][i];
		}
		REP(j, W + 1) {
			bjs[1][j] = (j + sqW - 1) / sqW;
			bjs[0][j] = (j % sqW) ? bjs[1][j] - 1 : bjs[1][j];
			bjs[2][j] = j / sqW;
			bjs[3][j] = (j % sqW) ? bjs[2][j] + 1 : bjs[2][j];
		}
	}

	void print() {
		for (int i = 0; i < H; i++) {
			for (int j = 0; j < W; j++) {
				fprintf(stderr, "%3d", data[i][j] + bcAdd[i / sqH][j / sqW]);
			}
			fprintf(stderr, "\n");
		}
		fprintf(stderr, "\n");
		for (int i = 0; i < bcH; i++) {
			for (int j = 0; j < bcW; j++) {
				fprintf(stderr, "%3d", bcAdd[i][j]);
			}
			fprintf(stderr, "\n");
		}
		fprintf(stderr, "\n");
		for (int i = 0; i < bcH; i++) {
			for (int j = 0; j < bcW; j++) {
				fprintf(stderr, "%3d", bcMax[i][j]);
			}
			fprintf(stderr, "\n");
		}
		fprintf(stderr, "\n");
		for (int i = 0; i < bcH; i++) {
			for (int j = 0; j < bcW; j++) {
				fprintf(stderr, "%3d", bcMin[i][j]);
			}
			fprintf(stderr, "\n");
		}
		fprintf(stderr, "\n");
	}

	inline int at(int i, int j) {
		return data[i][j] + bcAdd[i / sqH][j / sqW];
	}

	void add(int i1, int j1, int i2, int j2, int x) {
		for (int by = bis[0][i1]; by < bis[3][i2]; by++) for (int bx = bjs[0][j1]; bx < bjs[3][j2]; bx++) {
			if (by >= bis[1][i1] && by < bis[2][i2] && bx >= bjs[1][j1] && bx < bjs[2][j2]) {
				bcAdd[by][bx] += x;
				bcMax[by][bx] += x;
				bcMin[by][bx] += x;
			}
			else {
				int bcmax = INT_MIN, bcmin = INT_MAX, bcadd = bcAdd[by][bx];
				int ii = by * sqH, jj = bx * sqW;
				int si = max(i1, ii), ei = min(i2, ii + sqH);
				int sj = max(j1, jj), ej = min(j2, jj + sqW);
				for (int i = si; i < ei; i++) for (int j = sj; j < ej; j++) {
					data[i][j] += x;
				}
				for (int i = ii; i < ii + sqH; i++) for (int j = jj; j < jj + sqW; j++) {
					bcmax = max(bcmax, data[i][j] + bcadd);
					bcmin = min(bcmin, data[i][j] + bcadd);
				}
				bcMax[by][bx] = bcmax; bcMin[by][bx] = bcmin;
			}
		}
	}

	int maxval(int i1, int j1, int i2, int j2) {
		int ret = INT_MIN;
		for (int by = bis[0][i1]; by < bis[3][i2]; by++) for (int bx = bjs[0][j1]; bx < bjs[3][j2]; bx++) {
			if (by >= bis[1][i1] && by < bis[2][i2] && bx >= bjs[1][j1] && bx < bjs[2][j2]) {
				ret = max(ret, bcMax[by][bx]);
			}
			else {
				int ii = by * sqH, jj = bx * sqW;
				int si = max(i1, ii), ei = min(i2, ii + sqH);
				int sj = max(j1, jj), ej = min(j2, jj + sqW);
				for (int i = si; i < ei; i++) for (int j = sj; j < ej; j++) {
					ret = max(ret, data[i][j] + bcAdd[by][bx]);
				}
			}
		}
		return ret;
	}

	int minval(int i1, int j1, int i2, int j2) {
		int ret = INT_MAX;
		for (int by = bis[0][i1]; by < bis[3][i2]; by++) for (int bx = bjs[0][j1]; bx < bjs[3][j2]; bx++) {
			if (by >= bis[1][i1] && by < bis[2][i2] && bx >= bjs[1][j1] && bx < bjs[2][j2]) {
				ret = min(ret, bcMin[by][bx]);
			}
			else {
				int ii = by * sqH, jj = bx * sqW;
				int si = max(i1, ii), ei = min(i2, ii + sqH);
				int sj = max(j1, jj), ej = min(j2, jj + sqW);
				for (int i = si; i < ei; i++) for (int j = sj; j < ej; j++) {
					ret = min(ret, data[i][j] + bcAdd[by][bx]);
				}
			}
		}
		return ret;
	}
};


#ifdef _MSC_VER
istringstream testcase(unsigned seed) {
	SXor128 r(seed);
	REP(i, 1000) r.nextUInt();

	int H, W, h, w, Kmin, Kmax;
	H = r.nextUInt(5, 50);
	W = r.nextUInt(5, 50);
	h = r.nextUInt(2, H - 1);
	w = r.nextUInt(2, W - 1);
	Kmax = r.nextUInt(1, h*w - 1);
	Kmin = r.nextUInt(0, Kmax - 1);

	if (seed == 1) {
		H = 41; W = 43; h = 11; w = 8; Kmin = 34; Kmax = 39;
	}

	vector<string> numbers(H, string(W, ' '));
	REP(i, H) REP(j, W) numbers[i][j] = char(r.nextUInt(0, 9) + '0');

	string s;
	s +=
		to_string(H) + " " + to_string(W) + " " + to_string(h) + " " +
		to_string(w) + " " + to_string(Kmin) + " " + to_string(Kmax) + "\n";
	REP(i, H) s += numbers[i] + "\n";

	cerr << s << endl;

	return istringstream(s);
}

typedef struct SMouseParams {
	int x, y, event, flags;
	SMouseParams() : x(0), y(0), event(0), flags(0) {}
}*SMouseParamsPtr;
ostream& operator << (ostream& ostr, const SMouseParamsPtr mp) {
	ostr << "{x:" << mp->x << ", y:" << mp->y << ", event:" << mp->event << ", flags:" << mp->flags << "}";
	return ostr;
}
void mouse_callback_function(int event, int x, int y, int flags, void* userdata) {
	SMouseParamsPtr ptr = static_cast<SMouseParamsPtr>(userdata);
	ptr->x = x;
	ptr->y = y;
	ptr->event = event;
	ptr->flags = flags;
}
#endif



int H, W, h, w, Kmin, Kmax;
vector<vector<int>> board;

class State {
public:
	enum EOpType { PAINT, ERASE, SLIDE };
	vector<pii> operations;

	vector<string> cell;
	//SqrtDecomp2D painted;
	NaiveMat2D painted;
	//Segtree2D painted;
	int score;

	State() :
		cell(H, string(W, '.')),
		painted(H, W),
		score(0) {
	}

	inline bool checkPaint(int i, int j) {
		if (cell[i][j] == 'x') return false;
		int i1 = max(0, i - h + 1);
		int i2 = min(H - h + 1, i + 1);
		int j1 = max(0, j - w + 1);
		int j2 = min(W - w + 1, j + 1);
		if (painted.maxval(i1, j1, i2, j2) == Kmax) return false;
		return true;
	};

	inline void paint(int i, int j, bool undo_mode = false) {
		cell[i][j] = 'x';
		int i1 = max(0, i - h + 1), i2 = min(H, i + 1);
		int j1 = max(0, j - w + 1), j2 = min(W, j + 1);
		painted.add(i1, j1, i2, j2, 1);
		score += board[i][j];
		if (!undo_mode) operations.emplace_back(PAINT, XY(j, i));
	}

	inline bool checkErase(int i, int j) {
		if (cell[i][j] == '.') return false;
		int i1 = max(0, i - h + 1), i2 = min(H - h + 1, i + 1);
		int j1 = max(0, j - w + 1), j2 = min(W - w + 1, j + 1);
		if (painted.minval(i1, j1, i2, j2) == Kmin) return false;
		return true;
	}

	inline void erase(int i, int j, bool undo_mode = false) {
		cell[i][j] = '.';
		int i1 = max(0, i - h + 1), i2 = min(H, i + 1);
		int j1 = max(0, j - w + 1), j2 = min(W, j + 1);
		painted.add(i1, j1, i2, j2, -1);
		score -= board[i][j];
		if (!undo_mode) operations.emplace_back(ERASE, XY(j, i));
	}

	inline bool checkSlideLeft(int i, int j) {
		if (cell[i][j] == '.' || j == 0 || cell[i][j - 1] == 'x') return false;
		int i1 = max(0, i - h + 1), i2 = min(H - h + 1, i + 1);
		if (j < W - w + 1 && painted.minval(i1, j, i2, j + 1) == Kmin) return false;
		int nj = j - w;
		if (nj < 0) return true;
		if (painted.maxval(i1, nj, i2, nj + 1) == Kmax) return false;
		return true;
	}

	inline void slideLeft(int i, int j) {
		cell[i][j] = '.'; cell[i][j - 1] = 'x';
		score -= board[i][j]; score += board[i][j - 1];
		int i1 = max(0, i - h + 1), i2 = min(H, i + 1);
		painted.add(i1, j, i2, j + 1, -1);
		int nj = j - w;
		if (nj < 0) return;
		painted.add(i1, nj, i2, nj + 1, 1);
	}

	inline bool checkSlideRight(int i, int j) {
		if (cell[i][j] == '.' || j == W - 1 || cell[i][j + 1] == 'x') return false;
		int i1 = max(0, i - h + 1), i2 = min(H - h + 1, i + 1);
		int nj = j - w + 1;
		if (nj >= 0 && painted.minval(i1, nj, i2, nj + 1) == Kmin) return false;
		nj = j + 1;
		if (nj < W - w + 1 && painted.maxval(i1, nj, i2, nj + 1) == Kmax) return false;
		return true;
	}

	inline void slideRight(int i, int j) {
		cell[i][j] = '.'; cell[i][j + 1] = 'x';
		score -= board[i][j]; score += board[i][j + 1];
		int i1 = max(0, i - h + 1), i2 = min(H, i + 1);
		int nj = j - w + 1;
		if (nj >= 0) painted.add(i1, nj, i2, nj + 1, -1);
		nj = j + 1;
		painted.add(i1, nj, i2, nj + 1, 1);
	}

	inline bool checkSlideUp(int i, int j) {
		if (cell[i][j] == '.' || i == 0 || cell[i - 1][j] == 'x') return false;
		int j1 = max(0, j - w + 1), j2 = min(W - w + 1, j + 1);
		if (i < H - h + 1 && painted.minval(i, j1, i + 1, j2) == Kmin) return false;
		int ni = i - h;
		if (ni < 0) return true;
		if (painted.maxval(ni, j1, ni + 1, j2) == Kmax) return false;
		return true;
	}

	inline void slideUp(int i, int j) {
		cell[i][j] = '.'; cell[i - 1][j] = 'x';
		score -= board[i][j]; score += board[i - 1][j];
		int j1 = max(0, j - w + 1), j2 = min(W, j + 1);
		painted.add(i, j1, i + 1, j2, -1);
		int ni = i - h;
		if (ni < 0) return;
		painted.add(ni, j1, ni + 1, j2, 1);
	}

	inline bool checkSlideDown(int i, int j) {
		if (cell[i][j] == '.' || i == H - 1 || cell[i + 1][j] == 'x') return false;
		int j1 = max(0, j - w + 1), j2 = min(W - w + 1, j + 1);
		int ni = i - h + 1;
		if (ni >= 0 && painted.minval(ni, j1, ni + 1, j2) == Kmin) return false;
		ni = i + 1;
		if (ni < H - h + 1 && painted.maxval(ni, j1, ni + 1, j2) == Kmax) return false;
		return true;
	}

	inline void slideDown(int i, int j) {
		cell[i][j] = '.'; cell[i + 1][j] = 'x';
		score -= board[i][j]; score += board[i + 1][j];
		int j1 = max(0, j - w + 1), j2 = min(W, j + 1);
		int ni = i - h + 1;
		if (ni >= 0) painted.add(ni, j1, ni + 1, j2, -1);
		ni = i + 1;
		painted.add(ni, j1, ni + 1, j2, 1);
	}

	inline bool checkSlide(int i, int j, int dir) {
		switch (dir)
		{
		case 0: return checkSlideRight(i, j);
		case 1: return checkSlideUp(i, j);
		case 2: return checkSlideLeft(i, j);
		case 3: return checkSlideDown(i, j);
		}
	}

	inline void slide(int i, int j, int dir, bool undo_mode = false) {
		static int di[4] = { 0, -1, 0, 1 };
		static int dj[4] = { 1, 0, -1, 0 };
		switch (dir)
		{
		case 0:
			slideRight(i, j);
			break;
		case 1:
			slideUp(i, j);
			break;
		case 2:
			slideLeft(i, j);
			break;
		case 3:
			slideDown(i, j);
			break;
		}
		int xyd = ((dir ^ 2) << 16) | XY(j + dj[dir], i + di[dir]);
		if (!undo_mode) operations.emplace_back(SLIDE, xyd);
	}

	inline bool move(int i1, int j1, int i2, int j2) {
		if (cell[i1][j1] == '.' || cell[i2][j2] == 'x') return false;
		erase(i1, j1);
		paint(i2, j2);
		if (painted.maxval(0, 0, H - h + 1, W - w + 1) > Kmax ||
			painted.minval(0, 0, H - h + 1, W - w + 1) < Kmin) {
			undo(); undo();
			return false;
		}
		return true;
	}

	inline void undo() {
		auto op = operations.back(); operations.pop_back();
		switch (EOpType(op.first))
		{
		case State::PAINT:
			erase(Y(op.second), X(op.second), true);
			break;
		case State::ERASE:
			paint(Y(op.second), X(op.second), true);
			break;
		case State::SLIDE:
		{
			int dir = op.second >> 16;
			int y = Y(op.second & 0xFFFF);
			int x = X(op.second & 0xFFFF);
			slide(y, x, dir, true);
		}
		break;
		default:
			break;
		}
	}

	bool valid() {
		REP(i, H - h + 1) REP(j, W - w + 1) {
			int cnt = 0;
			for (int ii = i; ii < i + h; ii++) for (int jj = j; jj < j + w; jj++) {
				if (cell[ii][jj] == 'x') cnt++;
			}
			if (cnt < Kmin || cnt > Kmax) return false;
		}
		return true;
	}

#ifdef _MSC_VER
	void drawLineNum(cv::Mat3b& img) {
		int mag = 800 / max(H, W);
		REP(i, H + 1) cv::line(img, cv::Point(0, i * mag), cv::Point(W * mag, i * mag), cv::Scalar(0, 0, 0), 1);
		REP(j, W + 1) cv::line(img, cv::Point(j * mag, 0), cv::Point(j * mag, H * mag), cv::Scalar(0, 0, 0), 1);
		REP(i, H) REP(j, W) {
			cv::putText(img, to_string(board[i][j]), cv::Point(j * mag + mag / 3, i * mag + mag * 0.7), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
		}
	}

	void drawLineSum(cv::Mat3b& img) {
		int mag = 800 / max(H, W);
		REP(i, H + 1) cv::line(img, cv::Point(0, i * mag), cv::Point(W * mag, i * mag), cv::Scalar(0, 0, 0), 1);
		REP(j, W + 1) cv::line(img, cv::Point(j * mag, 0), cv::Point(j * mag, H * mag), cv::Scalar(0, 0, 0), 1);
		REP(i, H) REP(j, W) {
			int i1 = max(0, i - h + 1), i2 = min(H, i + 1);
			int j1 = max(0, j - w + 1), j2 = min(W, j + 1);
			cv::putText(img, to_string(painted.maxval(i, j, i + 1, j + 1)), cv::Point(j * mag + mag / 3, i * mag + mag * 0.7), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
		}
	}

	bool vis(int delay = 0) {
		bool flag = true;
		int mag = 800 / max(H, W);
		cv::Mat3b img(H * mag, W * mag, cv::Vec3b(255, 255, 255));
		cv::String winname = "vis";
		REP(i, H) REP(j, W) if (cell[i][j] == 'x') {
			cv::rectangle(img, cv::Rect(j * mag, i * mag, mag, mag), cv::Scalar(255, 255, 0), cv::FILLED);
		}
		drawLineNum(img);
		REP(i, H - h + 1) REP(j, W - w + 1) {
			int i1 = max(0, i - h + 1), i2 = min(H, i + 1);
			int j1 = max(0, j - w + 1), j2 = min(W, j + 1);
			if (painted.maxval(i1, j1, i2, j2) > Kmax) {
				flag = false;
				cv::rectangle(img, cv::Rect(j * mag, i * mag, mag * w, mag * h), cv::Scalar(0, 0, 255), 2);
			}
			else if (painted.minval(i1, j1, i2, j2) < Kmin) {
				flag = false;
				cv::rectangle(img, cv::Rect(j * mag, i * mag, mag * w, mag * h), cv::Scalar(255, 0, 0), 2);
			}
		}
		cv::imshow(winname, img);
		cv::waitKey(delay);
		return flag;
	}

	void play() {
		int mag = 800 / max(H, W);
		cv::Mat3b img(H * mag, W * mag, cv::Vec3b(255, 255, 255));
		cv::Mat3b sum(H * mag, W * mag, cv::Vec3b(255, 255, 255));

		cv::String winname = "vis";
		cv::String winname2 = "sum";
		cv::namedWindow(winname);
		cv::namedWindow(winname2);
		cv::imshow(winname, img);
		cv::imshow(winname2, img);

		SMouseParamsPtr mp(new SMouseParams);

		cv::setMouseCallback(winname, mouse_callback_function, mp);

		int pflag = 0;
		while (true) {
			pflag = mp->flags;
			int key = cv::waitKey(15);
			if (key == 27) break;

			int ix = mp->x / mag, iy = mp->y / mag;

			if (pflag == 0 && mp->flags == 1) {
				if (cell[iy][ix] == 'x') {
					erase(iy, ix);
				}
				else {
					paint(iy, ix);
				}
				cerr << score << endl;
			}

			cv::Mat3b img_cpy = img.clone();
			cv::Mat3b sum_cpy = sum.clone();
			REP(i, H) REP(j, W) if (cell[i][j] == 'x') {
				cv::rectangle(img_cpy, cv::Rect(j * mag, i * mag, mag, mag), cv::Scalar(255, 255, 0), cv::FILLED);
			}
			drawLineNum(img_cpy);
			drawLineSum(sum_cpy);
			REP(i, H - h + 1) REP(j, W - w + 1) {
				int i1 = max(0, i - h + 1), i2 = min(H, i + 1);
				int j1 = max(0, j - w + 1), j2 = min(W, j + 1);
				if (painted.at(i, j) > Kmax) {
					cv::rectangle(img_cpy, cv::Rect(j * mag, i * mag, mag * w, mag * h), cv::Scalar(0, 0, 255), 2);
				}
				else if (painted.at(i, j) < Kmin) {
					cv::rectangle(img_cpy, cv::Rect(j * mag, i * mag, mag * w, mag * h), cv::Scalar(255, 0, 0), 2);
				}
			}

			REP(i, H) REP(j, W) {
				if ((cell[i][j] == '.' && checkPaint(i, j)) || (cell[i][j] == 'x' && checkErase(i, j))) {
					cv::rectangle(img_cpy, cv::Rect(j * mag + 2, i * mag + 2, mag - 4, mag - 4), cv::Scalar(0, 255, 0), 2);
				}
			}
			cv::rectangle(img_cpy, cv::Rect(ix * mag, iy * mag, mag, mag), cv::Scalar(0, 0, 0), 2);

			cv::imshow(winname, img_cpy);
			cv::imshow(winname2, sum_cpy);
		}

		cv::imshow(winname, img);
		cv::waitKey(0);
	}
#endif


	bool bestMove(int i, int j) {
		static int c[64][64];
		erase(i, j);

		memset(c, 0, sizeof(c));
		REP(i, H - h + 1) REP(j, W - w + 1) if (painted.data[i][j] == Kmax) c[i + 1][j + 1]++;
		REP(i, H - h + 1 + 1) for (int j = 0; j < W - w + 1; j++) c[i][j + 1] += c[i][j];
		REP(j, W - w + 1 + 1) for (int i = 0; i < H - h + 1; i++) c[i + 1][j] += c[i][j];
		auto cs = [&](int i1, int j1, int i2, int j2) {
			return c[i2][j2] - c[i1][j2] - c[i2][j1] + c[i1][j1];
		};

		int v = board[i][j];
		int maxDiff = -1, maxXY = -1;
		REP(y, H) REP(x, W) {
			int i1 = max(0, y - h + 1), i2 = min(H - h + 1, y + 1);
			int j1 = max(0, x - w + 1), j2 = min(W - w + 1, x + 1);
			if (cell[y][x] == 'x' || cs(i1, j1, i2, j2)) continue;
			int diff = board[y][x] - v;
			if (maxDiff < diff) {
				maxDiff = diff;
				maxXY = XY(x, y);
			}
		}
		paint(Y(maxXY), X(maxXY));
		return XY(j, i) != maxXY;
	}

	inline bool randomMove(int i, int j) {
		static int c[64][64];
		erase(i, j);

		memset(c, 0, sizeof(c));
		REP(i, H - h + 1) REP(j, W - w + 1) if (painted.data[i][j] == Kmax) c[i + 1][j + 1]++;
		REP(i, H - h + 1 + 1) for (int j = 0; j < W - w + 1; j++) c[i][j + 1] += c[i][j];
		REP(j, W - w + 1 + 1) for (int i = 0; i < H - h + 1; i++) c[i + 1][j] += c[i][j];
		auto cs = [&](int i1, int j1, int i2, int j2) {
			return c[i2][j2] - c[i1][j2] - c[i2][j1] + c[i1][j1];
		};

		int v = board[i][j];
		vector<int> cand;
		REP(y, H) REP(x, W) {
			int i1 = max(0, y - h + 1), i2 = min(H - h + 1, y + 1);
			int j1 = max(0, x - w + 1), j2 = min(W - w + 1, x + 1);
			if (cell[y][x] == 'x' || cs(i1, j1, i2, j2) || board[y][x] < v) continue;
			cand.push_back(XY(x, y));
		}
		int n = cand.size();
		int r = rnd.nextUInt(n);
		int xy = cand[r];
		paint(Y(xy), X(xy));
		return XY(j, i) != xy;
	}

	void paintMinimum() {
		vector<vector<int>> groupSum(h, vector<int>(w, 0));
		for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) {
			for (int ii = i; ii < H; ii += h) for (int jj = j; jj < W; jj += w) {
				groupSum[i][j] += board[ii][jj];
			}
		}

		vector<pii> v;
		for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) {
			v.emplace_back(XY(j, i), groupSum[i][j]);
		}

		sort(all(v), [](const pii& a, const pii& b) {
			return a.second > b.second;
		});

		for (int k = 0; k < Kmin; k++) {
			auto p = v[k].first;
			int x = X(p), y = Y(p);
			for (int i = y; i < H; i += h) for (int j = x; j < W; j += w) {
				paint(i, j);
			}
		}
	}

	void paintMaximum() {
		vector<vector<int>> groupSum(h, vector<int>(w, 0));
		for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) {
			for (int ii = i; ii < H; ii += h) for (int jj = j; jj < W; jj += w) {
				groupSum[i][j] += board[ii][jj];
			}
		}

		vector<pii> v;
		for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) {
			v.emplace_back(XY(j, i), groupSum[i][j]);
		}

		sort(all(v), [](const pii& a, const pii& b) {
			return a.second > b.second;
		});

		for (int k = 0; k < Kmax; k++) {
			auto p = v[k].first;
			int x = X(p), y = Y(p);
			for (int i = y; i < H; i += h) for (int j = x; j < W; j += w) {
				paint(i, j);
			}
		}
	}

	vector<int> getList(char ch = '.') {
		vector<vector<int>> digits(10);
		REP(i, H) REP(j, W) if (cell[i][j] == ch) digits[board[i][j]].push_back(XY(j, i));

		vector<int> ret;
		for (int d = 9; d >= 0; d--) for (auto xy : digits[d]) ret.push_back(xy);

		return ret;
	}

	vector<int> getErasableList() {
		vector<int> xys;
		REP(i, H) REP(j, W) if (checkErase(i, j)) xys.push_back(XY(j, i));
		return xys;
	}

	vector<int> getPaintableList() {
		vector<int> xys;
		REP(i, H) REP(j, W) if (checkPaint(i, j)) xys.push_back(XY(j, i));
		return xys;
	}

	vector<string> paintList(const vector<int>& unpainted) {

		for (auto xy : unpainted) {
			int x = X(xy), y = Y(xy);
			if (checkPaint(y, x)) {
				paint(y, x);
			}
		}
		return cell;
	}
};

class PointsOnGrid {
public:
	void init(int _H, int _W, int _h, int _w, int _Kmin, int _Kmax, const vector<string>& _board) {
		H = _H; W = _W; h = _h; w = _w; Kmin = _Kmin; Kmax = _Kmax;
		board = vector<vector<int>>(H, vector<int>(W));
		REP(i, H) REP(j, W) board[i][j] = _board[i][j] - '0';
	}

	inline double getTemp(double startTemp, double endTemp, double t, double T, double deg = 1.0) {
		return endTemp + (startTemp - endTemp) * pow((T - t) / T, deg);
	}

	State solve1() {
		State bestState;
		int bestScore = INT_MIN;

		State initState;
		initState.paintMinimum();

		vector<int> v = initState.getList();

		State state(initState);
		state.paintList(v);

		bestState = state;
		bestScore = state.score;

		int n = v.size();
		if (n == 0) return bestState;
		int r1 = -1, r2 = -1, loop = 0;
		do {
			loop++;
			while (true) {
				int prev_score = state.score;

				auto v2 = state.getList('x');
				for (int i = v2.size() - 1; i >= 0; i--) {
					int x = X(v2[i]), y = Y(v2[i]);
					if (!state.checkErase(y, x)) continue;
					state.bestMove(y, x);
				}

				if (prev_score == state.score) break;
			}

			if (state.score >= bestScore) {
				if (state.score > bestScore) {
					cerr << state.score << endl;
				}
				bestState = state;
				bestScore = state.score;
			}
			else if (r1 >= 0)
				swap(v[r1], v[r2]);

			r1 = rnd.nextUInt(n);
			do {
				r2 = rnd.nextUInt(n);
			} while (r1 == r2);

			swap(v[r1], v[r2]);

			state = initState;
			state.paintList(v);
		} while (timer.time() - timer.t < 7.7);
		dump(loop);
		return bestState;
	}

	vector<string> findSolution(int _H, int _W, int _h, int _w, int _Kmin, int _Kmax, vector<string> _board) {
		timer.measure();
		init(_H, _W, _h, _w, _Kmin, _Kmax, _board);
		dump(H, W, h, w, Kmin, Kmax);

		State state = solve1();
		
		State bestState = state;

		vector<State> numEraseToState;
		vector<int> numEraseToScore;

		double start = timer.time() - timer.t;
		double end = 9.7;
		const unsigned RR = (1 << 30), mask = (1 << 30) - 1;
		while (timer.time() - timer.t < end) {

			REP(i, 10000) {
				int prev_score = state.score;

				int x = rnd.nextUInt(W);
				int y = rnd.nextUInt(H);
				int dir = rnd.nextUInt(4);
				if (!state.checkSlide(y, x, dir)) continue;
				state.slide(y, x, dir);

				int diff = prev_score - state.score;
				double temp = getTemp(1.0, 0.1, timer.time() - timer.t - start, end - start);
				double prob = exp(-diff / temp);

				if (RR * prob <= (rnd.nextUInt() & mask)) {
					state.undo();
				}
			}

			numEraseToState.push_back(state);
			numEraseToScore.push_back(state.score);
			if (bestState.score < state.score) {
				cerr << state.score << endl;
				bestState = state;
			}

			{
				auto erasable = state.getErasableList();
				if (erasable.empty()) break;
				sort(all(erasable), [&](int a, int b) {
					return board[Y(a)][X(a)] < board[Y(b)][X(b)];
				});
				int xy = erasable[0];
				state.erase(Y(xy), X(xy));
			}

		}

		int NE = numEraseToState.size();
		int erased, maxScore = -1;
		for (int i = 0; i < numEraseToState.size(); i++) {
			State& s = numEraseToState[i];
			if (maxScore < s.score) {
				state = s;
				maxScore = s.score;
				erased = i;
			}
		}

		int accepted = 0, loopcnt = 0;
		while (timer.time() - timer.t < end) {

			REP(i, 100000) {
				loopcnt++;
				int prev_score = state.score;

				if (rnd.nextUInt(100)) {
					int x = rnd.nextUInt(W);
					int y = rnd.nextUInt(H);
					int dir = rnd.nextUInt(4);
					if (!state.checkSlide(y, x, dir)) {
						continue;
					}
					state.slide(y, x, dir);

					int diff = prev_score - state.score;
					double temp = getTemp(1.0, 0.1, timer.time() - timer.t - start, end - start);
					double prob = exp(-diff / temp);

					if (RR * prob <= (rnd.nextUInt() & mask)) {
						state.undo();
					}
					else {
						accepted++;
					}
				}
				else {
					int i1, j1, i2, j2;
					do {
						i1 = rnd.nextUInt(H);
						j1 = rnd.nextUInt(W);
					} while (state.cell[i1][j1] == '.');

					if (state.checkErase(i1, j1)) {
						state.bestMove(i1, j1);
					}
				}

				if (timer.time() - timer.t > end) break;
			}

			if (timer.time() - timer.t > end) break;

			numEraseToScore[erased] = state.score;
			if (bestState.score < state.score) {
				cerr << state.score << endl;
				bestState = state;
			}

			double eraseProb = 0, paintProb = 0, continueProb = 0;
			if (erased < NE - 1) {
				int diff = numEraseToScore[erased] - numEraseToScore[erased + 1];
				if (diff < 0) eraseProb = 1.0;
				else {
					eraseProb = 1.0 / pow(diff + 1.0, 1);
				}
			}
			if (erased > 0) {
				int diff = numEraseToScore[erased] - numEraseToScore[erased - 1];
				if (diff < 0) paintProb = 1.0;
				else {
					paintProb = 1.0 / pow(diff + 1.0, 1);
				}
			}
			if (eraseProb + paintProb > 1.0) {
				double s = eraseProb + paintProb;
				eraseProb /= s;
				paintProb /= s;
			}
			continueProb = 1.0 - eraseProb - paintProb;

			double x = rnd.nextDouble();
			if (x < eraseProb) {
				auto erasable = state.getErasableList();
				if (!erasable.empty()) {
					sort(all(erasable), [&](int a, int b) {
						return board[Y(a)][X(a)] < board[Y(b)][X(b)];
					});
					int xy = erasable[0];
					state.erase(Y(xy), X(xy));
					erased++;
				}
			}
			else if (x < paintProb + eraseProb) {
				auto paintable = state.getPaintableList();
				if (!paintable.empty()) {
					sort(all(paintable), [&](int a, int b) {
						return board[Y(a)][X(a)] > board[Y(b)][X(b)];
					});
					int xy = paintable[0];
					state.paint(Y(xy), X(xy));
					erased--;
				}
			}

			if (loopcnt % 1000000 == 0)
				cerr << accepted << " / " << loopcnt << ", " << state.score << endl;
		}

		return bestState.cell;
	}
};

// -------8<------- end of solution submitted to the website -------8<-------

template<class T> void getVector(vector<T>& v) {
	for (int i = 0; i < v.size(); ++i)
		cin >> v[i];
}

#ifdef _MSC_VER
int _main() {

	PointsOnGrid pog;
	istringstream iss = testcase(1);
	int H, W, h, w, Kmin, Kmax;
	iss >> H >> W >> h >> w >> Kmin >> Kmax;
	vector<string> grid(H);
	iss >> grid;
	vector<string> ret = pog.findSolution(H, W, h, w, Kmin, Kmax, grid);
	for (auto s : ret) cout << s << endl;

	return 0;
}
#endif
int main() {
	PointsOnGrid pog;
	int H;
	int W;
	int h;
	int w;
	int Kmin;
	int Kmax;
	cin >> H;
	cin >> W;
	cin >> h;
	cin >> w;
	cin >> Kmin;
	cin >> Kmax;
	vector<string> grid(H);
	getVector(grid);

	vector<string> ret = pog.findSolution(H, W, h, w, Kmin, Kmax, grid);
	cout << ret.size() << endl;
	for (int i = 0; i < (int)ret.size(); ++i)
		cout << ret[i] << endl;
	cout.flush();
	return 0;
}