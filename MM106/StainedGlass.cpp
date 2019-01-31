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

//呪文
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

#define X(xy) ((xy) & 0xFFFF)
#define Y(xy) ((xy) >> 16)
#define XY(x, y) (((y) << 16) | (x))



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

class FastQueue {
	int front, back;
	int v[1 << 20];
public:
	FastQueue() : front(0), back(0) {}
	inline bool empty() { return front == back; }
	inline void push(int x) { v[front++] = x; }
	inline int pop() { return v[back++]; }
	inline void reset() { front = back = 0; }
	inline int size() { return front - back; }
} fqu;



static constexpr int LMAX = 1000;
static constexpr int OFFSET = 100;
static constexpr int NMAX = 1000;
static constexpr int OUTSIDE = INT_MIN;
static constexpr int UNDEFINED = -1;

inline int R(int c) { return c >> 16; }
inline int G(int c) { return (c >> 8) & 0xFF; }
inline int B(int c) { return c & 0xFF; }
inline int RGB(int r, int g, int b) { return (r << 16) | (g << 8) | b; }
inline int colorDiff(int c1, int c2) {
	return abs(R(c1) - R(c2)) + abs(G(c1) - G(c2)) + abs(B(c1) - B(c2));
}

typedef int Point;
inline int dist2(Point p1, Point p2) {
	int dx = X(p1) - X(p2), dy = Y(p1) - Y(p2);
	return dx * dx + dy * dy;
}

int __img[LMAX + 2 * OFFSET][LMAX + 2 * OFFSET];
int* _img[LMAX + 2 * OFFSET];
int** img;
int __voronoi[LMAX + 2 * OFFSET][LMAX + 2 * OFFSET];
int* _voronoi[LMAX + 2 * OFFSET];
int** voronoi;

/* Voronoi cell */
struct Cell {
	int id;
	Point site;

	int area;
	int r[256];
	int g[256];
	int b[256];

	int h[800];
	int w[800];

	int color;
	int difference;

	Point center(bool verbose = false) {
		if (verbose) {
			int mx = 0, my = 0;
			REP(i, 800) {
				mx += i * w[i];
				my += i * h[i];
			}
			return XY(mx / area, my / area);
		}
		else {
			int mx = 0, my = 0;
			REP(i, 800) {
				mx += i * w[i];
				my += i * h[i];
			}
			return XY(mx / area, my / area);
		}
	}
} cells[NMAX];

/* operation */
namespace {
	enum EOpType {
		INSERT, ERASE, MOVE, SLIDE, CHANGE, OP_ARRAY
	};
	class OpBase {
	public:
		EOpType type;
		OpBase() {}
		OpBase(EOpType ty) : type(ty) {}
	};
	typedef shared_ptr<OpBase> OpBasePtr;
	class OpInsert : public OpBase {
	public:
		int sid;
		OpInsert(int s) : OpBase(INSERT), sid(s) {}
	};
	typedef shared_ptr<OpInsert> OpInsertPtr;
	class OpErase : public OpBase {
	public:
		int sid, color;
		Point p_seed;
		OpErase(int s, int c, Point p) : OpBase(ERASE), sid(s), color(c), p_seed(p) {}
	};
	typedef shared_ptr<OpErase> OpErasePtr;
	class OpChange : public OpBase {
	public:
		int sid, oldColor;
		OpChange(int s, int oldc) : OpBase(CHANGE), sid(s), oldColor(oldc) {}
	};
	typedef shared_ptr<OpChange> OpChangePtr;
	class OpMove : public OpBase {
	public:
		int sid, oldXY;
		OpMove(int s, int oxy) : OpBase(MOVE), sid(s), oldXY(oxy) {}
	};
	typedef shared_ptr<OpMove> OpMovePtr;
	class OpSlide : public OpBase {
	public:
		int sid, dir;
		OpSlide(int s, int d) : OpBase(SLIDE), sid(s), dir(d) {}
	};
	typedef shared_ptr<OpSlide> OpSlidePtr;
	class OpArray : public OpBase {
	public:
		vector<OpBasePtr> ops;
		OpArray(const vector<OpBasePtr>& os) : OpBase(OP_ARRAY), ops(os) {}
	};
	typedef shared_ptr<OpArray> OpArrayPtr;
}

class StainedGlass{
	int LH;
	int LW;

	int H;
	int W;
	int N;

	unordered_map<int, int> distinctColors;
	int totalDiff;

	vector<OpBasePtr> operations;

	int loopcnt = 0;

	void addPoint(int cid, Point p);
	void removePoint(int cid, Point p);
	void init(int _H, const vector<int>& px, int _N);
	void findContourI8(int cid, vector<Point>& contour, vector<int>& edge);
	void findContourO4(int cid, vector<Point>& contour, vector<int>& edge);
	bool inflateCellI8(int cid);
	bool inflateCellO4(int cid);
	void createVoronoiWavefront();
	vector<vector<int>> createVoronoiInstanceNaive();
	bool isTrueVoronoi();
	void setInitialSites();

	void undo();
	OpErasePtr erase(int cid, bool undo_mode = false);
	OpInsertPtr insert(int cid, Point s, int color, bool undo_mode = false);
	OpArrayPtr move(int sid, Point nxy, bool undo_mode = false);
	OpChangePtr changeMedianColor(int cid);
	OpArrayPtr changeAll(unordered_map<int, int>& palette);
	OpArrayPtr changeMedianColor(const vector<int>& cids);
	OpChangePtr change(int cid, int newColor, bool undo_mode = false);

	int sampling(const vector<double>& prob);
	int nearestCentroid(int x, int y, int z, const vector<int>& cxs, const vector<int>& cys, const vector<int>& czs);
	vector<vector<int>> getVoronoiImage();
	unordered_set<int> getColorSet(const vector<vector<int>>& img);
	unordered_set<int> getColorSet();
	unordered_map<int, int> getColorSetWithArea();
	unordered_map<int, int> getColorMap(const unordered_map<int, int>& wc, const int k, SXor128 & rnd);
		
#ifdef _MSC_VER
	cv::Mat_<cv::Vec3b> getVoronoi(int numCluster);
	void showVoronoi(int delay = 0);
	void showVoronoi(const vector<vector<int>>& vor, int delay = 0);
	void diffToTrueVoronoi(int delay = 0);
#endif

public:
	vector<int> ret;
	void add(vector<int> t) {
		ret.insert(ret.end(), t.begin(), t.end());
	}
	vector<int> create(int _H, vector<int> px, int _N);
};

void StainedGlass::addPoint(int cid, Point p) {
	Cell& cell = cells[cid];
	int x = X(p), y = Y(p), c = img[y][x];
	cell.r[R(c)]++; cell.g[G(c)]++; cell.b[B(c)]++;
	voronoi[y][x] = cid;
	int diff = colorDiff(c, cell.color);
	totalDiff += diff;
	cell.difference += diff;
	cell.area++;
	cell.h[y]++;
	cell.w[x]++;
}

void StainedGlass::removePoint(int cid, Point p) {
	static const int dx[] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	static const int dy[] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	Cell& cell = cells[cid];
	int x = X(p), y = Y(p), c = img[y][x];
	cell.r[R(c)]--; cell.g[G(c)]--; cell.b[B(c)]--;
	int diff = colorDiff(c, cell.color);
	totalDiff -= diff;
	cell.difference -= diff;
	cell.area--;
	cell.h[y]--;
	cell.w[x]--;
}

void StainedGlass::init(int _H, const vector<int>& px, int _N) {
	H = _H;
	W = px.size() / H;
	N = _N;

	REP(i, H + 2 * OFFSET) {
		_img[i] = &__img[i][OFFSET];
		_voronoi[i] = &__voronoi[i][OFFSET];
	}
	img = &_img[OFFSET];
	voronoi = &_voronoi[OFFSET];

	LH = H + 2 * OFFSET;
	LW = W + 2 * OFFSET;
	REP(i, LH) REP(j, LW) __img[i][j] = __voronoi[i][j] = OUTSIDE;
	
	REP(i, H) REP(j, W) {
		int c = px[i*W + j];
		img[i][j] = c;
		voronoi[i][j] = UNDEFINED;
	}

	totalDiff = 0;
	memset(cells, 0, sizeof(cells));
}

void StainedGlass::findContourI8(int cid, vector<Point>& contour, vector<int>& edge) {
	static const int dy[] = { 0, -1, -1, -1, 0, 1, 1, 1, 0 };
	static const int dx[] = { 1, 1, 0, -1, -1, -1, 0, 1, 1 };
	static const int prevDirToStartIdx[] = { 3, 3, 5, 5, 7, 7, 1, 1 };
	static bitset<NMAX> flags;
	flags.reset();
	contour.clear();
	edge.clear();
	contour.reserve(1024);

	Cell& cell = cells[cid];
	Point inner = cell.site;
	int ix = X(inner), iy = Y(inner);
	while (voronoi[iy - 1][ix] == cid) { iy--; }

	Point s = XY(ix, iy); // 境界上にある
	int sx = X(s), sy = Y(s);
	contour.push_back(s);

	bool exist_val = false;
	bool bw[9];
	REP(d, 9) {
		int x = sx + dx[d], y = sy + dy[d], ccid = voronoi[y][x];
		bw[d] = (ccid == cid);
		if (!bw[d] && ccid >= 0 && !flags[ccid]) {
			flags[ccid] = true;
			edge.push_back(ccid);
		}
		exist_val |= bw[d];
	}
	if (!exist_val) return; // 孤立点

	// p[i]: 白(val 以外) -> p[i+1]: 黒(val) となったら前回の追跡点 pi = (i+5)&7
	int stop_dir = -1;
	int prev_dir = -1;
	Point cur_p;
	REP(d, 8) if (!bw[d] & bw[d + 1]) {
		prev_dir = stop_dir = (d + 5) & 7;
		cur_p = XY(sx + dx[d + 1], sy + dy[d + 1]);
		break;
	}
	do {
		contour.push_back(cur_p);
		// 次の領域境界を探す
		int cx = X(cur_p), cy = Y(cur_p);
		int start_dir = prevDirToStartIdx[prev_dir];
		for (int dir = start_dir; dir < start_dir + 8; dir++) {
			int d = dir & 7;
			int x = cx + dx[d], y = cy + dy[d], ccid = voronoi[y][x];
			if (ccid < 0) continue;
			if (ccid != cid) {
				if (!flags[ccid]) {
					flags[ccid] = true;
					edge.push_back(ccid);
				}
				continue;
			}
			prev_dir = (d + 4) & 7;
			cur_p = XY(x, y);
			break;
		}
	} while (XY(X(cur_p) + dx[prev_dir], Y(cur_p) + dy[prev_dir]) != s);

	return;
}

void StainedGlass::findContourO4(int cid, vector<Point>& contour, vector<int>& edge)
{
	static const int dx4[] = { 1, 0, -1, 0, 1 };
	static const int dy4[] = { 0, -1, 0, 1, 0 };
	static bitset<NMAX> flags;
	flags.reset();
	contour.clear();
	edge.clear();
	contour.reserve(1024);

	Cell& cell = cells[cid];
	Point inner = cell.site;
	int ix = X(inner), iy = Y(inner);
	while (voronoi[iy][ix] == cid) { iy--; }

	Point s = XY(ix, iy); // 境界上にある
	int sx = X(s), sy = Y(s);
	if(voronoi[sy][sx] != OUTSIDE) contour.push_back(s);

	// p[i]: 黒(val) -> p[i+1]: 白(val 以外) となったら前回の追跡点 pi = (i+2)&3
	int stop_dir = -1;
	int prev_dir = -1;
	Point cur_p;
	int cx, cy;
	REP(d, 4) {
		cx = sx + dx4[d]; cy = sy + dy4[d];
		if (voronoi[cy][cx] == cid) continue;
		prev_dir = stop_dir = (d + 2) & 3;
		cur_p = XY(cx, cy);
		break;
	}
	do {
		if (voronoi[cy][cx] != OUTSIDE) contour.push_back(cur_p);
		// 次の領域境界を探す
		int start_dir = (prev_dir + 1) & 3;
		for (int dir = start_dir; dir < start_dir + 4; dir++) {
			int d = dir & 3;
			int x = cx + dx4[d], y = cy + dy4[d], ccid = voronoi[y][x];
			if (ccid == cid) continue;
			if (ccid >= 0) {
				if (!flags[ccid]) {
					flags[ccid] = true;
					edge.push_back(ccid);
				}
			}
			prev_dir = (d + 2) & 3;
			cur_p = XY(x, y); cx = x; cy = y;
			break;
		}
	} while (XY(cx + dx4[prev_dir], cy + dy4[prev_dir]) != s);
	return;
}

bool StainedGlass::inflateCellI8(int cid) {
	static const int dy[] = { 0, -1, -1, -1, 0, 1, 1, 1, 0 };
	static const int dx[] = { 1, 1, 0, -1, -1, -1, 0, 1, 1 };

	Cell& cell = cells[cid];
	const Point s = cell.site;
	int color = cell.color;
	vector<Point> contour;
	vector<Point> edge;
	findContourI8(cid, contour, edge);

	bool update = false;
	for (Point p : contour) {
		int x = X(p), y = Y(p);
		REP(dir, 8) {
			int nx = x + dx[dir], ny = y + dy[dir], ccid = voronoi[ny][nx];
			if (ccid == OUTSIDE || ccid == cid) continue;
			int np = XY(nx, ny);
			if (ccid == UNDEFINED) {
				addPoint(cid, np);
				update = true;
			}
			else {
				// 他領域と競合している場合
				Point np = XY(nx, ny);
				Cell& ccell = cells[ccid];
				int d = dist2(np, cell.site), cd = dist2(np, ccell.site);
				if (d < cd || (d == cd && cid < ccid)) {
					removePoint(ccid, np);
					addPoint(cid, np);
					update = true;
				}
			}
		}
	}
	return update;
}

bool StainedGlass::inflateCellO4(int cid)
{
	Cell& cell = cells[cid];
	const Point s = cell.site;
	int color = cell.color;
	vector<Point> contour;
	vector<Point> edge;
	findContourO4(cid, contour, edge);

	bool update = false;
	for (Point p : contour) {
		int x = X(p), y = Y(p);
		int ccid = voronoi[y][x];
		if (ccid == cid) continue;
		if (ccid == UNDEFINED) {
			addPoint(cid, p);
			update = true;
		}
		else {
			Cell& ccell = cells[ccid];
			int d = dist2(p, cell.site), cd = dist2(p, ccell.site);
			if (d < cd || (d == cd && cid < ccid)) {
				removePoint(ccid, p);
				addPoint(cid, p);
				update = true;
			}
		}
	}
	return update;
}

void StainedGlass::createVoronoiWavefront() {
	bool finished[NMAX] = {};
	//int lc = 0;
	while (true) {
		bool update = false;
		for (int cid = 0; cid < N; cid++) {
			if (finished[cid]) continue;
			finished[cid] = !inflateCellO4(cid);
			update = true;
		}
		if (!update) break;
		//auto img = getVoronoi();
		//cv::imwrite("vis/wavefront" + to_string(lc) + ".png", img);
		//lc++;
	}
}

vector<vector<int>> StainedGlass::createVoronoiInstanceNaive() {
	vector<vector<int>> vor(H, vector<int>(W));
	REP(y, H) REP(x, W) {
		Point p = XY(x, y);
		int minDist = INT_MAX, minIdx = -1;
		REP(i, N) {
			int d = dist2(p, cells[i].site);
			if (d < minDist) {
				minDist = d;
				minIdx = i;
			}
		}
		vor[y][x] = minIdx;
	}
	return vor;
}

bool StainedGlass::isTrueVoronoi()
{
	auto naive = createVoronoiInstanceNaive();
	bool flag = true;
	REP(y, H) REP(x, W)
		if (naive[y][x] != voronoi[y][x]) {
			flag = false;
		}
	return flag;
}

void StainedGlass::setInitialSites() {
	// ボロノイセルがハニカム状になるよう母点を配置
	double L = sqrt(2 * H * W / N / sqrt(3));
	double D = sqrt(sqrt(3) * H * W / N / 2);
	vector<int> xs, ys;
	unordered_set<Point> xys;
	int n = 0, i = 0, j = 0;
	for (double y = 2.0 * D / 3; y <= H - 2.0 * D / 3; i++, y += D) {
		for (double x = L * (i % 2 == 0 ? 0.5 : 1.0); x <= W - L * 0.5; j++, x += L) {
			int ix = round(x), iy = round(y);
			xs.push_back(ix);
			ys.push_back(iy);
			xys.insert(XY(ix, iy));
			n++;
			if (n == N) break;
		}
		if (n == N) break;
	}
	// 配置しきれなかったものは(他の母点と重ならないよう)ランダムに配置
	for (int i = 0; i < N - n; i++) {
		while (true) {
			int ix = rnd.nextUInt(W);
			int iy = rnd.nextUInt(H);
			int ixy = XY(ix, iy);
			if (xys.find(ixy) == xys.end()) {
				xs.push_back(ix);
				ys.push_back(iy);
				xys.insert(ixy);
				break;
			}
		}
	}
	// 母点を設置
	auto setSite = [&](int cid, Point site) {
		int sx = X(site), sy = Y(site);
		int c = img[sy][sx];
		Cell& cell = cells[cid];
		cell.id = cid;
		cell.site = site;
		cell.color = c;
		addPoint(cid, site);
		distinctColors[c]++;
	};
	REP(cid, N) setSite(cid, XY(xs[cid], ys[cid]));
	
	createVoronoiWavefront();
}

OpArrayPtr StainedGlass::changeMedianColor(const vector<int>& cids) {
	vector<OpBasePtr> ops;

	int oldColor = cells[cids.front()].color;
	int thresh = 0;
	int r[256] = {};
	int g[256] = {};
	int b[256] = {};

	for (auto cid : cids) {
		Cell& cell = cells[cid];
		thresh += cell.area;
		REP(c, 256) {
			r[c] += cell.r[c];
			g[c] += cell.g[c];
			b[c] += cell.b[c];
		}
	}

	thresh = (thresh + 1) >> 1;

	int med[3], cum[3] = {};
	int* arr[3] = { r, g, b };
	REP(i, 3) {
		REP(c, 256) {
			cum[i] += arr[i][c];
			if (cum[i] >= thresh) {
				med[i] = c;
				break;
			}
		}
	}

	int newColor = RGB(med[0], med[1], med[2]);
	
	for (auto cid : cids)
		ops.push_back(change(cid, newColor));

	OpArrayPtr ap(new OpArray(ops));
	operations.push_back(ap);

	return ap;
}

OpChangePtr StainedGlass::changeMedianColor(int cid)
{
	Cell& cell = cells[cid];
	int oldColor = cell.color;
	int thresh = (cell.area + 1) >> 1;

	int diff = 0;
	int med[3], cum[3] = {};
	int* arr[3] = { cell.r, cell.g, cell.b };
	REP(i, 3) {
		REP(c, 256) {
			cum[i] += arr[i][c];
			if (cum[i] >= thresh) {
				med[i] = c;
				break;
			}
		}
		REP(c, 256) diff += abs(med[i] - c) * arr[i][c];
	}

	int newColor = RGB(med[0], med[1], med[2]);
	cell.color = newColor;

	distinctColors[oldColor]--;
	if (distinctColors[oldColor] == 0)
		distinctColors.erase(oldColor);

	int prev_diff = cell.difference;
	totalDiff += diff - prev_diff;
	cell.difference = diff;

	distinctColors[newColor]++;

	OpChangePtr cp(new OpChange(cid, oldColor));
	operations.push_back(cp);

	return cp;
}

OpArrayPtr StainedGlass::changeAll(unordered_map<int, int>& palette)
{
	vector<OpBasePtr> ops;
	REP(cid, N) {
		int c = cells[cid].color;
		ops.push_back(change(cid, palette[c]));
	}
	OpArrayPtr ap(new OpArray(ops));
	operations.push_back(ap);
	return ap;
}

OpChangePtr StainedGlass::change(int cid, int newColor, bool undo_mode) {
	Cell& cell = cells[cid];
	int oldColor = cell.color;
	int thresh = (cell.area + 1) >> 1;
	
	int diff = 0;
	int rn = R(newColor), gn = G(newColor), bn = B(newColor);
	
	REP(c, 256)	diff += abs(rn - c) * cell.r[c];
	REP(c, 256)	diff += abs(gn - c) * cell.g[c];
	REP(c, 256)	diff += abs(bn - c) * cell.b[c];
	cell.color = newColor;

	distinctColors[oldColor]--;
	if (distinctColors[oldColor] == 0)
		distinctColors.erase(oldColor);

	int prev_diff = cell.difference;
	totalDiff += diff - prev_diff;
	cell.difference = diff;

	distinctColors[newColor]++;

	OpChangePtr cp(new OpChange(cid, oldColor));
	if (!undo_mode)
		operations.push_back(cp);

	return cp;
}

int StainedGlass::sampling(const vector<double>& prob)
{
	int n = prob.size();
	vector<double> cumu(n, 0);
	for (int i = 1; i < n; i++)
		cumu[i] = cumu[i - 1] + prob[i - 1];
	double d = rnd.nextDouble() * cumu[n - 1];
	int idx = upper_bound(all(cumu), d) - cumu.begin() - 1;
	return idx;
	return 0;
}

int StainedGlass::nearestCentroid(int x, int y, int z, const vector<int>& cxs, const vector<int>& cys, const vector<int>& czs)
{
	int mindist = INT_MAX;
	int minIdx;
	for (int i = 0; i < cxs.size(); i++) {
		int dist = abs(x - cxs[i]) + abs(y - cys[i]) + abs(z - czs[i]);
		if (dist < mindist) {
			mindist = dist;
			minIdx = i;
		}
	}
	return minIdx;
}

vector<vector<int>> StainedGlass::getVoronoiImage()
{
	vector<vector<int>> img(H, vector<int>(W));
	REP(i, H) REP(j, W) img[i][j] = cells[voronoi[i][j]].color;
	return img;
}

unordered_set<int> StainedGlass::getColorSet(const vector<vector<int>>& img)
{
	unordered_set<int> colorSet;
	int H = img.size(), W = img[0].size();
	REP(i, H) REP(j, W) colorSet.insert(img[i][j]);
	return colorSet;
}

unordered_set<int> StainedGlass::getColorSet() {
	unordered_set<int> colorSet;
	REP(i, N) colorSet.insert(cells[i].color);
	return colorSet;
}

unordered_map<int, int> StainedGlass::getColorSetWithArea() {
	unordered_map<int, int> colorSet;
	REP(i, N) colorSet[cells[i].color] += cells[i].area;
	return colorSet;
}

unordered_map<int, int> StainedGlass::getColorMap(const unordered_map<int, int>& wc, const int k, SXor128& rnd) {
	int n = wc.size();
	int wn = 0;

	vector<int> xs, ys, zs, ws;
	for (auto col : wc) {
		xs.push_back(R(col.first));
		ys.push_back(G(col.first));
		zs.push_back(B(col.first));
		ws.push_back(col.second);
		wn += col.second;
	}

	vector<double> prob(n);
	REP(i, n) prob[i] = (double)ws[i] / wn;

	// centroid
	vector<int> cxs, cys, czs;

	for (int i = 0; i < k; i++) {
		int idx = sampling(prob);
		cxs.push_back(xs[idx]);
		cys.push_back(ys[idx]);
		czs.push_back(zs[idx]);

		// update prob
		for (int j = 0; j < n; j++) {
			int nearest = nearestCentroid(xs[j], ys[j], zs[j], cxs, cys, czs);
			prob[j] = pow(abs(xs[j] - cxs[nearest]) + abs(ys[j] - cys[nearest]) + abs(zs[j] - czs[nearest]), 2);
		}
		double sum = accumulate(all(prob), 0.0);
		for (int j = 0; j < n; j++) prob[j] /= sum;
	}

	vector<int> cluster(n);
	for (int i = 0; i < n; i++)
		cluster[i] = nearestCentroid(xs[i], ys[i], zs[i], cxs, cys, czs);

	while (true) {
		// update centroid
		vector<int> cnt(k, 0);
		vector<map<int, int>> ncxs(k), ncys(k), nczs(k);
		for (int i = 0; i < n; i++) {
			ncxs[cluster[i]][xs[i]] += ws[i];
			ncys[cluster[i]][ys[i]] += ws[i];
			nczs[cluster[i]][zs[i]] += ws[i];
			cnt[cluster[i]] += ws[i];
		}
		for (int i = 0; i < k; i++) {
			if (!cnt[i]) continue;

			int area = cnt[i];
			int thresh = (area + 1) >> 1;
			auto& xhist = ncxs[i];
			int xsum = 0, xmed;
			for (auto e : xhist) {
				xsum += e.second;
				if (xsum >= thresh) {
					xmed = e.first;
					break;
				}
			}
			auto& yhist = ncys[i];
			int ysum = 0, ymed;
			for (auto e : yhist) {
				ysum += e.second;
				if (ysum >= thresh) {
					ymed = e.first;
					break;
				}
			}
			auto& zhist = nczs[i];
			int zsum = 0, zmed;
			for (auto e : zhist) {
				zsum += e.second;
				if (zsum >= thresh) {
					zmed = e.first;
					break;
				}
			}

			cxs[i] = xmed;
			cys[i] = ymed;
			czs[i] = zmed;
		}
		
		// update cluster
		bool update = false;
		for (int i = 0; i < n; i++) {
			int idx = nearestCentroid(xs[i], ys[i], zs[i], cxs, cys, czs);
			if (cluster[i] != idx)
				update = true;
			cluster[i] = idx;
		}

		if (!update)
			break;
	}

	unordered_map<int, int> colorMap;
	for (int i = 0; i < n; i++)
		colorMap[RGB(xs[i], ys[i], zs[i])] = RGB(cxs[cluster[i]], cys[cluster[i]], czs[cluster[i]]);
	return colorMap;
}

void StainedGlass::undo() {
	OpBasePtr bp = operations.back();
	operations.pop_back();
	switch (bp->type)
	{
	case ERASE:
	{
		OpErasePtr ep = static_pointer_cast<OpErase>(bp);
		insert(ep->sid, ep->p_seed, ep->color, true);
	}
	break;
	case INSERT:
	{
		OpInsertPtr ip = static_pointer_cast<OpInsert>(bp);
		erase(ip->sid, true);
	}
	break;
	case MOVE:
	{
		OpMovePtr mp = static_pointer_cast<OpMove>(bp);
		move(mp->sid, mp->oldXY, true);
	}
	break;
	case CHANGE:
	{
		OpChangePtr cp = static_pointer_cast<OpChange>(bp);
		change(cp->sid, cp->oldColor, true);
	}
	break;
	case OP_ARRAY:
	{
		OpArrayPtr ap = static_pointer_cast<OpArray>(bp);
		REP(i, ap->ops.size()) undo();
	}
	break;
	default:
		break;
	}
}

OpArrayPtr StainedGlass::move(int cid, Point nxy, bool undo_mode) {
	int nx = X(nxy), ny = Y(nxy);

	vector<OpBasePtr> ops;
	int color = cells[cid].color;
	ops.push_back(erase(cid));
	ops.push_back(insert(cid, nxy, color));
	ops.push_back(changeMedianColor(cid));

	OpArrayPtr ap(new OpArray(ops));
	if (!undo_mode)
		operations.push_back(ap);

	return ap;
}

OpErasePtr StainedGlass::erase(int cid, bool undo_mode) {
	static const int dx[] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	static const int dy[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	fqu.reset();

	Cell& cell = cells[cid];
	int color = cell.color;
	vector<int> contour, edge;
	findContourI8(cid, contour, edge);
	sort(all(edge));
	int n = edge.size();

	// bfs
	Point s = cell.site;
	fqu.push(s);
	while (!fqu.empty()) {
		Point p = fqu.pop();
		int x = X(p), y = Y(p);
		if (voronoi[y][x] != cid) continue;
		int minDist = INT_MAX, minId = -1;
		for (int ccid : edge) {
			int d = dist2(p, cells[ccid].site);
			if (d < minDist) {
				minDist = d;
				minId = ccid;
			}
		}
		Cell& ccell = cells[minId];
		removePoint(cid, p);
		addPoint(minId, p);
		REP(d, 8) {
			int nx = x + dx[d], ny = y + dy[d], v = voronoi[ny][nx];
			if (v == INT_MIN || v != cid) continue;
			fqu.push(XY(nx, ny));
		}
	}

	OpErasePtr ep(new OpErase(cell.id, cell.color, cell.site));
	if (!undo_mode)
		operations.push_back(ep);

	distinctColors[color]--;
	if (distinctColors[color] == 0)
		distinctColors.erase(color);

	return ep;
}

OpInsertPtr StainedGlass::insert(int cid, Point s, int color, bool undo_mode) {
	static const int dx[] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	static const int dy[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	fqu.reset();

	Cell& cell = cells[cid];
	cell.site = s;
	cell.color = color;

	distinctColors[color]++;

	fqu.push(s);
	while (!fqu.empty()) {
		Point p = fqu.pop();
		int x = X(p), y = Y(p), ccid = voronoi[y][x];
		if (ccid == cid) continue;
		int icol = img[y][x];
		Cell& ccell = cells[ccid];
		int ccolor = ccell.color;
		int d = dist2(s, p);
		int cd = dist2(ccell.site, p);
		if (d < cd || (d == cd && cid < ccid)) {
			removePoint(ccid, p);
			addPoint(cid, p);
		}
		else
			continue;

		REP(dir, 8) {
			int nx = x + dx[dir], ny = y + dy[dir], v = voronoi[ny][nx];
			if (v == OUTSIDE || v == cid) continue;
			fqu.push(XY(nx, ny));
		}
	}

	OpInsertPtr ip(new OpInsert(cid));
	if (!undo_mode)
		operations.push_back(ip);

	return ip;
}

double getTemp(double startTemp, double endTemp, double t, double T, double deg = 1.0) {
	return endTemp + (startTemp - endTemp) * pow((T - t) / T, deg);
}



vector<int> StainedGlass::create(int _H, vector<int> px, int _N) {
	timer.measure();
	init(_H, px, _N);
	setInitialSites();
	REP(cid, N) changeMedianColor(cid);


	{
		vector<Cell*> pcells;
		REP(cid, N) pcells.push_back(&cells[cid]);

		double maxTemp = double(totalDiff) / 10000;
		double start = timer.time() - timer.t;
		double T = 18.0 - start;
		const unsigned RR = (1 << 30), mask = (1 << 30) - 1;
		while (timer.time() - timer.t - start < T) {

			double remain = T + timer.t + start - timer.time();
			int coef = max(3, (int)round(rnd.nextDouble() * remain + 3));
			if (coef % 2 == 0) coef++;

			if (loopcnt % 30 == 0) {
				int pdiff = totalDiff;
				sort(all(pcells), [](const Cell* a, const Cell* b) {
					return a->difference > b->difference;
				});

				int cid1 = pcells.back()->id;
				int cid2 = pcells.front()->id;

				int nx, ny;
				{
					Point s2 = cells[cid2].site;
					int s2x = X(s2), s2y = Y(s2);
					while (true) {
						nx = s2x + rnd.nextUInt(5) - 2;
						ny = s2y + rnd.nextUInt(5) - 2;
						int ccid = voronoi[ny][nx];
						if (ccid == OUTSIDE || cells[voronoi[ny][nx]].site == XY(nx, ny)) continue;
						break;
					}
				}

				erase(cid1);
				insert(cid1, XY(nx, ny), 0);
				changeMedianColor(cid1);

				if (pdiff < totalDiff) {
					REP(i, 3) undo();
				}

			}

			int prev_diff = totalDiff;
			int cid = rnd.nextUInt(N);


			Cell& cell = cells[cid];
			int x = X(cell.site), y = Y(cell.site);
			int dx, dy, nx, ny;
			while (true) {
				dx = rnd.nextUInt(coef) - coef / 2;
				dy = rnd.nextUInt(coef) - coef / 2;
				nx = x + dx;
				ny = y + dy;
				if (voronoi[ny][nx] == OUTSIDE) continue;
				if (cells[voronoi[ny][nx]].site == XY(nx, ny)) continue;
				break;
			}

			move(cid, XY(nx, ny));

			int diff = totalDiff - prev_diff;
			double temp = getTemp(maxTemp, 1, timer.time() - timer.t - start, T);
			double prob = exp(-diff / temp);

			if (RR * prob <= (rnd.nextUInt() & mask)) {
				undo();
			}

			if (loopcnt % 1000 == 0) {
				cerr << remain << ", distinct: " << distinctColors.size() << ", diff: " << totalDiff << endl;
#ifdef _MSC_VER
				//auto vor = getVoronoi(temp, remain, loopcnt);
				//cv::imwrite("vis/annealing" + to_string(loopcnt) + ".png", vor);
#endif
			}

			loopcnt++;
		}
	}
	
	REP(i, N) changeMedianColor(i);

	auto wColorSet = getColorSetWithArea();
	int minK = -1;
	double minScore = DBL_MAX;
	unordered_map<int, int> minColorMap;

	for (int k = 2; k <= min(70, N); k++) {
		auto colorMap = getColorMap(wColorSet, k, rnd);

		map<int, vector<int>> colToCids;
		REP(cid, N) {
			int c = colorMap[cells[cid].color];
			colToCids[c].push_back(cid);
		}

		changeAll(colorMap);

		for (auto e : colToCids) {
			changeMedianColor(e.second);
		}

		//cv::Mat_<cv::Vec3b> vor = getVoronoi(k);
		//cv::imwrite("vis/kmeans_" + to_string(k) + ".png", vor);
		//cv::imshow("kmeans++", vor);
		//cv::waitKey(100);


		double score = totalDiff * pow(1 + (double)k / N, 2);
		if (score < minScore) {
			cerr << "numCluster: " << k << ", diff: " << totalDiff << ", score: " << score << endl;
			minK = k;
			minScore = score;
			minColorMap = colorMap;
		}

		REP(i, colToCids.size()) undo();
		undo();
	}

	int dk = (minK + 9) / 10;
	while (timer.time() - timer.t < 19.8) {
		int k = minK + rnd.nextUInt(2 * dk + 1) - dk;
		auto colorMap = getColorMap(wColorSet, k, rnd);

		map<int, vector<int>> colToCids;
		REP(cid, N) {
			int c = colorMap[cells[cid].color];
			colToCids[c].push_back(cid);
		}

		changeAll(colorMap);

		for (auto e : colToCids) {
			changeMedianColor(e.second);
		}

		double score = totalDiff * pow(1 + (double)k / N, 2);
		if (score < minScore) {
			cerr << "numCluster: " << k << ", diff: " << totalDiff << ", score: " << score << endl;
			minScore = score;
			minColorMap = colorMap;
		}

		REP(i, colToCids.size()) undo();
		undo();
	}

	map<int, vector<int>> colToCids;
	for (int cid = 0; cid < N; cid++) {
		int c = minColorMap[cells[cid].color];
		change(cid, c);
		colToCids[c].push_back(cid);
	}

	for (auto e : colToCids) {
		changeMedianColor(e.second);
	}

	// 最後に id でソート
	vector<Cell*> cc;
	for (int cid = 0; cid < N; cid++) cc.push_back(&cells[cid]);
	sort(cc.begin(), cc.end(), [](const Cell* c1, const Cell* c2) {
		return c1->id < c2->id;
	});
	for (int cid = 0; cid < N; cid++) {
		const Cell* c = cc[cid];
		add({ Y(c->site), X(c->site), c->color });
	}

	return ret;
}

#ifdef _MSC_VER
cv::Mat_<cv::Vec3b> StainedGlass::getVoronoi(int numCluster) {
	int X = 64;
	cv::Mat_<cv::Vec3b> img_info(X, W, cv::Vec3b(255, 255, 255));
	double score = totalDiff * pow(1.0 + double(distinctColors.size()) / N, 2.0);
	cv::putText(img_info, "numCluster: " + to_string(numCluster), cv::Point(10, 16), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
	cv::putText(img_info, "diff: " + to_string(totalDiff) + ", score: " + to_string(score), cv::Point(10, 48), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
	cv::Mat_<cv::Vec3b> img_rgb(H, W, cv::Vec3b(255, 255, 255));
	
	REP(i, H) REP(j, W) {
		int cid = voronoi[i][j];
		if (cid < 0) continue;
		int c = cells[cid].color;
		img_rgb[i][j][0] = B(c);
		img_rgb[i][j][1] = G(c);
		img_rgb[i][j][2] = R(c);
	}
	vector<Point> contour, edge;
	REP(i, N) {
		findContourI8(i, contour, edge);
		for (auto p : contour) {
			cv::Point cp(X(p), Y(p));
			img_rgb.at<cv::Vec3b>(cp) = cv::Vec3b(0, 0, 0);
		}
		Point p = cells[i].site;

		cv::Point cp(X(p), Y(p));
		cv::circle(img_rgb, cp, 2, cv::Scalar(0, 0, 0), cv::FILLED);
	}

	cv::Mat_<cv::Vec3b> img_total(H + X, W, cv::Vec3b(255, 255, 255));
	img_info.copyTo(img_total(cv::Rect(0, 0, W, X)));
	img_rgb.copyTo(img_total(cv::Rect(0, X, W, H)));

	return img_total;
}
void StainedGlass::showVoronoi(int delay) {
	cv::Mat_<cv::Vec3b> img_rgb(H, W, cv::Vec3b(255, 255, 255));
	REP(i, H) REP(j, W) {
		int cid = voronoi[i][j];
		//assert(cid >= 0);
		if (cid < 0) continue;
		int c = cells[cid].color;
		img_rgb[i][j][0] = B(c);
		img_rgb[i][j][1] = G(c);
		img_rgb[i][j][2] = R(c);
		//assert(B(c) >= 0 && B(c) < 256 && G(c) >= 0 && G(c) < 256 && R(c) >= 0 && R(c) < 256);
	}
	vector<Point> contour, edge;
	REP(i, N) {
		findContourI8(i, contour, edge);
		for (auto p : contour) {
			cv::Point cp(X(p), Y(p));
			img_rgb.at<cv::Vec3b>(cp) = cv::Vec3b(0, 0, 0);
		}
		Point p = cells[i].site;

		cv::Point cp(X(p), Y(p));
		cv::circle(img_rgb, cp, 2, cv::Scalar(0, 0, 0), cv::FILLED);
	}
	cv::imshow("img", img_rgb);
	cv::waitKey(delay);
}
void StainedGlass::showVoronoi(const vector<vector<int>>& vor, int delay)
{
	cv::Mat_<cv::Vec3b> img_rgb(H, W, cv::Vec3b(255, 255, 255));
	REP(i, H) REP(j, W) {
		int cid = vor[i][j];
		//assert(cid >= 0);
		if (cid < 0) continue;
		int c = cells[cid].color;
		img_rgb[i][j][0] = B(c);
		img_rgb[i][j][1] = G(c);
		img_rgb[i][j][2] = R(c);
		//assert(B(c) >= 0 && B(c) < 256 && G(c) >= 0 && G(c) < 256 && R(c) >= 0 && R(c) < 256);
	}
	REP(i, N) {
		Point p = cells[i].site;
		cv::circle(img_rgb, cv::Point(X(p), Y(p)), 2, cv::Scalar(0, 0, 0), cv::FILLED);
	}
	cv::imshow("img", img_rgb);
	cv::waitKey(delay);
}
void StainedGlass::diffToTrueVoronoi(int delay) {
	auto naive = createVoronoiInstanceNaive();
	cv::Mat3b diffImg(H, W);
	REP(y, H) REP(x, W) {
		if (naive[y][x] == voronoi[y][x]) {
			int c = cells[naive[y][x]].color;
			diffImg[y][x][0] = B(c);
			diffImg[y][x][1] = G(c);
			diffImg[y][x][2] = R(c);
		}
		else {
			diffImg[y][x] = cv::Vec3b(0, 0, 255);
		}
	}
	cv::imshow("diff", diffImg);
	cv::waitKey(delay);
}
#endif

// -------8<------- end of the solution submitted to the website -------8<-------

template<class T> void getVector(vector<T>& v) {
	for (int i = 0; i < (int)v.size(); ++i)
		cin >> v[i];
}


int _main() {
	StainedGlass sg;
	int H;
	cin >> H;
	int S;
	cin >> S;
	vector<int> pixels(S);
	getVector(pixels);

	int N;
	cin >> N;

	vector<int> ret = sg.create(H, pixels, N);
	cout << ret.size() << endl;
	for (int i = 0; i < (int)ret.size(); ++i)
		cout << ret[i] << endl;
	cout.flush();
	return 0;
}

#ifdef _MSC_VER
int main() {
	StainedGlass sg;

	cv::Mat_<cv::Vec3b> img = cv::imread("images/2.jpg");

	int H = img.rows;
	int W = img.cols;
	int S = H * W;
	int N = 1000;
	vector<int> pixels(S);
	REP(i, H) REP(j, W) {
		pixels[i * W + j] = RGB(img[i][j][2], img[i][j][1], img[i][j][0]);
	}

	vector<int> ret = sg.create(H, pixels, N);
	//cerr << ret << endl;

	return 0;
}
#endif