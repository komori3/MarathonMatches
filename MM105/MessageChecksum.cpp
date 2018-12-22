#include "bits/stdc++.h"
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#ifdef _MSC_VER
//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
#include <ppl.h>
#endif

using namespace std;

//呪文
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
#define EPS 1e-10
#define FOR(i,a,n) for(int i=(a);i<(n);++i)
#define REP(i,n)  FOR(i,0,n)
#define all(j) (j).begin(), (j).end()
#define SZ(j) ((int)(j).size())
#define fake false

#define LOCALTEST

class timer {
	vector<timer> timers;
	int n = 0;
public:
#ifdef _MSC_VER
	double limit = 9.9;
#else
	double limit = 9.9;
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
};

#ifdef LOCALTEST
class MessageChecksumVis {
public:
	unsigned seed;
	SXor128 rnd;

	string message;
	vector<string> state;
	double errorRate;
	vector<int> checkSum;
	ll cost;
	bool fail;

	int checkCount;

	MessageChecksumVis() {}
	void generateTestCase(unsigned s, int len = -1, double err = -1.0) {
		rnd.setSeed(s);
		REP(i, 100) rnd.nextUInt();

		cost = 0;
		fail = false;
		checkCount = 0;

		int length = rnd.nextUInt(9901) + 100;
		if (len > 0) length = len;
		checkSum = vector<int>(length + 1);

		for (int i = 0; i < length; i++) {
			int pick = rnd.nextUInt(26);
			message.push_back('A' + pick);
			checkSum[i + 1] = (checkSum[i] + pick) % 26;
		}
		errorRate = 0.01 + rnd.nextDouble() * 0.49;
		if (err >= 0) errorRate = err;
		state = vector<string>(27);
		for (int i = 0; i < 26; i++) state[i].push_back('A' + i);
		for (int i = 26; i > 0; i--) {
			int pick = rnd.nextUInt(i + 1);
			if (pick == i) continue;
			string temp = state[pick];
			state[pick] = state[i];
			state[i] = temp;
		}
	}

	int getChecksum(int start, int length) {
		checkCount++;
		if ((start < 0) || (length < 1) || (length > message.size()) || (start + length > message.size())) {
			cerr << "Invalid parameters to getChecksum: " << start << ", " << length << endl;
			fail = true;
			return -1;
		}
		cost += 5;
		return (26 + checkSum[start + length] - checkSum[start]) % 26;
	}

	string getMessage(int start, int length) {
		if ((start < 0) || (length < 1) || (length > message.size()) || (start + length > message.size())) {
			cerr << "Invalid parameters to getMessage: " << start << ", " << length << endl;
			fail = true;
			return "";
		}
		cost += length;
		string ret = "";
		for (int i = start; i < start + length; i++) {
			if (rnd.nextDouble() < errorRate)
				ret += state[(int)sqrt(floor(rnd.nextDouble() * 729))];
			else
				ret += message[i];
		}
		return ret;
	}

	short editDistance(string s1, string s2) {
		short n = s1.size(), m = s2.size();
		vector<vector<short>> dp(n + 1, vector<short>(m + 1, 0));
		REP(i, n + 1) dp[i][0] = i;
		REP(j, m + 1) dp[0][j] = j;
		for (short i = 1; i <= n; i++) {
			for (short j = 1; j <= m; j++) {
				if (s1[i - 1] == s2[j - 1]) {
					dp[i][j] = dp[i - 1][j - 1];
				}
				else {
					dp[i][j] = 1 + min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
				}
			}
		}
		return dp[n][m];
	}

	ll score(string ans) {
		return (100LL + cost) * (editDistance(message, ans) + 1);
	}
};

namespace Sender {
	MessageChecksumVis vis;
	int getChecksum(int start, int length) {
		return vis.getChecksum(start, length);
	}
	string getMessage(int start, int length) {
		return vis.getMessage(start, length);
	}
	string getTrueMessage() {
		return vis.message;
	}
}
#else
// --8<-- begin of library methods, not submitted to the website ---8<--
namespace Sender {                                             //---8<--
	int getChecksum(int start, int length) {                    //---8<--
		cout << "?getChecksum" << endl;                        //---8<--
		cout << start << endl;                                 //---8<--
		cout << length << endl;                                //---8<--
		cout.flush();                                          //---8<--
																   //---8<--
		string s;                                              //---8<--
		getline(cin, s);                                        //---8<--
		int ret = atoi(s.c_str());                             //---8<--
		return ret;                                            //---8<--
	}                                                          //---8<--
	string getMessage(int start, int length) {                  //---8<--
		cout << "?getMessage" << endl;                         //---8<--
		cout << start << endl;                                 //---8<--
		cout << length << endl;                                //---8<--
		cout.flush();                                          //---8<--
																   //---8<--
		string ret;                                            //---8<--
		getline(cin, ret);                                      //---8<--
		return ret;                                            //---8<--
	}                                                          //---8<--
}                                                              //---8<--
// ---8<--- end of library methods, not submitted to the website ---8<--
#endif

short editDistance(string s1, string s2) {
	short n = s1.size(), m = s2.size();
	vector<vector<short>> dp(n + 1, vector<short>(m + 1, 0));
	REP(i, n + 1) dp[i][0] = i;
	REP(j, m + 1) dp[0][j] = j;
	for (short i = 1; i <= n; i++) {
		for (short j = 1; j <= m; j++) {
			if (s1[i - 1] == s2[j - 1]) {
				dp[i][j] = dp[i - 1][j - 1];
			}
			else {
				dp[i][j] = 1 + min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
			}
		}
	}
	return dp[n][m];
}

// [0, 0.1) [0.1, 0.2) [0.2, 0.3) [0.3, 0.4)
vector<vector<int>> pi({
		{	1,	2,	4,	-1,	-1, -1	}, // [0, 2000]
		{	2,	16,	24,	-1,	-1, -1	},
		{	2,	16,	64,	-1,	-1, -1	},
		{	2,	16,	64,	-1,	-1, -1	},
		{	2,	16,	64,	-1,	-1, -1	},
		{	2,	16,	64,	-1,	-1, -1	}
	});
vector<vector<int>> pw({
		{	10,	15,	15,	-1,	-1,	-1	},
		{	20,	25,	30,	-1,	-1,	-1	},
		{	25,	30,	35,	-1,	-1,	-1	},
		{	30,	30,	35,	-1,	-1,	-1	},
		{	30,	30,	35,	-1,	-1,	-1	},
		{	30,	30,	35,	-1,	-1,	-1	}
	});
vector<vector<int>> pt({
		{	9,	14,	14,	-1,	-1,	-1	},
		{	18,	23,	28,	-1,	-1,	-1	},
		{	23,	28,	32,	-1,	-1,	-1	},
		{	28,	28,	32,	-1,	-1,	-1	},
		{	28,	28,	32,	-1,	-1,	-1	},
		{	28,	28,	32,	-1,	-1,	-1	}
	});
vector<int> ps({ 8, 32, 256, 256, 8, 8 });

// T have +-*/ and 0
template<class T> class BIT {
public:
	vector<T> dat;
	int N;

	BIT() {}
	BIT(int N) : N(N) { dat.assign(N, 0); }
	// sum [0,i)
	T sum(int i) {
		T ret = 0;
		for (--i; i >= 0; i = (i&(i + 1)) - 1) ret += dat[i];
		return ret;
	}
	// sum [i1,i2)
	T sum(int i1, int i2) { return sum(i2) - sum(i1); }
	T at(int i) { return sum(i, i + 1); }
	// add x to i
	void add(int i, T x) { for (; i < N; i |= i + 1) dat[i] += x; }
};

class MessageChecksum {
	string ans;
	BIT<int> checksum;
	int checkcnt;

	// [begin, end) の誤り訂正
	void errorcheck(int begin, int end, const int truecs) {
		int cs = checksum.sum(begin, end) % 26;
		if (cs != truecs) {
			if (end - begin == 1) {
				ans[begin] = char(truecs + 'A');
				checksum.add(begin, (truecs + 26 - cs) % 26);
				return;
			}

			int mid = (begin + end) / 2;
			int leftcs, rightcs;
			leftcs = Sender::getChecksum(begin, mid - begin);
			rightcs = (truecs + 26 - leftcs) % 26;
			errorcheck(begin, mid, leftcs);
			errorcheck(mid, end, rightcs);
		}
	}

	string solve_ver1(int n, string s) {
		for (int i = 0; i < n; i++) {
			int cs = Sender::getChecksum(i, 1);
			ans.push_back(cs + 'A');
		}
		return ans;
	}

	string solve_short(int n, string s) {
		ans += s.substr(100);

		checksum = BIT<int>(n);
		for (int i = 0; i < n; i++)
			checksum.add(i, ans[i] - 'A');

		int interval = ps[n / 2000];
		while (interval) {
			int d = n / interval / 3;
			vector<int> ns;
			for (int i = 0; i <= interval; i++) ns.push_back(n * i / interval);
			for (int i = 0; i < ns.size() - 1; i++) {
				int l = max(0, ns[i] - d), r = min(ns[i + 1] + d, n);
				errorcheck(l, r, Sender::getChecksum(l, r - l));
			}
			interval = interval * 2 / 5;
		}
		return ans;
	}

	string solve_ver4(int n, string s1) {
		if (n <= 100) return solve_ver1(n, s1);

		int L = 100;
		for (int i = 0; i < L; i++) {
			int c = Sender::getChecksum(i, 1);
			ans.push_back('A' + c);
		}

		int edit_dist = editDistance(s1.substr(0, L), ans);
		double pseudo_error = double(edit_dist) / L;

		if (pseudo_error < 0.05 && s1.size() == n && n < 6000) {
			return solve_short(n, s1);
		}

		int lp = n / 2000;
		int ep = min(int(pseudo_error * 10), 4);

		if (pi[lp][ep] == -1) {
			for (int i = L; i < n; i++) {
				int c = Sender::getChecksum(i, 1);
				ans.push_back(c + 'A');
			}
			return ans;
		}

		// 削除を含まない長さ n の文字列を得る　余分なコストはせいぜい数百
		string s2 = ans;
		for (int i = L; i < n; i++) {
			while (true) {
				string ch = Sender::getMessage(i, 1);
				if (!ch.empty()) {
					s2 += ch;
					break;
				}
			}
		}

		// s1 の適切な位置に空白を挿入して長さ n の文字列にする (要改良)
		int n1 = s1.size();
		int width = pw[lp][ep];
		int threshold = pt[lp][ep];
		for (int i = 0; i + width < s1.size() && s1.size() < n; i++) {
			int error_sum = 0;
			for (int j = i; j < i + width; j++)
				if (s1[j] != s2[j]) error_sum++;
			if (error_sum >= threshold) {
				s1 = s1.substr(0, i) + " " + s1.substr(i);
			}
		}
		while (s1.size() < n) {
			s1.push_back(' ');
		}

		if (pseudo_error < 0.25) {
			for (int i = L; i < n; i++) {
				if (s1[i] == s2[i]) {
					ans.push_back(s1[i]);
				}
				else {
					int c = Sender::getChecksum(i, 1);
					ans.push_back(c + 'A');
				}
			}
		}
		else {
			string s3 = ans;
			for (int i = L; i < n; i++) {
				while (true) {
					string ch = Sender::getMessage(i, 1);
					if (!ch.empty()) {
						s3 += ch;
						break;
					}
				}
			}

			for (int i = L; i < n; i++) {
				if (s1[i] == s2[i] && s2[i] == s3[i]) {
					ans.push_back(s1[i]);
				}
				else {
					int c = Sender::getChecksum(i, 1);
					ans.push_back(c + 'A');
				}
			}
		}

		// checksum による最終修正
		checksum = BIT<int>(n);
		for (int i = 0; i < n; i++)
			checksum.add(i, ans[i] - 'A');

		int interval = pi[lp][ep];
		while (interval) {
			int d = n / interval / 3;
			vector<int> ns;
			for (int i = 0; i <= interval; i++) ns.push_back(n * i / interval);
			for (int i = 0; i < ns.size() - 1; i++) {
				int l = max(0, ns[i] - d), r = min(ns[i + 1] + d, n);
				errorcheck(l, r, Sender::getChecksum(l, r - l));
			}
			interval = interval * 2 / 5;
		}
		
		return ans;
	}

public:
	string receiveMessage(int n, string s) {
		return solve_ver4(n, s);
	}
};

// -------8<------- end of solution submitted to the website -------8<-------


#ifdef LOCALTEST
int main() {
	double ratio = 0;
	unsigned seed = 0;

	SXor128 rnd;
	REP(i, 100) rnd.nextUInt();

	int numTest = 500, totaledit = 0;
	for (int seed = 0; seed < numTest; seed++) {
		Sender::vis = MessageChecksumVis();
		Sender::vis.generateTestCase(seed);
		MessageChecksum mc;
		int n = Sender::vis.message.size();
		string s = Sender::vis.getMessage(0, n);
		Sender::vis.cost = 0;
		string ans = mc.receiveMessage(n, s);
		ll base = Sender::vis.message.size() * 5 + 100;
		ll score = Sender::vis.score(ans);
		double r = double(base) / score;
		totaledit += Sender::vis.editDistance(Sender::vis.message, ans);
		ratio += r;
		bool exact = false;
		if (s.size() == n) exact = true;
		if(!exact) fprintf(stderr, "%3d: %f,\ttotaledit: %d,\tratio: %f\n", seed, r, totaledit, ratio / (seed + 1));
		else fprintf(stderr, "\t%3d: %f,\ttotaledit: %d,\tratio: %f\n", seed, r, totaledit, ratio / (seed + 1));
	}
	cerr << "totaledit: " << totaledit << endl;
	cerr << "ratio: " << ratio / numTest << endl;
	return 0;
}
#else
int main() {
	MessageChecksum mc;
	int length;
	string message, s;
	getline(cin, s);
	length = atoi(s.c_str());
	getline(cin, message);
	string ret = mc.receiveMessage(length, message);
	cout << ret << endl;
	cout.flush();
}
#endif