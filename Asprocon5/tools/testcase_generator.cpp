//testcase_generator.cpp Ver.15

#include <iostream>
#include <algorithm>
#include <bitset>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <cstring>
#include <utility>
#include <vector>
#include <complex>
#include <valarray>
#include <fstream>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <numeric>
#include <climits>
#include <random>
#include <chrono>
#include <list>

#define all(X) (X).begin(),(X).end()
#define len(X) ((int)(X).size())
#define fi first
#define sc second
using namespace std;
using ll = long long;
using ull = unsigned long long;
using Pii = pair<int, int>;
using Pll = pair<ll, ll>;

const string doc = 
	"Test Case Generator\n"
	"usage: ./a.out [OPTION]\n"
	"       各オプションを指定しなかった場合の初期値はソースコードを参照してください\n"
	"options:\n"
	"  --help ヘルプを読む\n"
	"  -M     設備数\n"
	"  -I     品目数\n"
	"  -R     オーダ数\n"
	"  -C     製造能力の最大値\n"
	"  -ST    段取り時間の最大値\n"
	"  -Q     オーダの製造数量の最大値\n"
	"  -E     納期の最大値\n"
	"  -Dmin  (納期-最早開始時間)の最小値\n"
	"  -Dmax  (納期-最早開始時間)の最大値\n"
	"  -A     納期遅れ許容時間の最大値\n"
	"  -PR    オーダの粗利金額の最大値\n"
	"  -SU    品目間の段取り時間の生成方法(0: ランダム、1: 対称的)\n"
	"  -S     資源種別番号数(1: 主資源のみ、2-: 副資源あり)\n"
	"  -X     前段取り時間使用フラグ(0: ランダム、1: 主資源も副資源も割り付ける、2: 主資源のみ、3: 副資源のみ)\n"
	"  -Y     製造時間使用フラグ(0: ランダム、1: 主資源も副資源も割り付ける、2: 主資源のみ)\n"
	"example:\n"
	"  ./a.out -M 10 -I 5 -R 10 -C 100 -ST 1000 -Q 1 -E 259200 -Dmin 1 -Dmax 86400 -A 86400 -PR 20000 -SU 0 -S 1 -X 0 -Y 0";

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
};

struct Bom {
	int i = 0;
	int pr = 0;
	int p = 0;
	vector<tuple<int, int, int, int, int>> smcxy;
	double mean_s = 0.;
};

struct Order {
	int r = 0;        
	int i = 0;        
	int e = 0;        
	int d = 0;        
	int q = 0;        
	int pr = 0;       
	int a = 0;        
};


const int SECOND_A_DAY = 86400;

int M = 10;
int I = 5;
int R = 10;
int C = 100;
int ST = 1000;
int Q = 1;
int E = 3 * SECOND_A_DAY;
int Dmin = 1;
int Dmax = SECOND_A_DAY;
int A = SECOND_A_DAY;
int PR = 20000;
int SU = 0;
int S = 3;
int X = 0;
int Y = 0;

int MM = 1;

double manuTimeToPeriodRatio = 2.0;

random_device rnd;
default_random_engine engine(rnd());

void init_rand(int seed) {
	Xorshift r;
	r.x = seed;
	for(int i = 0; i < 1000; i++) r.nextUInt();
	engine = default_random_engine(r.nextUInt() >> 2);
}

int rand(int mi, int ma) {
	uniform_int_distribution<uint32_t> dist(mi, ma);
	int t = dist(engine);
	return t;	
}

void check_parameters(int param, int mi, int ma, string s) {
	if (param >= mi && param <= ma) {
		return;
	}
	cerr << "error:" << s << " must be equal to or larger than " << mi << "and be equal to or less than" << ma << endl;
	exit(1);
}

bool setParameters(vector<string>& params) {
	for (int i = 0; i < len(params); ++i) {
		if (params[i] == "--help") {
			cout << doc << endl;
			return false;
		}
	}
	if (len(params) % 2 != 0) {
		cerr << "error: invalid option" << endl;
		return false;
	}
	for (int i = 0; i < len(params); i += 2) {
		if (params[i][0] != '-') {
			cerr << "error: invalid option " << params[i] << endl;
			return false;
		}

		int t = 0;
		try {
			t = stoi(params[i + 1]);
		}
		catch (...) {
			cerr << "error: invalid option " << params[i + 1] << endl;
			return false;
		}

		string p = params[i].substr(1);

		auto ma = numeric_limits<int>::max(), mi = 1;
		if (p == "M") {
			M = t;
			ma = 100;
		}
		else if (p == "I") {
			I = t;
			ma = 100;
		}
		else if (p == "R") {
			R = t;
			ma = 10'000;
		}
		else if (p == "C") {
			C = t;
			ma = 1'000;
		}
		else if (p == "ST") {
			ST = t;
			ma = 100'000;
		}
		else if (p == "Q") {
			Q = t;
			ma = 200;
		}
		else if (p == "E") {
			E = t;
			ma = 1'000'000;
		}
		else if (p == "Dmin") {
			Dmin = t;
			ma = 1'000'000;
		}
		else if (p == "Dmax") {
			Dmax = t;
			ma = 1'000'000;
		}
		else if (p == "A") {
			A = t;
			ma = 100'000;
		}
		else if (p == "PR") {
			PR = t;
			ma = 100'000;
		}
		else if (p == "SU") {
			SU = t;
			mi = 0;
			ma = 1;
		}
		else if (p == "S") {
			S = t;
			ma = 10;
		}
		else if (p == "X") {
			X = t;
			mi = 0;
			ma = 3;
		}
		else if (p == "Y") {
			Y = t;
			mi = 0;
			ma = 2;
		}
		else if(p == "Seed") {
			init_rand(t);
		}
		else {
			cerr << "error: invalid option " << params[i] << endl;
			return false;
		}
		check_parameters(t, mi, ma, p);
	}
	if (M < S) {
		cerr << "error: M must be larger than or equal to S." << endl;
		return false;
	}
	if (Dmax < Dmin) {
		cerr << "error: Dmin must be less than or equal to Dmax." << endl;
		return false;
	}
	if (E < Dmax) {
		cerr << "error: E must be larger than or equal to Dmax." << endl;
		return false;
	}
	if (X == 2 && Y == 2 && S != 1) {
		cerr << "error: if X == Y == 2, S must be equal to 1 because of no sub resourse." << endl;
		return false;
	}

	return true;
}

void writeTestCase(const vector<int>& item, const vector<Bom>& boms, const vector<map<Pii, int>>& setupTimes, const vector<Order>& orders) {
	vector<string> bomlines;
	vector<string> combilines;
	for (int i = 0; i < len(boms); ++i) {
		const Bom& b = boms[i];
		for (int j = 0; j < b.p; ++j) {
			bomlines.push_back("BOM\t" + to_string(b.i) + '\t' + to_string(get<0>(b.smcxy[j])) + '\t' + to_string(get<1>(b.smcxy[j])) + '\t' + to_string(get<2>(b.smcxy[j])) + '\t' + to_string(get<3>(b.smcxy[j])) + '\t' + to_string(get<4>(b.smcxy[j])));
		}
	}
	for (int i = 0; i < len(setupTimes); ++i) {
		for (const auto& p : setupTimes[i]) {
			combilines.push_back("COMBI\t" + to_string(i) + '\t' + to_string(p.fi.fi) + '\t' + to_string(p.fi.sc) + '\t' + to_string(p.sc));
		}
	}
	cout << "HEADER\t" << I << '\t' << M << '\t' << MM << '\t' << len(bomlines) << '\t' << len(combilines) << '\t' << R << endl;
	for (int i = 0; i < len(item); i++) {
		cout << "ITEM\t" << i << '\t' << item[i] + 1 << endl;
	}
	for (int i = 0; i < len(bomlines); ++i) {
		cout << bomlines[i] << endl;
	}
	for (int i = 0; i < len(combilines); ++i) {
		cout << combilines[i] << endl;
	}
	for (int i = 0; i < len(orders); ++i) {
		const Order& o = orders[i];
		cout << "ORDER\t" << o.r << '\t' << o.i << '\t' << o.e << '\t' << o.d << '\t' << o.q << '\t' << o.pr << '\t' << o.a << endl;
	}
}

void main_material_num() {
	MM = 1;
	for (int i = 0; i < M - S; i++) {
		MM += rand(0, S - 1) == 0 ? 1 : 0;
	}
}

int main(int argc, char const* argv[]) {

	vector<string> params;
	for (int i = 1; i < argc; ++i) {
		params.push_back(argv[i]);
	}

	if (!setParameters(params)) return 1;

	main_material_num();

	vector<Bom> boms(I);
	vector<int> item(I, 1);
	vector<bool> flag(I, true);
	vector<bool> used(M, false);
	vector<vector<Pii>> Sxy(I);
	vector<int> Mx(I);
	for (int i = 0; i < I; ++i) {
		Bom& b = boms[i];
		b.i = i;
		b.pr = rand(1, max(PR / Q, 1));
		b.p = 0;
		
		auto mx = 1;
		if (X == 0) {
			mx = rand(0, 1);
		}
		else if (X == 3) {
			mx = 0;
		}
		if (mx == 1) {
			flag[i] = false;
		}
		Mx[i] = mx;
		for (int j = 0; j < MM; ++j) {
			
			if (rand(0, 1) == 1) {
				used[j] = true;
				b.smcxy.push_back({0, j, rand(1, C), mx, 1});
				b.p++;
			}
		}
		if (b.p == 0) {
			int j = rand(0, MM - 1);
			used[j] = true;
			b.smcxy.push_back({0, j, rand(1, C), mx, 1});
			b.p++;
		}
		for (int j = 0; j < b.p; ++j) {
			b.mean_s += get<2>(b.smcxy[j]);
		}
		b.mean_s /= b.p;

		if (X != 2 || Y != 2) {	
			vector<Pii> s_xy(S);
			for (auto j = 1; j < S; j++) {
				auto sx = 1, sy = 0;
				if (X == 0) {
					sx = rand(0, 1);
				}
				else if (X == 2) {
					sx = 0;
				}

				if (sx == 0 || Y == 1) {
					sy = 1;
				}
				else if (Y == 0) {
					sy = rand(0, 1);
				}

				
				s_xy[j].first = sx;
				s_xy[j].second = sy;
			}
			for (int j = MM; j < M; j++) {
				if (rand(0, 1) == 0) {
					auto s = rand(1, item[i]);
					if (s == item[i]) {
						item[i] = min(item[i] + 1, S - 1);
					}
					if (s_xy[s].first == 1) {
						flag[i] = false;
					}
					b.smcxy.push_back({s, j, -1, s_xy[s].first, s_xy[s].second});
					b.p++;
					used[j] = true;
				}
			}
			Sxy[i] = move(s_xy);
		}
	}

	for (auto i = 0; i < M; i++) {
		if (!used[i]) {	
			auto it = rand(0, I - 1);
			if (i < MM) {
				boms[it].smcxy.push_back({0, i, rand(1, C), Mx[it], 1});
				boms[it].mean_s = (boms[it].mean_s * boms[it].p + get<2>(boms[it].smcxy.back())) / (boms[it].p + 1);
				boms[it].p++;
			}
			else {
				boms[it].smcxy.push_back({1, i, -1, Sxy[it][1].first, Sxy[it][1].second});
				boms[it].p++;
			}
		}
	}
	
	vector<vector<int>> mToItems(M);
	for (int i = 0; i < len(boms); ++i) {
		Bom& b = boms[i];
		for (int j = 0; j < b.p; ++j) {
			mToItems[get<1>(b.smcxy[j])].push_back(b.i);
		}
	}

	for (auto& b : boms) {
		sort(all(b.smcxy), [](const auto& x, const auto& y) {
				return get<0>(x) < get<0>(y) || (get<0>(x) == get<0>(y) && get<1>(x) < get<1>(y));
			});
		item[b.i] = get<0>(b.smcxy.back());
	}

	vector<map<Pii, int>> setupTimes(MM);
	if (SU == 0) {
		for (int i = 0; i < MM; ++i) {
			map<Pii, int>& pairToTime = setupTimes[i];
			vector<int>& items = mToItems[i];
			for (int j = 0; j < len(items); ++j) {
				for (int k = 0; k < len(items); ++k) {
					if (flag[items[k]] || items[j] == items[k]) pairToTime[Pii(items[j], items[k])] = 0;
					else pairToTime[Pii(items[j], items[k])] = rand(1, ST);
				}
			}
		}
	}
	else {
		for (int i = 0; i < MM; ++i) {
			map<Pii, int>& pairToTime = setupTimes[i];
			vector<int>& items = mToItems[i];
			for (int j = 0; j < len(items); ++j) {
				for (int k = 0; k < len(items); ++k) {
					if (flag[items[k]] || items[j] == items[k]) pairToTime[Pii(items[j], items[k])] = 0;
					else if (j < k) pairToTime[Pii(items[j], items[k])] = rand(1, ST);
					else pairToTime[Pii(items[j], items[k])] = pairToTime[Pii(items[k], items[j])];
				}
			}
		}
	}

	vector<Order> orders(R);
	for (int i = 0; i < R; ++i) {
		Order& o = orders[i];
		o.r = i;
		o.i = rand(0, I - 1);
		o.q = rand(1, Q);
		o.pr = o.q * boms[o.i].pr;
		o.a = rand(1, A);

		int meanManuTime = static_cast<int>(o.q * boms[o.i].mean_s);
		int _Dmin = max(Dmin, int(meanManuTime * manuTimeToPeriodRatio));
		int period;
		if (_Dmin > Dmax) {
			period = Dmax;
		}
		else {
			period = rand(_Dmin, Dmax);
		}

		o.e = rand(0, E - period);
		o.d = o.e + period;
	}

	writeTestCase(item, boms, setupTimes, orders);

	return 0;
}
