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

#define all(X) (X).begin(),(X).end()
using namespace std;
typedef long long ll;
typedef pair<int, int> Pii;

struct Order {
    int r;
    int i;
    int e;
    int d;
    int q;
    ll pr;
    int a;

    Order(int _r, int _i, int _e, int _d, int _q, int _pr, int _a) :
            r(_r), i(_i), e(_e), d(_d), q(_q), pr(_pr), a(_a) {}
};

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
    vector<SubResources> s_to_sub_resources;
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


int I;
int M;
int MM;
int MS;
int BL;
int CL;
int R;
const int MAX_TIME = 86400000;
const ll P_MAX = 1e10;
vector<Order> orders;
vector<Bom> boms;
vector<Operation> operations;
vector<map<Pii, int>> setup_times;

void read_problem() {
    string _str;
    cin >> _str >> I >> M >> MM >> BL >> CL >> R;
    MS = M - MM;
    boms.resize(I);
    for (int n = 0; n < I; ++n) {
        int i, ss;
        cin >> _str >> i >> ss;
        boms[i].ss = --ss;
        boms[i].s_to_sub_resources.resize(ss);
    }
    for (int n = 0; n < BL; ++n) {
        int i, s, m, c, x, y;
        cin >> _str >> i >> s >> m >> c >> x >> y;
        Bom &bom = boms[i];
        if (s == 0) {
            bom.main_resources.need_setup = x == 1;
            bom.main_resources.m_to_c[m] = c;
        } else {
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

    setup_times.resize(MM);
    for (int n = 0; n < CL; ++n) {
        int m, i_pre, i_next, t;
        cin >> _str >> m >> i_pre >> i_next >> t;
        setup_times[m][Pii(i_pre, i_next)] = t;
    }

    for (int n = 0; n < R; ++n) {
        int r, i, e, d, q, pr, a;
        cin >> _str >> r >> i >> e >> d >> q >> pr >> a;
        orders.emplace_back(r, i, e, d, q, pr, a);
    }
    sort(all(orders), [](const Order &o1, const Order &o2) { return o1.r < o2.r; });
}

void write_solution() {
    for (Operation &ope : operations) {
        cout << ope.r << '\t' << ope.t1 << '\t' << ope.t2 << '\t' << ope.t3 << '\t' << ope.main_m;
        for (int ms : ope.sub_ms_list) {
            int m = ms < 0 ? -1 : ms + MM;
            cout << '\t' << m;
        }
        cout << endl;
    }
}

void solve() {
    operations.resize(R);
    for (int r = 0; r < R; ++r) {
        Order &order = orders[r];
        Operation &ope = operations[r];
        Bom &bom = boms[order.i];
        ope.r = r;
        ope.i = order.i;
        ope.main_m = bom.main_resources.m_to_c.begin()->first;
        for (int s = 0; s < bom.ss; ++s) {
            int ms = bom.s_to_sub_resources[s].ms_list[0];
            ope.sub_ms_list.push_back(ms);
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
        int mainM = ope.main_m;
        Bom::MainResources &mr = bom.main_resources;
        int previousI = mToPreviousI[mainM];
        int setupT = 0;
        if (previousI >= 0) {
            setupT = setup_times[mainM][Pii(previousI, ope.i)];
        }
        ope.need_setup = setupT > 0;
        int startT = mr.need_setup ? mainMToPreviousT3[mainM] : mainMToPreviousT3[mainM] - setupT;
        int e = max(order.e, startT);
        for (int s = 0; s < bom.ss; ++s) {
            int subMs = ope.sub_ms_list[s];
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

int main() {
    read_problem();
    solve();
    write_solution();
}
