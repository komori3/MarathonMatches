// version 5.0

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <utility>
#include <numeric>
#include <algorithm>

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
        vector<int> m_list;
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
    int main_resource = 0;
    vector<Pii> sub_resources;

    Operation(int _r, int _i, int _t1, int _t2, int _t3) :
            r(_r), i(_i), t1(_t1), t2(_t2), t3(_t3) {}
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


int main(int argc, const char *argv[]) {
    try {
        ifstream in(argv[1]);
        ifstream out(argv[2]);

        string _str;
        in >> _str >> I >> M >> MM >> BL >> CL >> R;
        MS = M - MM;
        boms.resize(I);
        for (int n = 0; n < I; ++n) {
            int i, ss;
            in >> _str >> i >> ss;
            boms[i].ss = ss;
            boms[i].s_to_sub_resources.resize(ss - 1);
        }
        for (int n = 0; n < BL; ++n) {
            int i, s, m, c, x, y;
            in >> _str >> i >> s >> m >> c >> x >> y;
            Bom &bom = boms[i];
            if (s == 0) {
                bom.main_resources.need_setup = x == 1;
                bom.main_resources.m_to_c[m] = c;
            } else {
                Bom::SubResources &subResource = bom.s_to_sub_resources[s - 1];
                subResource.for_setup = x == 1;
                subResource.for_manufacturing = y == 1;
                subResource.m_list.push_back(m - MM);
            }
        }

        for (Bom &bom : boms) {
            if (bom.main_resources.need_setup) ++bom.setup_res_count;
            for (Bom::SubResources &sr : bom.s_to_sub_resources) {
                sort(all(sr.m_list));
                if (sr.for_setup) ++bom.setup_res_count;
            }
        }

        setup_times.resize(MM);
        for (int n = 0; n < CL; ++n) {
            int m, i_pre, i_next, t;
            in >> _str >> m >> i_pre >> i_next >> t;
            setup_times[m][Pii(i_pre, i_next)] = t;
        }

        for (int n = 0; n < R; ++n) {
            int r, i, e, d, q, pr, a;
            in >> _str >> r >> i >> e >> d >> q >> pr >> a;
            orders.emplace_back(r, i, e, d, q, pr, a);
        }
        sort(all(orders), [](const Order &o1, const Order &o2) { return o1.r < o2.r; });

        for (int n = 0; n < R; ++n) {
            int r, t1, t2, t3;
            out >> r >> t1 >> t2 >> t3;
            if (out.eof()) {
                cerr << "出力形式違反: Output is too small" << endl;
                return 1;
            }
            if (r < 0 || r >= R) {
                cerr << "オーダ番号間違い: r is out of range" << endl;
                return 1;
            }
            if (t1 < 0 || t1 > MAX_TIME) {
                cerr << "段取り開始時刻間違い: t1 is out of range" << endl;
                return 1;
            }
            if (t2 < 0 || t2 > MAX_TIME) {
                cerr << "製造開始時刻間違い: t2 is out of range" << endl;
                return 1;
            }
            if (t3 <= 0 || t3 > MAX_TIME) {
                cerr << "製造終了時刻間違い: t3 is out of range" << endl;
                return 1;
            }
            if (t1 > t2) {
                cerr << "製造開始時刻違反: Invalid manufacturing start time" << endl;
                return 1;
            }
            if (t2 > t3) {
                cerr << "製造終了時刻違反: Invalid manufacturing end time" << endl;
                return 1;
            }

            Order &order = orders[r];
            Bom &bom = boms[order.i];
            Operation ope = Operation(r, order.i, t1, t2, t3);

            for (int s = 0; s < bom.ss; ++s) {
                int m;
                out >> m;
                if (out.eof()) {
                    cerr << "出力形式違反: Output is too small" << endl;
                    return 1;
                }
                if (m < -1 || m >= M) {
                    cerr << "資源番号間違い: m is out of range" << endl;
                    return 1;
                }
                if (s == 0) {
                    ope.main_resource = m;
                } else {
                    int ms = m < 0 ? m : m - MM;
                    ope.sub_resources.emplace_back(s - 1, ms);
                }
            }
            operations.push_back(ope);
        }

        sort(all(operations), [](const Operation &o1, const Operation &o2) { return o1.r < o2.r; });
        for (int r = 1; r < R; ++r) {
            Operation &o1 = operations[r - 1];
            Operation &o2 = operations[r];
            if (o1.r == o2.r) {
                cerr << "オーダ番号重複: Duplicated order number" << endl;
                return 1;
            }
            if ((o2.r - o1.r) > 1) {
                cerr << "オーダ番号飛ばし: Skipped order number" << endl;
                return 1;
            }
        }

        // 主資源について、割り付けが正しく行われているかチェック。
        vector<vector<int>> mToOrderIds(MM);
        for (int r = 0; r < R; ++r) {
            Operation &ope = operations[r];
            Bom &bom = boms[ope.i];
            int m = ope.main_resource;
            if (bom.main_resources.m_to_c.count(m) == 0) {
                cerr << "BOM違反: Bom violation" << endl;
                return 1;
            }
            mToOrderIds[m].push_back(ope.r);
        }
        for (int m = 0; m < MM; ++m) {
            sort(all(mToOrderIds[m]), [](const int r1, const int r2) {
                Operation &ope1 = operations[r1];
                Bom::MainResources &mr1 = boms[ope1.i].main_resources;
                Operation &ope2 = operations[r2];
                Bom::MainResources &mr2 = boms[ope2.i].main_resources;
                int startT1 = mr1.need_setup ? ope1.t1 : ope1.t2;
                int startT2 = mr2.need_setup ? ope2.t1 : ope2.t2;
                return startT1 < startT2;
            });
            int previousI = -1;
            int previousT3 = 0;
            for (int r : mToOrderIds[m]) {
                Operation &ope = operations[r];
                Order &order = orders[r];
                Bom &bom = boms[order.i];
                Bom::MainResources &mainResources = bom.main_resources;
                int startT = mainResources.need_setup ? ope.t1 : ope.t2;
                if (ope.t1 < order.e) {
                    cerr << "段取り開始時刻違反: Invalid setup start time" << endl;
                    return 1;
                }
                if (startT < previousT3) {
                    if (mainResources.need_setup) {
                        cerr << "段取り開始時刻違反: Invalid setup start time" << endl;
                    } else {
                        cerr << "製造開始時刻違反: Invalid manufacturing start time" << endl;
                    }
                    return 1;
                }
                int setupT = 0;
                if (previousI >= 0) {
                    setupT = setup_times[m][Pii(previousI, order.i)];
                }

                if ((ope.t2 - ope.t1) != setupT) {
                    cerr << "段取り時間間違い: Invalid setup time" << endl;
                    return 1;
                }
                ope.need_setup = setupT != 0;
                int manuT = order.q * mainResources.m_to_c[m];
                if ((ope.t3 - ope.t2) != manuT) {
                    cerr << "製造時間間違い: Invalid manufacturing time " << endl;
                    return 1;
                }
                previousI = order.i;
                previousT3 = ope.t3;
            }
        }

        // 副資源について、割り付けが正しく行われているかチェック。
        vector<vector<Pii>> mToOrderIdAndS(MS);
        for (int r = 0; r < R; ++r) {
            Operation &ope = operations[r];
            Bom &bom = boms[ope.i];
            for (Pii &pii : ope.sub_resources) {
                int s = pii.first;
                int m = pii.second;
                Bom::SubResources &subResources = bom.s_to_sub_resources[s];
                if (ope.need_setup || subResources.for_manufacturing) {
                    if (!binary_search(all(subResources.m_list), m)) {
                        cerr << "BOM違反: Bom violation" << endl;
                        return 1;
                    }
                    mToOrderIdAndS[m].push_back(Pii(r, s));
                } else {
                    if (m != -1) {
                        cerr << "BOM違反: Bom violation" << endl;
                        return 1;
                    }
                }
            }
        }
        for (int m = 0; m < MS; ++m) {
            sort(all(mToOrderIdAndS[m]), [](const Pii &p1, const Pii &p2) {
                Operation &ope1 = operations[p1.first];
                Bom::SubResources &sr1 = boms[ope1.i].s_to_sub_resources[p1.second];
                Operation &ope2 = operations[p2.first];
                Bom::SubResources &sr2 = boms[ope2.i].s_to_sub_resources[p2.second];
                int startT1 = sr1.for_setup ? ope1.t1 : ope1.t2;
                int startT2 = sr2.for_setup ? ope2.t1 : ope2.t2;
                return startT1 < startT2;
            });
            int previousT3 = 0;
            for (Pii &pii : mToOrderIdAndS[m]) {
                Operation &ope = operations[pii.first];
                Bom::SubResources &subResources = boms[ope.i].s_to_sub_resources[pii.second];
                int startT = subResources.for_setup ? ope.t1 : ope.t2;
                int endT = subResources.for_manufacturing ? ope.t3 : ope.t2;
                if (startT < previousT3) {
                    if (subResources.for_setup) {
                        cerr << "段取り開始時刻違反: Invalid setup start time" << endl;
                    } else {
                        cerr << "製造開始時刻違反: Invalid manufacturing start time" << endl;
                    }
                    return 1;
                }
                previousT3 = endT;
            }
        }

        // 評価
        ll V1 = 0, V2 = 0;
        for (int n = 0; n < R; ++n) {
            Operation &ope = operations[n];
            Order &order = orders[ope.r];
            Bom &bom = boms[order.i];
            ll setupT = ope.t2 - ope.t1;
            V1 += setupT * bom.setup_res_count;
            if (ope.t3 <= order.d) {
                V2 += order.pr;
            } else {
                ll delay = ope.t3 - order.d;
                V2 += order.pr - (order.pr * delay + order.a - 1) / order.a;
            }
        }

        ll P = P_MAX - V1 + V2;

        printf("%lld\n", max(0ll, P));
        return 0;
    } catch (char *str) {
        cerr << "error: " << str << endl;
        return 1;
    }
}
