import copy

def factorial(n):
    ret = 1
    for i in range(1, n + 1):
        ret *= i
    return ret

def nCk(n, k):
    if n < k:
        return 0
    return factorial(n) // factorial(n - k) // factorial(k)
    
def nPk(n, k):
    if n < k:
        return 0
    ret = 1
    for i in range(n - k + 1, n + 1):
        ret *= i
    return ret

# 分割数列挙
def enumPartitions(n, max_n, v, vv):
    if n == 0:
        vv.append(copy.copy(v))
        return
    for i in range(min(n, max_n), 0, -1):
        v.append(i)
        enumPartitions(n - i, i, v, vv)
        v.pop(-1)

# 最も出現頻度の少ない数字が一種類に定まるかどうか
def isValidPartition(partition):
    if len(partition) == 1:
        return True
    return partition[-2] != partition[-1]

# 整数分割によって得られた valid な"個数の配列"に数字を割り当て、並べ替える場合の数
def singleLeastFrequentPatternCountForList(v, C):
    N = sum(v)
    ret = factorial(N)
    mp = {}
    for x in v:
        ret //= factorial(x)
        if x not in mp:
            mp[x] = 0
        mp[x] += 1
    ret *= nPk(C, len(v))
    for x in mp.values():
        ret //= factorial(x)
    return ret

# 最も出現頻度の少ない数字が一種類に定まる場合の数
def singleLeastFrequentPatternCountForAllPartitions(N, C):
    ret = 0
    vtmp = []
    vv = []
    enumPartitions(N, N, vtmp, vv)
    for v in vv:
        if isValidPartition(v):
            x = singleLeastFrequentPatternCountForList(v, C)
            ret += x
    return ret

# 最も出現頻度の少ない数字が一種類に定まり、かつそのうち一つをあるマスに固定したときの場合の数
def singleLeastFrequentPatternCountForAllPartitionsFixedFirstElement(N, C):
    ret = 0
    vtmp = []
    vv = []
    enumPartitions(N, N, vtmp, vv)
    for v in vv:
        if isValidPartition(v):
            x = nPk(C, len(v))
            v[-1] -= 1
            if v[-1] == 0:
                v.pop()
            x *= factorial(sum(v))
            mp = {}
            for i in v:
                x //= factorial(i)
                if i not in mp:
                    mp[i] = 0
                mp[i] += 1
            for i in mp.values():
                x //= factorial(i)
            ret += x
    return ret

def getProbabilities(N, C):
    x = singleLeastFrequentPatternCountForAllPartitions(N, C) // C
    y = singleLeastFrequentPatternCountForAllPartitionsFixedFirstElement(N, C)
    z = C ** (N - 1)
    
    p00 = z - x + y
    p01 = (C ** N - p00) // (C - 1)
    p10 = C ** N - x * (C - 1)
    p11 = x

    p00 /= C ** N
    p01 /= C ** N
    p10 /= C ** N
    p11 /= C ** N

    line = 'memo[{}][{}][{}]={};memo[{}][{}][{}]={};memo[{}][{}][{}]={};memo[{}][{}][{}]={};' \
            .format(N, C, 0, p00, N, C, 1, p01, N, C, 2, p10, N, C, 3, p11)

    print(line)


for N in range(2, 37):
    for C in range(2, 9):
        getProbabilities(N, C)