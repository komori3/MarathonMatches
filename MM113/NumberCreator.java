import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Set;

public class NumberCreator {

  public String[] findSolution(int Num0, int Num1, String T) {
    TestCase tc = new TestCase(Num0, Num1, T);

    String[] bsol, dsol;

    {
      BState firstState = new BState(tc);
      int maxTurn = firstState.runLengthBlocks.size();
      ArrayList<PriorityQueue<BState>> states = new ArrayList<>();
      for (int i = 0; i <= maxTurn; i++)
        states.add(new PriorityQueue<>(new BCmp()));

      states.get(0).add(firstState);

      int width = 1000;
      for (int t = 0; t < maxTurn; t++) {
        for (int w = 0; w < width; w++) {
          if (states.get(t).isEmpty())
            break;
          BState nowState = states.get(t).poll();
          for (BState nextState : nowState.getAllNextStates()) {
            int nt = nextState.ptrRLB;
            states.get(nt).add(nextState);
          }
        }
        states.get(t).clear();
      }

      bsol = states.get(maxTurn).poll().getAns();
    }

    {
      DState firstState = new DState(tc);

      firstState.preProcess();

      DState nowState = firstState.clone();
      while (!nowState.isCompleted()) {
        nowState = nowState.getBestNextState();
      }

      nowState.postProcess();

      dsol = nowState.getAns();
    }

    return bsol.length < dsol.length ? bsol : dsol;
  }

  public static void main(String[] args) {
    try {
      BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

      int Num0 = Integer.parseInt(br.readLine());
      int Num1 = Integer.parseInt(br.readLine());
      String T = br.readLine();

      NumberCreator nc = new NumberCreator();
      String[] ret = nc.findSolution(Num0, Num1, T);

      System.out.println(ret.length);
      for (int i = 0; i < ret.length; i++)
        System.out.println(ret[i]);
    } catch (Exception e) {
    }
  }
}

class TestCase {
  final int num0;
  final int num1;
  final String target;

  TestCase(int num0, int num1, String target) {
    this.num0 = num0;
    this.num1 = num1;
    this.target = target;
  }
}

class ScoreTable {
  final List<String> smallTargetStringList;
  final List<Integer> smallTargetList;
  final Set<Integer> smallTargetSet;

  ScoreTable(TestCase tc) {
    smallTargetStringList = new ArrayList<>();
    smallTargetList = new ArrayList<>();
    smallTargetSet = new HashSet<>();

    String target = tc.target;
    for (int p = 0; p < target.length(); p += 4) {
      int end = Math.min(p + 4, target.length());
      String sub = target.substring(p, end);
      smallTargetStringList.add(sub);
      int isub = Integer.parseInt(sub);
      smallTargetList.add(isub);
      if (isub != 0)
        smallTargetSet.add(Integer.parseInt(sub));
    }
  }
}

class DCmp implements Comparator<DState> {
  @Override
  public int compare(DState s0, DState s1) {
    if (s0.match < s1.match) {
      return 1;
    } else if (s0.match > s1.match) {
      return -1;
    } else {
      return 0;
    }
  }
}

class DState implements Cloneable {
  private static final int THRESH = 20000;

  ScoreTable table;
  int match;

  Map<Integer, Integer> numMap;
  List<Integer> cmds;

  DState(TestCase tc) {
    table = new ScoreTable(tc);
    match = 0;
    numMap = new HashMap<>();
    push(tc.num0);
    push(tc.num1);
    cmds = new ArrayList<>();
  }

  boolean isCompleted() {
    return match == table.smallTargetSet.size();
  }

  int push(int num) {
    int id = numMap.size();
    numMap.put(id, num);
    if (table.smallTargetSet.contains(num)) {
      match++;
    }
    return id;
  }

  @Override
  public DState clone() {
    DState s = null;
    try {
      s = (DState) super.clone();
      s.numMap = new HashMap<>(this.numMap);
      s.cmds = new ArrayList<>(this.cmds);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return s;
  }

  int eval() {
    int e = 0;
    Set<Integer> cand = new HashSet<>();
    Set<Integer> target = new HashSet<>(table.smallTargetSet);
    List<Integer> numList = new ArrayList<>(numMap.values());
    for (int id0 = 0; id0 < numList.size(); id0++) {
      int n0 = numList.get(id0);
      if (target.contains(n0))
        target.remove(n0);
      for (int id1 = id0; id1 < numList.size(); id1++) {
        int n1 = numList.get(id1);
        if (n0 + n1 < THRESH)
          cand.add(n0 + n1);
        cand.add(Math.abs(n0 - n1));
        if (n0 * n1 < THRESH)
          cand.add(n0 * n0);
        if (n0 * n1 < THRESH)
          cand.add(n0 * n1);
        if (n0 * n1 < THRESH)
          cand.add(n1 * n1);
        cand.add(Math.max(n0, n1) / Math.min(n0, n1));
      }
    }
    for (int c : cand) {
      if (target.contains(c))
        e += 100;
      if (target.contains(c >> 1))
        e++;
      if (target.contains(c << 1))
        e++;
      if (target.contains(c - 1))
        e++;
      if (target.contains(c + 1))
        e++;
      if (target.contains(c >> 2))
        e++;
      if (target.contains(c << 2))
        e++;
      if (target.contains(c - 2))
        e++;
      if (target.contains(c + 2))
        e++;
    }
    return match * 10000 + e;
  }

  DState getBestNextState() {
    DState bestState = null;
    int bestScore = Integer.MIN_VALUE;

    //List<Integer> numList = new ArrayList<>(numMap.values());
    for (int id0 = 0; id0 < numMap.size(); id0++) {
      for (int id1 = id0; id1 < numMap.size(); id1++) {
        int n0 = numMap.get(id0), n1 = numMap.get(id1);
        int n;

        n = n0 + n1;
        if (n < THRESH && !numMap.containsValue(n)) {
          operate(id0, '+', id1);
          int e = eval();
          if (bestScore < e) {
            bestScore = e;
            bestState = clone();
          }
          undo();
        }

        n = Math.abs(n0 - n1);
        if (n != 0 && !numMap.containsValue(n)) {
          if (n0 <= n1)
            operate(id1, '-', id0);
          else
            operate(id0, '-', id1);
          int e = eval();
          if (bestScore < e) {
            bestScore = e;
            bestState = clone();
          }
          undo();
        }

        n = n0 * n0;
        if (n < THRESH && !numMap.containsValue(n)) {
          operate(id0, '*', id0);
          int e = eval();
          if (bestScore < e) {
            bestScore = e;
            bestState = clone();
          }
          undo();
        }

        n = n0 * n1;
        if (n < THRESH && !numMap.containsValue(n)) {
          operate(id0, '*', id1);
          int e = eval();
          if (bestScore < e) {
            bestScore = e;
            bestState = clone();
          }
          undo();
        }

        n = n1 * n1;
        if (n < THRESH && !numMap.containsValue(n)) {
          operate(id1, '*', id1);
          int e = eval();
          if (bestScore < e) {
            bestScore = e;
            bestState = clone();
          }
          undo();
        }

        n = Math.max(n0, n1) / Math.min(n0, n1);
        if (n != 0 && !numMap.containsValue(n)) {
          if (n0 <= n1)
            operate(id1, '/', id0);
          else
            operate(id0, '/', id1);
          int e = eval();
          if (bestScore < e) {
            bestScore = e;
            bestState = clone();
          }
          undo();
        }
      }
    }
    return bestState;
  }

  List<DState> getAllNextStates() {
    List<DState> states = new ArrayList<>();
    if (isCompleted())
      return states;
    List<Integer> numList = new ArrayList<>(numMap.values());
    for (int id0 = 0; id0 < numList.size(); id0++) {
      for (int id1 = id0; id1 < numList.size(); id1++) {
        int n0 = numList.get(id0), n1 = numList.get(id1);
        int n;

        n = n0 + n1;
        if (n < THRESH && !numMap.containsValue(n)) {
          operate(id0, '+', id1);
          states.add(clone());
          undo();
        }

        n = Math.abs(n0 - n1);
        if (n != 0 && !numMap.containsValue(n)) {
          if (n0 <= n1)
            operate(id1, '-', id0);
          else
            operate(id0, '-', id1);
          states.add(clone());
          undo();
        }

        n = n0 * n0;
        if (n < THRESH && !numMap.containsValue(n)) {
          operate(id0, '*', id0);
          states.add(clone());
          undo();
        }

        n = n0 * n1;
        if (n < THRESH && !numMap.containsValue(n)) {
          operate(id0, '*', id1);
          states.add(clone());
          undo();
        }

        n = n1 * n1;
        if (n < THRESH && !numMap.containsValue(n)) {
          operate(id1, '*', id1);
          states.add(clone());
          undo();
        }

        n = Math.max(n0, n1) / Math.min(n0, n1);
        if (n != 0 && !numMap.containsValue(n)) {
          if (n0 <= n1)
            operate(id1, '/', id0);
          else
            operate(id0, '/', id1);
          states.add(clone());
          undo();
        }
      }
    }
    return states;
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
    for (Entry<Integer, Integer> entry : numMap.entrySet()) {
      if (num == entry.getValue())
        return entry.getKey();
    }
    return -1;
  }

  void postProcess() {
    int e4 = findId(10000);

    int pos = findId(table.smallTargetList.get(0));
    for (int i = 1; i < table.smallTargetList.size() - 1; i++) {
      operate(pos, '*', e4);
      pos = numMap.size() - 1;
      if (table.smallTargetList.get(i) != 0) {
        int tid = findId(table.smallTargetList.get(i));
        operate(pos, '+', tid);
        pos = numMap.size() - 1;
      }
    }

    if (table.smallTargetList.size() > 1) {
      int i = table.smallTargetList.size() - 1;
      int len = table.smallTargetStringList.get(i).length();
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
      if (table.smallTargetList.get(i) != 0) {
        int tid = findId(table.smallTargetList.get(i));
        operate(pos, '+', tid);
      }
    }
  }

  int packCmd(int id1, char op, int id2) {
    return (id1 << 20) + (op << 10) + id2;
  }

  String getCmdString(int pack) {
    int id1 = pack >> 20;
    char op = (char) ((pack >> 10) & 0b1111111111);
    int id2 = pack & 0b1111111111;
    return id1 + " " + op + " " + id2;
  }

  void operate(int id0, char op, int id1) {
    int n0 = numMap.get(id0);
    int n1 = numMap.get(id1);
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
    default:
      throw new IllegalArgumentException();
    }
    push((int) n);
    cmds.add(packCmd(id0, op, id1));
  }

  void undo() {
    int i = numMap.size() - 1;
    int num = numMap.get(i);
    numMap.remove(i);
    cmds.remove(cmds.size() - 1);

    if (table.smallTargetSet.contains(num)) {
      match--;
    }
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + ((cmds == null) ? 0 : cmds.hashCode());
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    DState other = (DState) obj;
    if (cmds == null) {
      if (other.cmds != null)
        return false;
    } else if (!cmds.equals(other.cmds))
      return false;
    return true;
  }

  String[] getAns() {
    String[] ans = new String[cmds.size()];
    for (int i = 0; i < cmds.size(); i++) {
      ans[i] = getCmdString(cmds.get(i));
    }
    return ans;
  }
}

class BCmp implements Comparator<BState> {
  @Override
  public int compare(BState s0, BState s1) {
    if (s0.eval() < s1.eval()) {
      return -1;
    } else if (s0.eval() > s1.eval()) {
      return 1;
    } else {
      return 0;
    }
  }
}

class BState implements Cloneable {

  List<BigInteger> nums;
  BigInteger target;

  List<String> cmds;

  String targetBit;

  HashMap<Integer, Integer> exp2;
  HashMap<Integer, Integer> rep1;

  int ptrId;
  int ptrRLB;
  List<Integer> runLengthBlocks;

  BState(TestCase tc) {
    nums = new ArrayList<>();
    nums.add(new BigInteger("" + tc.num0));
    nums.add(new BigInteger("" + tc.num1));
    this.target = new BigInteger(tc.target);
    cmds = new ArrayList<>();
    targetBit = this.target.toString(2);

    ptrRLB = 0;

    runLengthBlocks = new ArrayList<>();
    List<String> ones = new ArrayList<>(Arrays.asList(targetBit.split("0+")));
    List<String> zeros = new ArrayList<>(Arrays.asList(targetBit.split("1+")));
    if (!zeros.isEmpty())
      zeros.remove(zeros.get(0));
    for (int i = 0; i < ones.size() - 1; i++) {
      runLengthBlocks.add(ones.get(i).length());
      runLengthBlocks.add(zeros.get(i).length());
    }
    runLengthBlocks.add(ones.get(ones.size() - 1).length());
    if (ones.size() == zeros.size())
      runLengthBlocks.add(zeros.get(zeros.size() - 1).length());

    exp2 = new HashMap<>();
    rep1 = new HashMap<>();
  }

  public int eval() {
    return cmds.size() * 1000 - exp2.size() - rep1.size();
  }

  @Override
  public BState clone() {
    BState s = null;
    try {
      s = (BState) super.clone();
      s.nums = new ArrayList<>(this.nums);
      s.cmds = new ArrayList<>(cmds);
      s.exp2 = new HashMap<>(exp2);
      s.rep1 = new HashMap<>(rep1);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return s;
  }

  boolean isCompleted() {
    return ptrRLB == runLengthBlocks.size();
  }

  BState getNextState() {

    BState s = clone();
    if (ptrRLB == 0) {
      s.preProcess();
      return s;
    }
    if (ptrRLB == runLengthBlocks.size() - 1) {
      s.postProcess();
      return s;
    }

    int len0 = s.runLengthBlocks.get(s.ptrRLB);
    int len1 = s.runLengthBlocks.get(s.ptrRLB + 1);
    int len01 = len0 + len1;
    s.operate(s.ptrId, '*', s.getOrCreateExp2(len01));
    s.operate(s.nums.size() - 1, '+', s.getOrCreateRep1(len1));
    s.ptrId = s.nums.size() - 1;
    s.ptrRLB += 2;
    return s;
  }

  boolean canGetSub0State(int depth) {
    int end = ptrRLB + 2 * (depth + 1);
    if (end >= runLengthBlocks.size())
      return false;
    for (int i = 1; i <= depth; i++) {
      int p = ptrRLB + 2 * i;
      if (runLengthBlocks.get(p) != 1)
        return false;
    }
    return true;
  }

  BState getSub0State(int depth) {
    BState s = clone();
    int end = s.ptrRLB + 2 * (depth + 1);

    int e2 = 0;
    List<Integer> subtractExp2s = new ArrayList<>();
    for (int i = end - 1; i >= s.ptrRLB; i--) {
      e2 += s.runLengthBlocks.get(i);
      if (i > s.ptrRLB && i % 2 == 1) {
        subtractExp2s.add(e2 - 1);
      }
    }
    int r1 = e2 - s.runLengthBlocks.get(s.ptrRLB);

    s.operate(s.ptrId, '*', s.getOrCreateExp2(e2));
    s.ptrId = s.nums.size() - 1;

    s.operate(s.ptrId, '+', s.getOrCreateRep1(r1));
    s.ptrId = s.nums.size() - 1;

    for (int sube : subtractExp2s) {
      s.operate(s.ptrId, '-', s.getOrCreateExp2(sube));
      s.ptrId = s.nums.size() - 1;
    }

    s.ptrRLB = end;

    return s;
  }

  boolean canGetAdd1State(int depth) {
    int end = ptrRLB + 2 * (depth + 1);
    if (end >= runLengthBlocks.size())
      return false;
    for (int i = 1; i <= depth; i++) {
      int p = ptrRLB + 2 * i - 1;
      if (runLengthBlocks.get(p) != 1)
        return false;
    }
    return true;
  }

  BState getAdd1State(int depth) {
    BState s = clone();
    int end = s.ptrRLB + 2 * (depth + 1);

    int e2 = 0;
    List<Integer> addExp2s = new ArrayList<>();
    for (int i = end - 1; i >= s.ptrRLB; i--) {
      e2 += s.runLengthBlocks.get(i);
      int j = end - i;
      if (j % 2 == 0 && i != s.ptrRLB) {
        addExp2s.add(e2);
      }
    }
    int r1 = s.runLengthBlocks.get(end - 1);

    s.operate(s.ptrId, '*', s.getOrCreateExp2(e2));
    s.ptrId = s.nums.size() - 1;

    s.operate(s.ptrId, '+', s.getOrCreateRep1(r1));
    s.ptrId = s.nums.size() - 1;

    for (int adde : addExp2s) {
      s.operate(s.ptrId, '+', s.getOrCreateExp2(adde));
      s.ptrId = s.nums.size() - 1;
    }

    s.ptrRLB = end;

    return s;
  }

  List<BState> getAllNextStates() {

    List<BState> states = new ArrayList<>();

    if (ptrRLB == 0) {
      BState s = clone();
      s.preProcess();
      states.add(s);
      return states;
    }

    if (ptrRLB == runLengthBlocks.size() - 1) {
      BState s = clone();
      s.postProcess();
      states.add(s);
      return states;
    }

    states.add(getNextState());

    for (int depth = 1; canGetSub0State(depth); depth++) {
      states.add(getSub0State(depth));
    }

    for (int depth = 1; canGetAdd1State(depth); depth++) {
      states.add(getAdd1State(depth));
    }

    return states;
  }

  void preProcess() {
    int id1 = create1();
    operate(id1, '+', id1);
    exp2.put(1, nums.size() - 1);

    int flen = runLengthBlocks.get(0);
    ptrId = getOrCreateRep1(flen);
    ptrRLB = 1;
  }

  void postProcess() {
    if (runLengthBlocks.size() % 2 == 0) {
      int len = runLengthBlocks.get(runLengthBlocks.size() - 1);
      operate(nums.size() - 1, '*', getOrCreateExp2(len));
    }
    ptrRLB++;
  }

  String[] getAns() {
    return cmds.toArray(new String[cmds.size()]);
  }

  int getOrCreateRep1(int rep) {
    if (rep1.containsKey(rep))
      return rep1.get(rep);
    int e = getOrCreateExp2(rep);
    operate(e, '-', exp2.get(0));
    rep1.put(rep, nums.size() - 1);
    return rep1.get(rep);
  }

  int getOrCreateExp2(int e) {
    if (exp2.containsKey(e))
      return exp2.get(e);
    if (e % 2 == 1) {
      if (!exp2.containsKey(e - 1)) {
        operate(getOrCreateExp2(e / 2), '*', getOrCreateExp2(e / 2));
        exp2.put(e - 1, nums.size() - 1);
      }
      operate(getOrCreateExp2(e - 1), '*', getOrCreateExp2(1));
      exp2.put(e, nums.size() - 1);
      return exp2.get(e);
    } else {
      operate(getOrCreateExp2(e / 2), '*', getOrCreateExp2(e / 2));
      exp2.put(e, nums.size() - 1);
      return exp2.get(e);
    }
  }

  int create1() {
    operate(0, '/', 0);
    exp2.put(0, nums.size() - 1);
    rep1.put(1, nums.size() - 1);
    return nums.size() - 1;
  }

  void operate(int id1, char op, int id2) {
    switch (op) {
    case '+':
      nums.add(nums.get(id1).add(nums.get(id2)));
      break;
    case '-':
      nums.add(nums.get(id1).subtract(nums.get(id2)));
      break;
    case '*':
      nums.add(nums.get(id1).multiply(nums.get(id2)));
      break;
    case '/':
      nums.add(nums.get(id1).divide(nums.get(id2)));
      break;
    }
    cmds.add(id1 + " " + op + " " + id2);
  }
}