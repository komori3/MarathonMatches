# MM116: Lossy2dCompression

#### provisional  

## 概要

* グリッドを重ねていい感じに圧縮しような

## 垂れ流し

### 3/20

---

#### 1:28

重実装が出るといいな～って言ってたら何とも言えんやつ来た

とりあえずグリッドを被覆する最小サイズの長方形の全色を ('A'+'Z')/2 にしたやつを投げる

local: all_median_color_trim

submission 1: 76.49346

---

#### 2:18

圧縮前の各グリッドを<b>レイヤー</b>、圧縮後のグリッドを<b>ベース</b>とでも呼ぶことにする

MM106 StainedGlass でもあったけど絶対偏差和は中央値で最小化される

つまり、各レイヤーを置く場所さえ決めてしまえば loss 最小化するようなベースは最適が求まる

とりあえず全レイヤーのオフセットを (0, 0) に配置してベース計算する

直前より 6% くらい良くなる

local: adjust_median_of_each_cell

submission 2: 81.85141

同じ点の人がいた　まあそうだよね

---

#### 2:30

とりあえず盤面サイズは fix したままオフセットをガチャして乱択する

というのは建前で、そろそろスコア計算関数を書きたかったという話

直前より 4% くらい良くなる

local: naive_random

submission 3: 82.11632

---

#### 10:33

P があるから盤面サイズは小さければいいってもんじゃない

ランダムに盤面縦横を最小 + 30 までの範囲で変化させて乱択すると 5% くらい上がったので投げる

local: naive_varies_random2

submission 4: 83.26007

---

#### 15:14

P がめちゃくちゃ小さいときは圧縮の恩恵が少ないので、  
<a href="http://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1726-05.pdf">長方形ストリップパッキング</a>をして誤差 0 にしたほうがいい

Bottom-Left-Fill(BLF) アルゴリズムで正方形っぽく敷き詰めて得られたスコアと、以前の乱択解の良いほうを採用

これで 10% くらい向上　やったぜ

local: naive_varies_random2_with_blf

submission 5: 92.04158 (7)

---

#### 16:34

じゃあ P がちょっと小さいくらいの時はどうすんの？という話で、  
多分エッジの k ピクセルの重なりを許すようにしてパッキングしてやるといい感じになると思う

local: naive_varies_random2_with_blf_margin

-> いい感じにならなかった　想像以上に diff ペナルティが大きいため…

---

#### 18:00

とりあえず BLF の strip_width を可変にしたら 2% くらい上がった (非本質っぽい…)

local: naive_varies_random2_with_blf2

submission 6: 93.0047

---

#### 19:20

後半の乱択部分が適当すぎる

盤面サイズ最小に固定して、各レイヤーを最適な位置に移動させる処理を五回ほど繰り返すと大体局所最適に落ちる

```
submission                              score
--------------------------------------------------
sample_submit                           54.43693286370808
all_median_color                        74.51712683220735
all_median_color_trim                   74.8070201845515
adjust_median_of_each_cell              78.95417476405044
naive_random                            82.062609333891
naive_varies_random                     83.36015997352955
naive_varies_random2                    84.50191208370599
naive_varies_random2_with_blf_margin    94.116691029719
naive_varies_random2_with_blf           94.11738937601991
naive_varies_random2_with_blf2          95.68322306585821
slide_layers_greedy_with_blf2           99.94399124649874
```

ローカルで 4% くらい上がった　ふむふむ　方針はよさそう

local: slide_layers_greedy_with_blf2

submission 7: 96.73498 (4)

いいね～

---

最小に固定しないと 1% くらい伸びる

```
submission                                score
--------------------------------------------------
sample_submit                             53.80854030217885
all_median_color                          73.60213079066159
all_median_color_trim                     73.89163419397914
adjust_median_of_each_cell                77.9909007928933
naive_random                              81.06999731120621
naive_varies_random                       82.36448994250212
naive_varies_random2                      83.51091313176853
naive_varies_random2_with_blf_margin      93.11011434093318
naive_varies_random2_with_blf             93.11083714974689
naive_varies_random2_with_blf2            94.6636773221787
slide_layers_greedy_with_blf2             98.86205271199135
slide_layers_greedy_adaptive_with_blf2    99.8166762354329
slide_layers_greedy_multiple_with_blf2    99.8647358741247
```

local: slide_layers_greedy_adaptive_with_blf2

submission 8: 97.43517 (4)

---

### 3/21

#### 19:45

諸々の高速化をして多点スタートにした

```
submission                                        score
--------------------------------------------------
sample_submit                                     53.15337484861259
all_median_color                                  72.6922098170324
all_median_color_trim                             72.97816177441425
adjust_median_of_each_cell                        77.00570146866033
naive_random                                      80.03173417884884
naive_varies_random                               81.31286357640917
naive_varies_random2                              82.4622232726847
naive_varies_random2_with_blf_margin              92.01163454713564
naive_varies_random2_with_blf                     92.01238054313639
naive_varies_random2_with_blf2                    93.54311102585463
slide_layers_greedy_with_blf2                     97.69333085467169
slide_layers_greedy_adaptive_with_blf2            98.63314547718845
slide_layers_greedy_multiple_with_blf2            98.6787714958137
slide_layers_greedy_adaptive_multiple_with_blf    99.98333840393761
```

local: slide_layers_greedy_adaptive_multiple_with_blf

submission 9: 98.44252 (3)

---

### 3/22

#### 21:30

起きたら 8 位に下がっている

あとやることは

* BLF の乱択
* レイヤー移動の焼きなまし化

とかかな

BLF の充填率が 93% とかなのでここを詰める余地がある

---

#### 23:50

BLF 敷き詰め順序の焼きなましで 0.4% くらい上がる

```
submission                                           score
--------------------------------------------------
sample_submit                                        52.9877376026796
all_median_color                                     72.44993489987783
all_median_color_trim                                72.73549503553976
adjust_median_of_each_cell                           76.74040505452491
naive_random                                         79.74954722261934
naive_varies_random                                  81.00586990534404
naive_varies_random2                                 82.12664211799726
naive_varies_random2_with_blf_margin                 91.54566280146697
naive_varies_random2_with_blf                        91.5464096633338
naive_varies_random2_with_blf2                       93.05110088231007
slide_layers_greedy_with_blf2                        97.19289735502653
slide_layers_greedy_adaptive_with_blf2               98.13013638381754
slide_layers_greedy_multiple_with_blf2               98.17519599344632
slide_layers_greedy_adaptive_multiple_with_blf       99.47493594827259
slide_layers_greedy_adaptive_multiple_with_blf_SA    99.88518546281428
```

local: slide_layers_greedy_adaptive_multiple_with_blf_SA

submission 10: 98.55568 (7)

---

### 3/23

#### 3:15

レイヤー移動を焼きなましにして、諸々チューニング

```
submission                                           score
--------------------------------------------------
sample_submit                                        52.7521794706044
all_median_color                                     72.1094078492751
all_median_color_trim                                72.39492214086604
adjust_median_of_each_cell                           76.38543601483889
naive_random                                         79.38532445295047
naive_varies_random                                  80.63905593809912
naive_varies_random2                                 81.75959008735131
naive_varies_random2_with_blf_margin                 91.16136846967294
naive_varies_random2_with_blf                        91.1621295808266
naive_varies_random2_with_blf2                       92.65974019280803
slide_layers_greedy_with_blf2                        96.77982311732501
slide_layers_greedy_adaptive_with_blf2               97.70179639581794
slide_layers_greedy_multiple_with_blf2               97.74610694766798
slide_layers_greedy_adaptive_multiple_with_blf       99.03643719146544
slide_layers_greedy_adaptive_multiple_with_blf_SA    99.44564103463958 <- prev
slide_layers_adaptive_multiple_SA_with_blf_SA        99.62029623962452
slide_layers_adaptive_SA_with_blf_SA                 99.69965998676754
slide_layers_SA_and_blf_SA                           99.70779448258128
```

local: slide_layers_SA_and_blf_SA

submission 11: 98.56444 (6)

99 点勢はもう一工夫してそうだなあ

---

#### ~ コンテスト終了まで

```
submission                                           score
--------------------------------------------------
sample_submit                                        52.59532491532023
all_median_color                                     71.88259776967053
all_median_color_trim                                72.16755313349623
adjust_median_of_each_cell                           76.14221986382688
naive_random                                         79.12722990259793
naive_varies_random                                  80.37605893144911
naive_varies_random2                                 81.48520378219189
naive_varies_random2_with_blf_margin                 90.79876482066955
naive_varies_random2_with_blf                        90.79952604421659
naive_varies_random2_with_blf2                       92.28610944257797
slide_layers_greedy_with_blf2                        96.39987635365655
slide_layers_greedy_adaptive_with_blf2               97.31774938325866
slide_layers_greedy_multiple_with_blf2               97.36151381624289
slide_layers_greedy_adaptive_multiple_with_blf       98.64819098854407
slide_layers_greedy_adaptive_multiple_with_blf_SA    99.05512610405205
slide_layers_adaptive_multiple_SA_with_blf_SA        99.228933517946
slide_layers_adaptive_SA_with_blf_SA                 99.3079729833499
slide_layers_SA_and_blf_SA                           99.31606118581523
test3                                                99.38808347011896
test                                                 99.42055309306741
test2                                                99.50926138777288
test4                                                99.57340494303301
```

書くのだるくなってきた　最終的に名前も全部 test とかにしてしまった

ちょっと点が上がったやつ：
* <a href="https://wiki.kimiyuki.net/%E4%B8%AD%E5%A4%AE%E5%80%A4">これ</a>を参考に中央値と偏差和を爆速で求めるデータ構造をつくった (seed2 で 7M iter くらい)
* BLF パートで<a href="https://www.keisu.t.u-tokyo.ac.jp/data/2008/METR08-22.pdf">徐々に狭めていく手法っぽいやつ</a>を真似した (狭めてはみ出たらランダムに飛ばす)
* 100000 回に一回必ず grid をランダムでどっかにふっ飛ばすようにした


P = 0.25 近辺の盤面サイズを上手く調整しきれていない感があった  
P, N, T と最良スコアが出る時の盤面サイズのデータを大量に集めて回帰っぽいことをしたかったけど、断念

12 位くらい…？キュー詰まりすぎてウケる