Nombre de descriptions de colonnes chargées: 133
=== Informations sur les données ===
Forme des données d'entraînement: (2811, 135)
Forme des données de test: (1206, 134)

Période des données d'entraînement: 2008-11-25 à 2019-09-03
Période des données de test: 2019-09-04 à 2024-04-17

=== Aperçu des données d'entraînement ===
[{"index":0,"Dates":"2008-11-25","Features_0":-2.8204092106314125,"Features_1":-2.6026152851661646,"Features_2":-0.9546427206257788,"Features_3":-2.599413785212495,"Features_4":-2.33528624703703,"Features_5":-1.6919599940242145,"Features_6":-1.780899488542422,"Features_7":-1.595611234135332,"Features_8":-1.235825125240457,"Features_9":2.04328109184001,"Features_10":1.9257778479160104,"Features_11":1.1818542189504255,"Features_12":-1.1600187836077005,"Features_13":-1.4017417201545566,"Features_14":-1.601822820063336,"Features_15":-1.2919218177093363,"Features_16":-1.5071574911707484,"Features_17":-1.5783320545818895,"Features_18":-1.1212466829549954},{"index":1,"Dates":"2008-11-26","Features_0":-2.8669909066099084,"Features_1":-2.7083993202222203,"Features_2":-1.15085059098512,"Features_3":-2.459071923185073,"Features_4":-2.327877231115884,"Features_5":-1.7035286655274702,"Features_6":-1.6926707925462636,"Features_7":-1.44413367698086,"Features_8":-1.1570473637448864,"Features_9":1.8087793119279791,"Features_10":1.6376155946124382,"Features_11":1.033183738815722,"Features_12":-1.1756302155256122,"Features_13":-1.3500928663224472,"Features_14":-1.5010881172494488,"Features_15":-1.3759439132681623,"Features_16":-1.514458204973519,"Features_17":-1.5485400986182527,"Features_18":-1.1956931697159263},{"index":2,"Dates":"2008-11-27","Features_0":-2.5411181372355403,"Features_1":-2.5455329157501043,"Features_2":-1.0865240100535598,"Features_3":-2.517488221380138,"Features_4":-2.3041471348978906,"Features_5":-1.6546268605010606,"Features_6":-1.6902150153400093,"Features_7":-1.417283015455722,"Features_8":-1.1121808780959048,"Features_9":1.9749503280956648,"Features_10":1.805696988629683,"Features_11":1.1143629492375349,"Features_12":-0.9999274503735528,"Features_13":-1.1962117798456928,"Features_14":-1.4167667369244283,"Features_15":-1.1694296802915436,"Features_16":-1.300434931249124,"Features_17":-1.4331724903542824,"Features_18":-0.9391854966412048},{"index":3,"Dates":"2008-11-28","Features_0":-2.6708754893158644,"Features_1":-2.5658616877173026,"Features_2":-1.041002485413028,"Features_3":-2.605889604992041,"Features_4":-2.2947844372658768,"Features_5":-1.5890888448850315,"Features_6":-1.6891149463030577,"Features_7":-1.3918608489484607,"Features_8":-1.0712084383472125,"Features_9":1.975350005325028,"Features_10":1.696516853754,"Features_11":1.05260555696825,"Features_12":-1.020623678761599,"Features_13":-1.2052565973376506,"Features_14":-1.41450348535071,"Features_15":-1.188690858262401,"Features_16":-1.305012437371333,"Features_17":-1.4159292698802997,"Features_18":-0.9683464922926978},{"index":4,"Dates":"2008-12-01","Features_0":-2.7520083772358133,"Features_1":-2.5984456920436734,"Features_2":-1.0743659344239556,"Features_3":-2.7210056427779232,"Features_4":-2.394467046720437,"Features_5":-1.6620950791667195,"Features_6":-1.7963815231282736,"Features_7":-1.4744411240255133,"Features_8":-1.098121695518076,"Features_9":2.166981677198375,"Features_10":1.7685185112796582,"Features_11":1.049993390877631,"Features_12":-0.9600309139380596,"Features_13":-1.2203108568711605,"Features_14":-1.3529469523903312,"Features_15":-1.17321150331028,"Features_16":-1.3745373860148955,"Features_17":-1.4098552586194095,"Features_18":-0.9537916001333784}]

Nombre de features: 133
Target column: ToPredict

=== Analyse des valeurs manquantes ===
Valeurs manquantes dans train: 0
Valeurs manquantes dans test: 0

=== Statistiques de la variable cible ===
count    2811.000000
mean        0.136481
std         0.278196
min        -0.626089
25%        -0.064782
50%         0.129309
75%         0.315561
max         1.144987
Name: ToPredict, dtype: float64


=== Test ADF pour Variable cible ===
Statistique ADF: -8.5173
p-value: 0.0000
Valeurs critiques:
  1%: -3.4327
  5%: -2.8626
  10%: -2.5673
Conclusion: La série Variable cible est stationnaire

=== Test ADF pour Feature Features_0 ===
Statistique ADF: -3.7989
p-value: 0.0029
Valeurs critiques:
  1%: -3.4327
  5%: -2.8626
  10%: -2.5673
Conclusion: La série Feature Features_0 est stationnaire

=== Test ADF pour Feature Features_1 ===
Statistique ADF: -3.8592
p-value: 0.0024
Valeurs critiques:
  1%: -3.4327
  5%: -2.8626
  10%: -2.5673
Conclusion: La série Feature Features_1 est stationnaire

=== Test ADF pour Feature Features_2 ===
Statistique ADF: -2.4003
p-value: 0.1417
Valeurs critiques:
  1%: -3.4327
  5%: -2.8626
  10%: -2.5673
Conclusion: La série Feature Features_2 n'est pas stationnaire



=== Évaluation de Ridge ===
  Fold 1: MSE=0.094038, MAE=0.250337, R2=-0.2198
  Fold 2: MSE=0.092944, MAE=0.240254, R2=-0.4194
  Fold 3: MSE=0.113981, MAE=0.280589, R2=-0.4404
  Fold 4: MSE=0.172794, MAE=0.345093, R2=-1.7049
  Fold 5: MSE=0.126293, MAE=0.295270, R2=-0.4003
  Moyenne: MSE=0.120010 (+/- 0.029220)

=== Évaluation de Lasso ===
  Fold 1: MSE=0.039962, MAE=0.159188, R2=0.4816
  Fold 2: MSE=0.039492, MAE=0.155377, R2=0.3969
  Fold 3: MSE=0.044400, MAE=0.164859, R2=0.4389
  Fold 4: MSE=0.046919, MAE=0.166759, R2=0.2655
  Fold 5: MSE=0.047703, MAE=0.174569, R2=0.4711
  Moyenne: MSE=0.043696 (+/- 0.003422)

=== Évaluation de ElasticNet ===
  Fold 1: MSE=0.034576, MAE=0.148464, R2=0.5515
  Fold 2: MSE=0.033523, MAE=0.142172, R2=0.4881
  Fold 3: MSE=0.035527, MAE=0.145041, R2=0.5510
  Fold 4: MSE=0.041075, MAE=0.153174, R2=0.3570
  Fold 5: MSE=0.035561, MAE=0.144367, R2=0.6057
  Moyenne: MSE=0.036052 (+/- 0.002620)

=== Évaluation de Huber ===
  Fold 1: MSE=0.069900, MAE=0.214271, R2=0.0933
  Fold 2: MSE=0.113616, MAE=0.270765, R2=-0.7351
  Fold 3: MSE=0.091104, MAE=0.249325, R2=-0.1513
  Fold 4: MSE=0.158003, MAE=0.322596, R2=-1.4733
  Fold 5: MSE=0.083791, MAE=0.233993, R2=0.0709
  Moyenne: MSE=0.103283 (+/- 0.030802)

=== Évaluation de RandomForest ===
  Fold 1: MSE=0.024738, MAE=0.121273, R2=0.6791
  Fold 2: MSE=0.014053, MAE=0.092629, R2=0.7854
  Fold 3: MSE=0.014767, MAE=0.091428, R2=0.8134
  Fold 4: MSE=0.014314, MAE=0.087115, R2=0.7759
  Fold 5: MSE=0.013987, MAE=0.090930, R2=0.8449
  Moyenne: MSE=0.016372 (+/- 0.004192)

=== Évaluation de ExtraTrees ===
  Fold 1: MSE=0.024283, MAE=0.120634, R2=0.6850
  Fold 2: MSE=0.015707, MAE=0.095122, R2=0.7601
  Fold 3: MSE=0.020965, MAE=0.104301, R2=0.7351
  Fold 4: MSE=0.017027, MAE=0.095843, R2=0.7335
  Fold 5: MSE=0.016515, MAE=0.094837, R2=0.8169
  Moyenne: MSE=0.018899 (+/- 0.003244)

=== Évaluation de GradientBoosting ===
  Fold 1: MSE=0.027392, MAE=0.132286, R2=0.6447
  Fold 2: MSE=0.015241, MAE=0.094555, R2=0.7673
  Fold 3: MSE=0.016352, MAE=0.094845, R2=0.7934
  Fold 4: MSE=0.017824, MAE=0.098019, R2=0.7210
  Fold 5: MSE=0.017951, MAE=0.104073, R2=0.8010
  Moyenne: MSE=0.018952 (+/- 0.004337)

=== Évaluation de XGBoost ===
  Fold 1: MSE=0.026275, MAE=0.128455, R2=0.6592
  Fold 2: MSE=0.016773, MAE=0.099243, R2=0.7439
  Fold 3: MSE=0.017737, MAE=0.098612, R2=0.7759
  Fold 4: MSE=0.015294, MAE=0.090266, R2=0.7606
  Fold 5: MSE=0.016059, MAE=0.094261, R2=0.8219
  Moyenne: MSE=0.018427 (+/- 0.004006)

=== Évaluation de LightGBM ===
  Fold 1: MSE=0.028192, MAE=0.131803, R2=0.6343
  Fold 2: MSE=0.015625, MAE=0.097791, R2=0.7614
  Fold 3: MSE=0.017159, MAE=0.096313, R2=0.7832
  Fold 4: MSE=0.015168, MAE=0.090317, R2=0.7626
  Fold 5: MSE=0.015216, MAE=0.091267, R2=0.8313
  Moyenne: MSE=0.018272 (+/- 0.005013)

=== Évaluation de CatBoost ===
  Fold 1: MSE=0.029879, MAE=0.136346, R2=0.6124
  Fold 2: MSE=0.020484, MAE=0.106238, R2=0.6872
  Fold 3: MSE=0.023118, MAE=0.108940, R2=0.7078
  Fold 4: MSE=0.017399, MAE=0.096355, R2=0.7276
  Fold 5: MSE=0.018954, MAE=0.102250, R2=0.7898
  Moyenne: MSE=0.021967 (+/- 0.004383)

=== Résumé des performances ===
              model   val_mse   val_mae    val_r2  overfit_ratio
4      RandomForest  0.016372  0.096675  0.779748       9.061404
8          LightGBM  0.018272  0.101498  0.754544       7.322312
7           XGBoost  0.018427  0.102167  0.752285      24.777688
5        ExtraTrees  0.018899  0.102147  0.746113      27.516801
6  GradientBoosting  0.018952  0.104755  0.745452      40.304126
9          CatBoost  0.021967  0.110026  0.704990       3.031195
2        ElasticNet  0.036052  0.146644  0.510665       1.231820
1             Lasso  0.043696  0.164150  0.410807       1.200670
3             Huber  0.103283  0.258190 -0.439095      11.483660
0             Ridge  0.120010  0.282309 -0.636962      16.634120