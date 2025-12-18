Through these experiments, we explore prominent methods in Distribution Matching Dataset Distillation (DMDD), focusing on Distribution Matching (DM) and DM-like approaches for tabular data. Our methodology draws on extensive research in image-based DMDD and adapts these techniques to the tabular domain, while also accounting for the unique characteristics of tabular data and leveraging them for more effective distillation.

We perform the following exploration:

- We select a number of tabular databases for both binary and multi-class classification. 
- We define a selection of embedders, where we compare simple MLP architectures, varying from 2 dense layers to MLPs using Layer Normalization and Batch Normalization, using residual connections and varying widths and depths. The embedders catalogue is detailed in the implementation noted (README.md)


### Experiments :

We observe that LayerNorm-based random embedders consistently outperform BatchNorm-based embedders for distribution matching on tabular data. We hypothesize this is because BatchNorm introduces batch-dependent feature coupling, which violates the assumption of fixed embedding spaces required for moment matching. In contrast, LayerNorm produces sample-wise normalized embeddings, yielding more stable and distribution-faithful statistics, particularly in the low-IPC regime common in dataset condensation.

We choose random initialization for each embedder, to avoid overfitting to a particular embedder. This is the standard approach for image DMDD

| Dataset   | Full AUC  | Rand AUC  | VQ AUC  | BN AUC   | BNDeep AUC | BNWide AUC | BNRes AUC | BNCascade AUC |
|-----------|-----------|-----------|-----------|----------|------------|------------|-----------|----------------|
| adult     | 0.895058  | 0.731950  | 0.840260  | 0.828890 | 0.823710   | 0.821695   | 0.839648  | 0.821203       |
| airlines  | 0.700636  | 0.526157  | 0.651348  | 0.533718 | 0.498597   | 0.495686   | 0.595988  | 0.491306       |
| bank      | 0.915145  | 0.692192  | 0.846062  | 0.710006 | 0.675850   | 0.676163   | 0.802547  | 0.668721       |
| covertype | 0.990084  | 0.732284  | 0.823773  | 0.796904 | 0.786684   | 0.786699   | 0.822952  | 0.784098       |
| credit    | 0.770316  | 0.670140  | 0.692166  | 0.657193 | 0.633681   | 0.633513   | 0.696522  | 0.630608       |
| drybean   | 0.999642  | 0.969517  | 0.977058  | 0.975577 | 0.978515   | 0.978541   | 0.973593  | 0.978011       |
| higgs     | 0.786457  | 0.508003  | 0.623686  | 0.531963 | 0.512929   | 0.512947   | 0.562216  | 0.512469       |

| Dataset   |   Full AUC |   Rand AUC | VQ AUC     | LN AUC     | LNDeep AUC   | LNWide AUC   | LNRes AUC    | LNCascade AUC   |
|:----------|-----------:|-----------:|:-------------|:-----------|:-------------|:-------------|:-------------|:----------------|
| adult     |   0.895058 |   0.73195  | 0.840271     | 0.842904   | 0.844155     | *0.844245*   | **0.845593** | 0.840814        |
| airlines  |   0.700636 |   0.526157 | **0.651348** | 0.605764   | 0.595468     | 0.603687     | *0.616599*   | 0.586157        |
| bank      |   0.915145 |   0.692192 | **0.846056** | *0.810129* | 0.809082     | 0.805788     | 0.808435     | 0.783345        |
| covertype |   0.990084 |   0.732284 | **0.823756** | 0.817764   | 0.815849     | 0.817965     | *0.821923*   | 0.809592        |
| credit    |   0.770316 |   0.67014  | 0.692139     | 0.727118   | **0.731637** | 0.728462     | *0.730588*   | 0.726402        |
| drybean   |   0.999642 |   0.969517 | **0.977028** | 0.973334   | 0.973476     | 0.973606     | 0.973151     | *0.974667*      |
| higgs     |   0.786457 |   0.508003 | **0.623686** | 0.579255   | 0.578725     | 0.578672     | *0.582558*   | 0.577789        |

**We experiment with DM through Batch Normalization Statistics extracted from a teacher model.**

| Dataset   |   Full AUC |   Rand AUC |   VQ AUC |   DM-BN AUC |
|:----------|-----------:|-----------:|-----------:|------------:|
| adult     |   0.895058 |   0.73195  |   0.840265 |    0.737701 |
| airlines  |   0.700636 |   0.526157 |   0.651348 |    0.523843 |
| bank      |   0.915145 |   0.692192 |   0.846069 |    0.594989 |
| covertype |   0.990084 |   0.732284 |   0.823724 |    0.691984 |
| credit    |   0.770316 |   0.67014  |   0.692119 |    0.620799 |
| drybean   |   0.999642 |   0.969517 |   0.977057 |    0.936522 |
| higgs     |   0.786457 |   0.508003 |   0.623692 |    0.535829 |

**We also perform an ablation test involving higher order moments, adding skewness and kurtosis**

|dataset  |acc_M1      |acc_M2      |acc_M3      |acc_M4      |
|---------|------------|------------|------------|------------|
|adult    |0.7589736591|0.767571994 |0.7626586598|0.7638869933|
|airlines |0.6008058536|0.6034755525|0.5997923568|0.6032654373|
|bank     |0.7583308758|0.7435859628|0.7446181068|0.7475670894|
|covertype|0.4674476776|0.4666789058|0.4637300349|0.4725078025|
|credit   |0.6257777778|0.6166666667|0.6084444444|0.6042222222|
|drybean  |0.6748285994|0.6728697356|0.6753183154|0.6762977473|
|higgs    |0.5575877074|0.5598993745|0.5596954039|0.5602393255|

**One conclusion is that Vector Quantization [VQ] or Herding regularly out-perform DM methods, as a potential particularity of tabular data low dimension feature space. We test the following methods:**

Quick summary:
- Vector Quantization uses K-Means to pick synthetic samples as the centroids of each cluster
- Voronoi-Restricted sampling uses K-means to pick real samples close to the centroids of each cluster
- Herding is an iterative algorithm that builds a real subset by minimizing distribution distance between the subset and the real dataset
- Gonzalez is an algorithm that builds a real subset to maximize global coverage of the entire class.

|dataset  |full_acc    |full_auc    |random_acc  |random_auc  |vq_acc             |vq_auc            |voronoi_acc        |voronoi_auc       |gonzalez_acc       |gonzalez_auc      |herding_acc       |herding_auc       |
|---------|------------|------------|------------|------------|-------------------|------------------|-------------------|------------------|-------------------|------------------|------------------|------------------|
|adult    |0.847413675446977|0.89607522989448|0.6925071652791047|0.7319790196149467|0.7376825440152859 |0.8456630091916857|0.6967380919885355 |0.8214922350551446|0.255766343660434  |0.7390752149526436|0.6286338201173741|0.7794852066609479|
|airlines |0.654088594452959|0.7004971043683992|0.5275374499431453|0.5261278196162932|0.6042789340979878 |0.6393872432757992|0.5811909823503238 |0.612620628847703 |0.5166238196470064 |0.4923791654681826|0.5664334800019776|0.5924191959504941|
|bank     |0.8998820406959599|0.9144989858456349|0.6470067826599823|0.6922303121085589|0.7565614862872309 |0.8420131106271544|0.6492185196107343 |0.7711792763403778|0.24609259805367148|0.5234478848043607|0.5958419345325863|0.7239605101997632|
|covertype|0.8923031026252983|0.9898091396073695|0.2986047365522306|0.7322710521990227|0.38710528731411786|0.8187704802513013|0.38945749954103176|0.8070701392571168|0.04824903616669726|0.6641993406797345|0.38664631907472  |0.8014007521060208|
|credit   |0.8171111111111111|0.7707324732720838|0.6517777777777778|0.6701358802791073|0.4088888888888889 |0.6921530299279309|0.4888888888888889 |0.6881239283618493|0.5264444444444445 |0.45802115425171  |0.6142222222222222|0.5148218444554473|
|drybean  |0.9862879529872673|0.9996302728698676|0.7340842311459354|0.9695059823998949|0.7203721841332027 |0.977058171324032 |0.7272282076395691 |0.9765513380306496|0.5200783545543585 |0.9532886231911162|0.7389813907933399|0.9638983370115831|
|higgs    |0.7124694044057656|0.7866580109316932|0.5192412292629861|0.5079964506391217|0.5893391351645363 |0.6207032119296875|0.5550040794125646 |0.573106783062235 |0.5314794669567582 |0.5193466770266371|0.5123062279031819|0.4918741084764885|



TO DO:
- Redo part of the evaluation to have uniform metrics (acc vs auc)
- visualization: the relatively low-dimensional feature space of tabular data may allow us to create significant visualizations
- *replace MLP with RF/XGBoost and match distributions based on specific decision tree statistics*. The motivation for this is the assumption that feature interactions are much better captured by decision trees than MLPs
- Gradient Matching and Trajectory Matching
- Analyze the impact of different methods in regards to outlier density and on heavy-tailed distributions
- concatenate databases. Test for heavy real-life dbs