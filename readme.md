## 0. 机器学习算法分类
数据集构成（特征值 + 目标值）
1. 监督学习 (有目标值)
    1. 目标值: 离散(类别) - 分类学问题
        - k-近邻算法、贝叶斯分类、决策树与随机森林、逻辑回归
    2. 目标值: 连续性的数据 - 回归问题
        - 线性回归、岭回归
2. 无监督学习 (无目标值)
    - 聚类、k-means
## 1. 可用数据集 
1. Kaggle: https://www.kaggle.com/datasets
2. UCI: http://archive.ics.uci.edu/ml/
3. scikit-learn: http://scikit-learn.org/stable/datasets/index.html#datasets

## 2. sklearn数据集
1. sklearn.datasets
    - `datasets.load_*()`
        获取小规模数据集，数据包含在datasets里
    - `datasets.fetch_*(data_home=None)`
        获取大规模数据集，需要从网络上下载。data_home表示数据集下载的目录，默认是 ~/scikit_learn

2. sklearn小数据集
    - `sklearn.datasets.load_iris()`
        加载并返回鸢尾花数据集
    - `sklearn.datasets.load_boston()`
        加载并返回波士顿房价数据集

3. sklearn大数据集
    - `sklearn.datasets.fetch_20newsgroups(data_home=None,subset='train')`
        - subset: 'train'或者'test', 'all'可选，选择要加载的数据集

4. sklearn数据集的使用
    - load和fetch返回的数据类型datasets.base.Bunch(字典dict格式)
        - **data**: 特征数据数组，是 [n_samples * n_features] 的二维 numpy.ndarray 数组
        - **target**: 标签数组，是n_samples的一维numpy.ndarray数组
        - **DESCR**: 数据描述
        - **feature_names**: 特征名，新闻数据，手写数字、回归数据集没有
        - **target_names**: 标签名
    - 获取数据方法:
        - `dict["key"] = values`
        - `bunch.key = values`

5. sklearn数据划分api
    - `sklean.model_selection.train_test_split(arrays, *options)`
    - x 数据集的特征值
    - y 数据集的标签值
    - test_size 测试集的大小，一般为float
    - random_state 随机数种子，不同的种子会造成不同的随机采样结果。相同的种子采样结果相同
    - return 训练集特征值(x_train)，测试集特征值(x_test)，训练集目标值(y_train)，测试集目标值(y_test)

## 3. 数据集的划分
**思考: 拿到的数据是否全部都用来训练一个模型?**  
1. 机器学习一般的数据集会划分为两个部分:
    - 训练数据: 用于训练、**构建模型**
    - 测试数据: 在模型检验时使用，用于**评估模型是否有效**
2. 划分比例
    - 训练集: 70% ~ 80%
    - 测试集: 20% ~ 30%

## 4. 特征工程
数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限  
1. 常用工具:  
    - pandas: 数据清洗、数据处理  
    - sklearn: 特征工程  
2. 特征工程:  
    - 特征抽取/特征提取  
    - 特征预处理  
    - 特征降维  
3. **特征提取**  
    - 将任意数据 (如文本或图像) 转换为可用于机器学习的数字特征
    - 特征值化是为了计算机更好的去理解数据
    1. 特征提取API  
       `sklearn.feature_extraction`
    2. 字典特征提取 (类别 -> [one-hot编码](https://blog.csdn.net/qq_41933542/article/details/106711111))
        - `sklearn.feature_extraction.DictVectorizer(sparse=True,...)` 父类: transformer
            - `DictVectorizer.fit_transform(X)` X: 字典或者包含字典的迭代器 返回值: 返回sparse(稀疏)矩阵
            - `DictVectorizer.inverse_transform(X)` X: array数组或者sparse矩阵 返回值: 转换之前数据格式
            - `DictVectorizer.get_feature_names()` 返回类别名称 
        - 稀疏矩阵: 将非零值 按位置表示出来 节省内存 - 提高加载效率
        - 应用场景
            1. pclass, sex 数据集当中类别特征比较多时
                - 将数据集的特征 -> 字典类型
                - DictVectorizer转换
            2. 本身拿到的数据就是字典类型
    3. 文本特征提取
        - 单词 作为 特征
        - 方法1: `sklearn.feature_extraction.text.CountVectorizer(stop_words=[])` 返回词频矩阵 (统计每个样本特征词出现的个数)
            - stop_words 停用词 (不进行统计)
            - `CountVectorizer.fix_transform(X)` X:文本或者包含文本的字符串的可迭代对象 返回值: 返回sparse矩阵
            - `CountVectorizer.inverse_transofrm(X)` X:array数组或者sparse矩阵 返回值: 转换之前数据格
            - `CountVectorizer.get_feature_names()` 返回值: 单词列表
        - 方法2: Tf-idf 文本特种提取 `TfidfVectorizer`  
            - 主要思想: 如果某个词在某一个类别的文章中，出现的概率高，但是在其他类别的文章中出现很少，则认为此词或者短语具有良好的类别区分能力，适合用来分类。
            - TF: 词频 (term frequenc) 指的是某一个词在文章中出现的频率
            - IDF: 逆向文档频率 (inverse document frequency) 是一个词语普遍重要性的度量。可以由总文件数目除以包含该词语文件的数目，再将得到的商取以10为底的对数 (log<sub>10</sub>)
            - TF-IDF (数值越大体现词越重要): <strong style="color:green">tfidf<sub>i,j</sub> = tf<sub>i,j</sub> * idf<sub>i</sub></strong>
            - Ex. 两个词 "经济" "非常"  
                1000篇文章 - 语料库  
                100篇文章包含"非常"  
                10篇文章包含"经济"

                文章A (100词): 10次"经济"   
                TF-IDF: 0.2
                - tf: 10/100 = 0.1
                - idf: log<sub>10</sub>(1000/10) = 2
                
                文章B (100词): 10次"非常"  
                TF-IDF: 0.1
                - tf: 10/100 = 0.1
                - idf: log<sub>10</sub>(1000/100) = 1
            - `sklearn.feature_extraction.text.TfidfVectorizer(stop_words=None,...)`
                - 返回词的权重矩阵
    4. **特征预处理**
        - 什么是特征预处理: 通过一些转换函数将特征数据转换成更加适合算法模型的特征数据过程
        - 归一化/标准化: 特征的单位或者大小相差较大，或者某特征的方差相比其他的特征要大出几个数量级，容易影响目标结果，所以要进行无量纲化
        - 归一化
            - $$ x' = {x - min \over max - min}$$
            - $$ x'' =  x' * (mx - mi) + mi$$
            - 作用于每一列，max为一列的最大值，min为一列的最小值，那么$x''$为最终结果，mx,mi分别为指定区间(默认值mx
            为1, mi为0)
            - `sklearn.preprocessing.MinMaxScaler(feature_range=(0,1),...)`
                - `MinMaxScalar.fit_transform(X)`       
                    - X: numpy array格式的数据[n_samples,n_features]
                    - 返回值: 转换后形状相同的array  
            - 缺陷: 
                - 异常值: 最大值、最小值异常会影响归一化结果
        - 标准化
            - 通过对原始数据进行变换把数据变换到均值为0，标准差为1范围内
            - $$ x' = {x - mean \over \sigma}$$
            - `sklearn.preprocessing.StandardScaler()`
                - 处理之后，对每列来说所有数据都聚集在均值为0附近，标准差为1
                - `StandardScler.fit_transform(X)` 
                    - X: numpy array格式的数据
                    [n_samples, n_features]
                    - 返回值: 转换后的形状相同的array
            - 在已有样本足够多的情况下比较稳定，适合现代嘈杂大数据场景
    5. **特征降维**
        - 降维是指在某些限定条件下，**降低随机变量(特征)个数**，得到一组"不相关"主变量的过程
            - 降低随机变量(特征)的个数
            - 相关特征(correlated feature)
                - 例如相对湿度与降雨量之间的相关
        - 降维的两种方式：
            - 特征选择
            - 主成分分析
        - 特征选择
            - 定义: 数据中包含冗余或者相关变量(或者特征、属性、指标等)，旨在从原有特征中找出主要特征






            
