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



            
