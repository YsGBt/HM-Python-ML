# 机器学习_周志华
## 术语
- `模型 "model" / 学习器 "learner"`·`: 数据中学习的结果

- `数据集 "data set"`: 记录的集合

- `样本 "sample" / 示例 "instance"`: 一个事件或对象  

- `属性 "attribute" / 特征 "feature"`: 反映事件或对象在某方面的表现或性质的事项

- `属性值 "attribute value"`: 属性上的取值

- `属性空间 "attribute space" / 样本空间 "sample space"`: 属性张成的空间

- `维数 "dimensionality"`: 令 $D = \{x_1, x_2, ..., x_m\}$ 表示包括 $m$ 个样本的数据集，每个样本由 $d$ 个属性描述，则每个样本 $x_i = (x_{i1}; x_{i2}; ...; x_{id})$ 是 $d$ 维样本空间 $\chi$ 中的一个向量，$x \in \chi$，其中 $x_{ij}$ 是 $x_i$ 在第 $j$ 个属性上的取值， $d$ 称为样本 $x_i$ 的 "维数"

- `训练数据 "training data"`: 训练过程中使用的数据

- `训练样本 "training sample"`: 训练数据中的每个样本

- `训练集 "training set"`: 训练样本组成的集合，是训练数据的子集

- `假设 "hypothesis"`: 学得模型对应了关于数据的某种潜在的规律，因此亦称"假设"

- `真相 / 真实 "ground-truth"`: 潜在规律自身 (学习过程就是为了找出或逼近真相)

- `标记 "label"`: 样本训练的"结果"信息 (例如，((色泽=绿;敲声=浑浊)，好瓜) 中的 "好瓜")

- `样例 "example"`: 拥有了标记信息的样本，一般用 $(x_i, y_i)$ 表示第 $i$ 个样例

- `标记空间 "label space" / 输出空间`: $y_i \in \Upsilon$ 是样本 $x_i$ 的标记，$\Upsilon$ 是所有标记的集合，亦称标记空间

- `分类 "classification"`: 预测离散值的学习任务

- `回归 "regression"`: 预测连续值的学习任务

- `二分类 "binary classification"`: 只涉及两个类别的"二分类"学习任务，通常称其中一个类为"正类" (positive class)，另一个类为"反类" (negative class)

- `多分类 "multi-class classification"`: 涉及多个类别的"多分类"学习任务

- `测试 "testing"`: 使用模型进行预测的过程

- `测试样本 "testing sample"`: 被预测的样本

- `聚类 "clustering"`: 将训练集中的样本分成若干组，每组称为一个"簇" (cluster); 自动形成的簇可能对应一些潜在的概念划分

- `监督学习 "supervised learning"`: 代表有分类和回归

- `无监督学习 "unsupervised learning"`: 代表有聚类

- `泛化能力 "generalization"`: 学得模型适用于新样本的能力

- `版本空间 "version space"`: 多个假设与训练集一致，即存在着一个与训练集一致的"假设集合"，称为"版本空间"

- `归纳偏好 "inductive bias"`: 机器学习算法在学习过程中对某种类型假设的偏好 (例如，算法偏好尽可能"特殊"的模型)

- `奥卡姆剃刀 "Occam's razor"`: 若有多个假设与观察一致，则选最简单的那个






