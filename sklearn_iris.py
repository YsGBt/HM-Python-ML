from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

def datasets_demo():
    """
    sklearn数据集使用
    """
    # 获取数据集
    iris = load_iris()
    print("鸢尾花数据集:\n", iris)
    print("查看数据集描述: \n", iris["DESCR"])
    print("查看特征值的名字:\n", iris.feature_names)
    print("查看特征值:\n", iris.data, iris.data.shape)
    print("查看目标值:\n", iris.target, iris.target.shape)

    # 数据集划分 (默认test_size=0.25)
    data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集特征值:\n", data_train, data_train.shape)

    return None

def dict_demo():
    """
    字典特征抽取
    """
    data = [{'city': '北京', 'temperature': 100},
            {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]
    
    # 1. 实例化一个转换器类
    transformer = DictVectorizer(sparse=False)

    # 2. 调用fit_transform()
    data_new = transformer.fit_transform(data)
    print("data_new:\n", data_new)
    print("特征名字", transformer.get_feature_names_out())

    return None

if __name__ == "__main__":
    # 代码1: sklearn数据集使用
    # datasets_demo()

    # 代码2: 字典特征抽取
    dict_demo()
