from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import jieba
import pandas as pd


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
    data_train, data_test, target_train, target_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=22)
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


def text_count_demo():
    """
    文本特征抽取: CountVectorizer
    """
    data = ["life is short, i like like python",
            "life is to long, i dislike python"]
    # 1. 实例化一个转换器类
    transfer = CountVectorizer()

    # 2. 调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字", transfer.get_feature_names_out())

    return None


def text_chinese_count_demo():
    """
    中文文本特征抽取: CountVectorizer
    """
    data = ["我 爱 北京 天安门",
            "天安门 上 太阳 升"]
    # 1. 实例化一个转换器类
    transfer = CountVectorizer()

    # 2. 调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字", transfer.get_feature_names_out())

    return None


def cut_word(text):
    """
    进行中文分词
    """
    return " ".join(jieba.lcut(text))


def text_chinese_count_demo2():
    """
    中文文本特征抽取，自动分词
    """
    # 1. 将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但是大多数人死在明天晚上，看不到后天的太阳",
            "我们看到的从很远星系来的光是在几百万年之前发出的, 在我们看到的最远的物体的情况下, 光是在80亿年前发出的。这样当我们看宇宙时, 我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的其他事物相联系。 通过联系，你可将想法内化于心，从各种角度看问题，直至找到适合自己的方法。这才是思考的真谛学科之间的界限并没有那么清晰，将知识视为整体，容易将所学的知识与其他知识相联系"]
    split_data = [cut_word(d) for d in data]

    # 2. 实例化一个转换器类
    transfer = CountVectorizer(stop_words=["一种", "只用"])

    # 3. 调用fit_transform()
    data_new = transfer.fit_transform(split_data)
    print("data_new:\n", data_new.toarray())
    print("特征名字", transfer.get_feature_names_out())

    return None


def tfidf_demo():
    """
    用TF-IDF的方法进行文本特征抽取
    """
    # 1. 将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但是大多数人死在明天晚上，看不到后天的太阳",
            "我们看到的从很远星系来的光是在几百万年之前发出的, 在我们看到的最远的物体的情况下, 光是在80亿年前发出的。这样当我们看宇宙时, 我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的其他事物相联系。 通过联系，你可将想法内化于心，从各种角度看问题，直至找到适合自己的方法。这才是思考的真谛学科之间的界限并没有那么清晰，将知识视为整体，容易将所学的知识与其他知识相联系"]
    split_data = [cut_word(d) for d in data]

    # 2. 实例化一个转换器类
    transfer = TfidfVectorizer(stop_words=["一种", "只用"])

    # 3. 调用fit_transform()
    data_new = transfer.fit_transform(split_data)
    print("data_new:\n", data_new.toarray())
    print("特征名字", transfer.get_feature_names_out())

    return None


def minmax_demo():
    """
    归一化
    """
    # 1. 获取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]
    print(data)

    # 2. 实例化一个转换器类
    transfer = MinMaxScaler(feature_range=(2, 3))

    # 3. 调用fit_transform()
    data_new = transfer.fit_transform(data)
    print(data_new)

    return None

def stand_demo():
    """
    标准化
    """
    # 1. 获取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]
    print(data)

    # 2. 实例化一个转换器类
    transfer = StandardScaler()

    # 3. 调用fit_transform()
    data_new = transfer.fit_transform(data)
    print(data_new)

    return None


if __name__ == "__main__":
    # 代码1: sklearn数据集使用
    # datasets_demo()

    # 代码2: 字典特征抽取
    # dict_demo()

    # 代码3: 文本特征抽取
    # text_count_demo()

    # 代码4: 中文文本特征抽取
    # text_chinese_count_demo()

    # 代码5: 中文文本特征抽取, 自动分词
    # text_chinese_count_demo2()

    # 代码6: 中文分词
    # print(cut_word("我爱北京天安门"))

    # 代码7: 用TF-IDF的方法进行文本特征抽取
    # tfidf_demo()

    # 代码8: 归一化
    # minmax_demo()

    # 代码9: 标准化 
    stand_demo()
