# coding=utf-8
import random
import pickle

random.seed(1234)

with open('../raw_data/remap.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count, example_count = pickle.load(f)


train_set = []
test_set = []
for reviewerID, hist in reviews_df.groupby('reviewerID'):
    # 按照reviewID 进行group
    # TODO asin 什么东西不知道，得基于真实数据才知道
    # tolist表名， hist应该是个numpy
    pos_list = hist['asin'].tolist()

    def gen_neg():
        """生产负样本

        Returns:
            [type]: [description]
        """
        # 随机到一个不在pos_list中的数字
        neg = pos_list[0]
        while neg in pos_list:
            neg = random.randint(0, item_count-1)
        return neg
    # 生成负样本集合， 正负样本集合比例1:1
    neg_list = [gen_neg() for i in range(len(pos_list))]

    for i in range(1, len(pos_list)):
        # 取得i前面的所有样本作为已知的样本
        hist = pos_list[:i]
        # 在获得最后一个样本之前
        if i != len(pos_list) - 1:
            # 评论者ID， 历史，下一个评论的内容， 1代表正样本
            train_set.append((reviewerID, hist, pos_list[i], 1))
            # 同上， 0 代表加入一个负样本
            train_set.append((reviewerID, hist, neg_list[i], 0))
        else:
            # 对最后一个样本， 设置label， 作为test集合中的元素
            label = (pos_list[i], neg_list[i])
            test_set.append((reviewerID, hist, label))

# 打乱训练集和测试集的顺序
random.shuffle(train_set)
random.shuffle(test_set)

# 务必保证每个用户至少有一个作为测试集
assert len(test_set) == user_count

with open('dataset.pkl', 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
