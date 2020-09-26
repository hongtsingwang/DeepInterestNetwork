# coding=utf-8

import json
import pickle
import pandas as pd


def to_df(file_path):
    """将数据转换为dataframe

    Args:
        file_path ([string]): [输入文件路径]

    Returns:
        [df]: [输出的转化过的dataframe]
    """
    with open(file_path, 'r') as fin:
        df = {}
        cnt = 0
        for line in fin:
            df[cnt] = json.loads(line)
            cnt += 1
        # 读进来一个dict是作为一行存储的
        df = pd.DataFrame.from_dict(df, orient='index')
        return df


# 将评论存储起来
reviews_df = to_df('../raw_data/reviews_Electronics_5.json')
with open('../raw_data/reviews.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

meta_df = to_df('../raw_data/meta_Electronics.json')
# 将原始meta文件中asin能对上的数据存储起来， 重新编索引
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)
with open('../raw_data/meta.pkl', 'wb') as f:
    pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)
