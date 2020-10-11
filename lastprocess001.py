import pandas as pd
import random
import os
pse_label_df = pd.read_csv('../combine/submit/sub_8223用于伪标签.csv')
pse_label_df.head()

test_df = pd.read_csv('test.csv')
test_df = test_df.loc[:,['id','sent0']]
test_df.head(3)


def split_train(train_df, train_label_df):
    random.seed(42)
    split_num = 12
    train_df = train_df.merge(train_label_df, on='id', how='left')
    train_df['label'] = train_df['label'].fillna(-1)
    train_df = train_df[train_df['label'] != -1]
    train_df['label'] = train_df['label'].astype(int)
    train_df['sent0'] = train_df['sent0'].fillna('.')

    index = set(range(train_df.shape[0]))
    K_fold = []
    for i in range(split_num):
        if i == split_num - 1:
            tmp = index
        else:
            tmp = random.sample(index, int(1.0 / split_num * train_df.shape[0]))
        index = index - set(tmp)
        print("Number:", len(tmp))
        K_fold.append(tmp)

    for i in range(split_num):
        print("Fold", i)
        data_dir = "data_pse_10fold/data_pse_42_{}".format(i)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        dev_index = list(K_fold[i])
        train_df.iloc[dev_index].to_csv((data_dir + "/train.csv").format(i))

split_train(test_df, pse_label_df)



# 伪标签和原训练集拼接

import pandas as pd
import os
def pse_merge_train(train_id, pse_id):
#     train_id, pse_id = 3,3
    train_dir = "data_StratifiedKFold_42/data_origin_{}/".format(train_id)
    pse_dir = "data_pse_10fold/data_pse_42_{}/".format(pse_id)
    train_pse_dir = "origin_data_pse/origin_data_train_pse{}/".format(train_id,pse_id)
    os.system("cp -r "+train_dir+" "+train_pse_dir)

    train_df = pd.read_csv(train_dir+'train.csv')
    pse_df = pd.read_csv(pse_dir+'train.csv')

    rst = pd.concat([train_df, pse_df],ignore_index=True)
    rst.loc[:,['id','sent0','label']].to_csv(train_pse_dir + "train.csv")

rst = pse_merge_train(3, 3)
rst = pse_merge_train(0, 0)
rst = pse_merge_train(1, 1)
rst = pse_merge_train(2, 2)
rst = pse_merge_train(4, 4)