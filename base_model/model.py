import tensorflow as tf


class Model(object):

    def __init__(self, user_count, item_count, cate_count, cate_list):
        """模型初始化

        Args:
            user_count ([int]): user 数量
            item_count ([int]): item 数量
            cate_count ([int]): category 数量
            cate_list ([list]): category 列表
        """
        self.u = tf.placeholder(tf.int32, [None, ])  # [B]
        # 要预测的next click
        self.i = tf.placeholder(tf.int32, [None, ])  # [B]
        # j 正确的next click
        self.j = tf.placeholder(tf.int32, [None, ])  # [B]
        self.y = tf.placeholder(tf.float32, [None, ])  # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B]
        self.lr = tf.placeholder(tf.float64, [])

        hidden_units = 128

        user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
        # TODO 为什么hidden_units要除2
        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
        item_b = tf.get_variable(
            "item_b",
            [item_count],
            initializer=tf.constant_initializer(0.0)
        )
        cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])
        cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)
        # 将每个user_id初始化，做第一层embedding映射
        u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)
        # 将每个item找到对应的category存储起来
        ic = tf.gather(cate_list, self.i)
        # 将item_id的embedding和对应category的embedding拼接起来
        concat_list = [
            tf.nn.embedding_lookup(item_emb_w, self.i),
            tf.nn.embedding_lookup(cate_emb_w, ic),
        ]
        i_emb = tf.concat(values=concat_list, axis=1)
        # 找到每个item_id对应的映射
        i_b = tf.gather(item_b, self.i)
        # TODO j 是什么含义
        # 找到j对应的category
        jc = tf.gather(cate_list, self.j)
        concat_list_j = [
            tf.nn.embedding_lookup(item_emb_w, self.j),
            tf.nn.embedding_lookup(cate_emb_w, jc),
        ]
        # 找到j对应的embedding， 拼接id和category
        j_emb = tf.concat(
            concat_list_j,
            axis=1
        )
        # 找到j对应的初始化的向量
        j_b = tf.gather(item_b, self.j)
        # 对历史行为， 找到对应的category
        hc = tf.gather(cate_list, self.hist_i)
        hc_concat = [
            tf.nn.embedding_lookup(item_emb_w, self.hist_i),
            tf.nn.embedding_lookup(cate_emb_w, hc),
        ]
        h_emb = tf.concat(hc_concat, axis=2)

        # -- sum begin -------
        # 做mask
        mask = tf.sequence_mask(self.sl, tf.shape(h_emb)[1], dtype=tf.float32)  # [B, T]
        mask = tf.expand_dims(mask, -1)  # [B, T, 1]
        mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]])  # [B, T, H]
        h_emb *= mask  # [B, T, H]
        hist = h_emb
        # 加权求和
        hist = tf.reduce_sum(hist, 1)
        hist = tf.div(hist, tf.cast(tf.tile(tf.expand_dims(self.sl, 1), [1, 128]), tf.float32))
        print(h_emb.get_shape().as_list())
        # -- sum end ---------

        # 做batch normalization
        hist = tf.layers.batch_normalization(inputs=hist)
        hist = tf.reshape(hist, [-1, hidden_units])
        hist = tf.layers.dense(hist, hidden_units)

        u_emb = hist
        # -- fcn begin -------
        # Fully Convolutional Networks
        din_i = tf.concat([u_emb, i_emb], axis=-1)
        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        # 全连接网络
        # 第1层 80
        d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1')
        # 第2层 40
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
        # 输出预测结果
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')
        # 将user embedding和真实结果拼接起来
        din_j = tf.concat([u_emb, j_emb], axis=-1)
        # batch normalization
        din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
        # 这个再预测一遍
        d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
        d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
        d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)
        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1])
        # TODO 这个是什么含义
        x = i_b - j_b + d_layer_3_i - d_layer_3_j  # [B]
        self.logits = i_b + d_layer_3_i
        u_emb_all = tf.expand_dims(u_emb, 1)
        u_emb_all = tf.tile(u_emb_all, [1, item_count, 1])
        # logits for all item:
        all_emb = tf.concat([
            item_emb_w,
            tf.nn.embedding_lookup(cate_emb_w, cate_list)
            ], axis=1)
        all_emb = tf.expand_dims(all_emb, 0)
        all_emb = tf.tile(all_emb, [512, 1, 1])
        din_all = tf.concat([u_emb_all, all_emb], axis=-1)
        din_all = tf.layers.batch_normalization(inputs=din_all, name='b1', reuse=True)
        d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3', reuse=True)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, item_count])
        self.logits_all = tf.sigmoid(item_b + d_layer_3_all)
        # -- fcn end -------

        self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
        self.score_i = tf.sigmoid(i_b + d_layer_3_i)
        self.score_j = tf.sigmoid(j_b + d_layer_3_j)
        self.score_i = tf.reshape(self.score_i, [-1, 1])
        self.score_j = tf.reshape(self.score_j, [-1, 1])
        self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)
        print(self.p_and_n.get_shape().as_list())

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = \
            tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step+1)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
            )

        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, l):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.y: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
            self.lr: l,
            })
        return loss

    def eval(self, sess, uij):
        """计算AUC

        Args:
            sess ([tf]): sesstion
            uij ([iterator]): 输入行为序列

        Returns:
            u_auc [int]: auc指标
            socre_p_and_n [list]: 正负样本的得分
        """
        u_auc, socre_p_and_n = sess.run([self.mf_auc, self.p_and_n], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.j: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
            })
        return u_auc, socre_p_and_n

    def test(self, sess, uid, hist_i, sl):
        """[summary]

        Args:
            sess ([type]): [description]
            uid ([type]): [description]
            hist_i ([type]): [description]
            sl ([type]): [description]

        Returns:
            [type]: [description]
        """
        feed_dict = {
            self.u: uid,
            self.hist_i: hist_i,
            self.sl: sl,
        }
        result = sess.run(self.logits_all, feed_dict=feed_dict)
        return result

    def save(self, sess, path):
        """模型存储

        Args:
            sess ([tf]): [session]
            path ([string]): [存储路径]
        """
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        """恢复路径

        Args:
            sess ([tf]): [session]
            path ([string]): [恢复的路径]
        """
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
