import tensorflow as tf


class EmbeddingGroupLayer(tf.keras.layers.Layer):
    def __init__(self, shape_list, emb_dim):
        super().__init__()
        self.start = list()
        self.end = list()
        self.n_var = len(shape_list)

        for i, element in enumerate(shape_list):
            if i == 0:
                self.start.append(0)
                self.end.append(element)
            else:
                self.start.append(self.end[i - 1])
                self.end.append(self.start[i] + element)
        self.var_list = list()
        for element in shape_list:
            init = tf.random_normal_initializer()
            self.var_list.append(tf.Variable(
                initial_value=init(shape=(element, emb_dim), dtype='float32'),
                trainable=True))

    def call(self, inputs):
        results = [inputs[:, self.start[i]:self.end[i]] @ self.var_list[i] for i in range(self.n_var)]
        return tf.concat(results, axis=1)
