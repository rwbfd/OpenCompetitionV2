import tensorflow as tf
from tensorflow_addons.activations import sparsemax
from .embedding_tf import EmbeddingGroupLayer
import os
from typing import List, Text
import pickle


class GhostBatchNormalization(tf.keras.Model):
    def __init__(
            self, momentum: float = 0.9, epsilon: float = 1e-5, divide=4
    ):
        super(GhostBatchNormalization, self).__init__()
        self.divide = divide
        self.bn = [tf.keras.layers.BatchNormalization(momentum=momentum) for _ in range(divide)]

    def call(self, x):
        x_split = tf.split(x, num_or_size_splits=self.divide, axis=0)
        normed = [self.bn[i](x_split[i]) for i in range(self.divide)]
        return tf.concat(normed, axis=0)


def glu(x, n_units=None):
    """Generalized linear unit nonlinear activation."""
    return x[:, :n_units] * tf.nn.sigmoid(x[:, n_units:])


class FeatureBlock(tf.keras.Model):
    def __init__(
            self,
            feature_dim: int,
            apply_glu: bool = True,
            bn_momentum: float = 0.9,
            virtual_batch_size: int = 4,
            fc: tf.keras.layers.Layer = None,
            epsilon: float = 1e-5,
    ):
        super(FeatureBlock, self).__init__()
        self.apply_gpu = apply_glu
        self.feature_dim = feature_dim
        units = feature_dim * 2 if apply_glu else feature_dim

        self.fc = tf.keras.layers.Dense(units, use_bias=False) if fc is None else fc
        self.bn = GhostBatchNormalization(
            virtual_batch_size=virtual_batch_size, momentum=bn_momentum
        )

    def call(self, x, training: bool = None):
        x = self.fc(x)
        x = self.bn(x, training=training)
        if self.apply_gpu:
            return glu(x, self.feature_dim)
        return x


class AttentiveTransformer(tf.keras.Model):
    def __init__(self, feature_dim: int, bn_momentum: float, virtual_batch_size: int):
        super(AttentiveTransformer, self).__init__()
        self.block = FeatureBlock(
            feature_dim,
            bn_momentum=bn_momentum,
            virtual_batch_size=virtual_batch_size,
            apply_glu=False,
        )

    def call(self, x, prior_scales, training=None):
        x = self.block(x, training=training)
        return sparsemax(x * prior_scales)


class FeatureTransformer(tf.keras.Model):
    def __init__(
            self,
            feature_dim: int,
            fcs: List[tf.keras.layers.Layer] = [],
            n_total: int = 4,
            n_shared: int = 2,
            bn_momentum: float = 0.9,
            virtual_batch_size: int = 4,
    ):
        super(FeatureTransformer, self).__init__()
        self.n_total, self.n_shared = n_total, n_shared

        kargs = {
            "feature_dim": feature_dim,
            "bn_momentum": bn_momentum,
            "virtual_batch_size": virtual_batch_size,
        }

        # build blocks
        self.blocks: List[FeatureBlock] = []
        for n in range(n_total):
            # some shared blocks
            if fcs and n < len(fcs):
                self.blocks.append(FeatureBlock(**kargs, fc=fcs[n]))
            # build new blocks
            else:
                self.blocks.append(FeatureBlock(**kargs))

    def call(
            self, x: tf.Tensor, training: bool = None
    ) -> tf.Tensor:
        x = self.blocks[0](x, training=training)
        for n in range(1, self.n_total):
            x = x * tf.sqrt(0.5) + self.blocks[n](x, training=training)
        return x

    @property
    def shared_fcs(self):
        return [self.blocks[i].fc for i in range(self.n_shared)]


from typing import List, Tuple


class TabNet(tf.keras.Model):
    def __init__(
            self,
            level_list: List,
            emb_dim: int,
            num_features: int,
            feature_dim: int,
            output_dim: int,
            n_step: int = 1,
            n_total: int = 2,
            n_shared: int = 1,
            relaxation_factor: float = 1.5,
            bn_epsilon: float = 1e-5,
            bn_momentum: float = 0.7,
            virtual_batch_size: int = 4,
    ):
        """TabNet
        Will output a vector of size output_dim.
        Args:
            num_features (int): Number of features.
            feature_dim (int): Embedding feature dimention to use.
            output_dim (int): Output dimension.
            feature_columns (List, optional): If defined will add a DenseFeatures layer first. Defaults to None.
            n_step (int, optional): Total number of steps. Defaults to 1.
            n_total (int, optional): Total number of feature transformer blocks. Defaults to 4.
            n_shared (int, optional): Number of shared feature transformer blocks. Defaults to 2.
            relaxation_factor (float, optional): >1 will allow features to be used more than once. Defaults to 1.5.
            bn_epsilon (float, optional): Batch normalization, epsilon. Defaults to 1e-5.
            bn_momentum (float, optional): Batch normalization, momentum. Defaults to 0.7.

        """
        super(TabNet, self).__init__()
        self.embedding_layer = EmbeddingGroupLayer(level_list, emb_dim)
        self.output_dim, self.num_features = output_dim, num_features
        self.n_step, self.relaxation_factor = n_step, relaxation_factor

        # ? Switch to Ghost Batch Normalization
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, epsilon=bn_epsilon
        )

        kargs = {
            "feature_dim": feature_dim + output_dim,
            "n_total": n_total,
            "n_shared": n_shared,
            "bn_momentum": bn_momentum,
            "virtual_batch_size": virtual_batch_size,
        }

        # first feature transformer block is built first to get the shared blocks
        self.feature_transforms: List[FeatureTransformer] = [
            FeatureTransformer(**kargs)
        ]
        self.attentive_transforms: List[AttentiveTransformer] = []
        for i in range(n_step):
            self.feature_transforms.append(
                FeatureTransformer(**kargs, fcs=self.feature_transforms[0].shared_fcs)
            )
            self.attentive_transforms.append(
                AttentiveTransformer(num_features, bn_momentum, virtual_batch_size)
            )

    def call(
            self, features: tf.Tensor, training: bool = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        features = self.embedding_layer(features)
        bs = tf.shape(features)[0]
        out_agg = tf.zeros((bs, self.output_dim))
        prior_scales = tf.ones((bs, self.num_features))
        masks = []

        features = self.bn(features, training=training)
        masked_features = features

        total_entropy = 0.0

        for step_i in range(self.n_step + 1):
            x = self.feature_transforms[step_i](
                masked_features, training=training
            )

            if step_i > 0:
                out = tf.keras.activations.relu(x[:, : self.output_dim])
                out_agg += out

            # no need to build the features mask for the last step
            if step_i < self.n_step:
                x_for_mask = x[:, self.output_dim:]

                mask_values = self.attentive_transforms[step_i](
                    x_for_mask, prior_scales, training=training
                )

                # relaxation factor of 1 forces the feature to be only used once.
                prior_scales *= self.relaxation_factor - mask_values

                masked_features = tf.multiply(mask_values, features)

                # entropy is used to penalize the amount of sparsity in feature selection
                total_entropy = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.multiply(mask_values, tf.math.log(mask_values + 1e-10)),
                        axis=1,
                    )
                )

                masks.append(tf.expand_dims(tf.expand_dims(mask_values, 0), 3))

        loss = total_entropy / self.n_step

        return out_agg, loss, masks


class TabNetClassifier(tf.keras.Model):
    def __init__(
            self,
            level_list: List,
            emb_dim: int,
            num_features: int,
            feature_dim: int,
            output_dim: int,
            n_classes: int,
            feature_columns: List = None,
            n_step: int = 1,
            n_total: int = 4,
            n_shared: int = 2,
            relaxation_factor: float = 1.5,
            sparsity_coefficient: float = 1e-5,
            bn_epsilon: float = 1e-5,
            bn_momentum: float = 0.7,
            virtual_batch_size: int = 4,
            dp: float = None,
            **kwargs
    ):
        super(TabNetClassifier, self).__init__()

        self.configs = {
            "num_features": num_features,
            "feature_dim": feature_dim,
            "output_dim": output_dim,
            "n_classes": n_classes,
            "n_step": n_step,
            "n_total": n_total,
            "n_shared": n_shared,
            "relaxation_factor": relaxation_factor,
            "sparsity_coefficient": sparsity_coefficient,
            "bn_epsilon": bn_epsilon,
            "bn_momentum": bn_momentum,
            "virtual_batch_size": virtual_batch_size,
            "dp": dp,
        }
        for k, v in kwargs.items():
            self.configs[k] = v

        self.sparsity_coefficient = sparsity_coefficient

        self.model = TabNet(
            level_list=level_list,
            emb_dim=emb_dim,
            num_features=num_features,
            feature_dim=feature_dim,
            output_dim=output_dim,
            n_step=n_step,
            relaxation_factor=relaxation_factor,
            bn_epsilon=bn_epsilon,
            bn_momentum=bn_momentum,
            virtual_batch_size=virtual_batch_size,
        )
        self.dp = tf.keras.layers.Dropout(dp) if dp is not None else dp
        self.head = tf.keras.layers.Dense(n_classes, activation=None, use_bias=False)

    def call(self, x, training: bool = None):
        out, sparse_loss, _ = self.model(x, training=training)
        if self.dp is not None:
            out = self.dp(out, training=training)
        y = self.head(out, training=training)

        if training:
            self.add_loss(-self.sparsity_coefficient * sparse_loss)

        return y

    def get_config(self):
        return self.configs

    def save_to_directory(self, path_to_folder: Text):
        self.save_weights(os.path.join(path_to_folder, "ckpt"), overwrite=True)
        with open(os.path.join(path_to_folder, "configs.pickle"), "wb") as f:
            pickle.dump(self.configs, f)

    @classmethod
    def load_from_directory(cls, path_to_folder: Text):
        with open(os.path.join(path_to_folder, "configs.pickle"), "rb") as f:
            configs = pickle.load(f)
        model: tf.keras.Model = cls(**configs)
        model.build((None, configs["num_features"]))
        load_status = model.load_weights(os.path.join(path_to_folder, "ckpt"))
        load_status.expect_partial()
        return model
