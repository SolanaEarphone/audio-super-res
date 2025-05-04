import sys
from typing import List, Tuple, Optional, Any

import numpy as np
import tensorflow as tf
from scipy import interpolate
from tensorflow.python.keras import backend as K
from keras.layers import (
    Concatenate, Add, MaxPooling1D, Activation, Dropout,
    Conv1D, LSTM, BatchNormalization, LeakyReLU
)
from keras.initializers import RandomNormal, Orthogonal

from .model import Model, default_opt
from .layers.subpixel import SubPixel1D

# Constants
DILATION_RATE = 2
DEFAULT_FILTERS = [128, 256, 512, 512, 512, 512, 512, 512]
DEFAULT_BLOCKS = [128, 64, 32, 16, 8]
DEFAULT_FILTER_SIZES = [65, 33, 17, 9, 9, 9, 9, 9, 9]
DEFAULT_LEAKY_RELU_ALPHA = 0.2
DEFAULT_DROPOUT_RATE = 0.5


class AudioTfilm(Model):
    """Audio super-resolution model using temporal feature learning."""

    def __init__(
        self,
        from_ckpt: bool = False,
        n_dim: Optional[int] = None,
        r: int = 2,
        pool_size: int = 4,
        strides: int = 4,
        opt_params: dict = default_opt,
        log_prefix: str = './run'
    ):
        """Initialize the AudioTfilm model.

        Args:
            from_ckpt: Whether to load from checkpoint
            n_dim: Input dimension
            r: Upsampling ratio
            pool_size: Pooling size
            strides: Stride size
            opt_params: Optimization parameters
            log_prefix: Log directory prefix
        """
        self.r = r
        self.pool_size = pool_size
        self.strides = strides
        super().__init__(
            from_ckpt=from_ckpt,
            n_dim=n_dim,
            r=r,
            opt_params=opt_params,
            log_prefix=log_prefix
        )

    def create_model(self, n_dim: int, r: int) -> tf.Tensor:
        """Create the model architecture.

        Args:
            n_dim: Input dimension
            r: Upsampling ratio

        Returns:
            Model output tensor
        """
        X, _, _ = self.inputs
        K.set_session(self.sess)

        with tf.compat.v1.name_scope('generator'):
            x = X
            L = self.layers
            downsampling_l = []

            # Downsampling layers
            for l, nf, fs in zip(range(L), DEFAULT_FILTERS, DEFAULT_FILTER_SIZES):
                x = self._create_downsampling_block(x, l, nf, fs)
                downsampling_l.append(x)

            # Bottleneck layer
            x = self._create_bottleneck_block(x, L)

            # Upsampling layers
            x = self._create_upsampling_blocks(x, L, downsampling_l)

            # Final convolution layer
            x = self._create_final_conv_block(x, X)

        return x

    def _create_downsampling_block(
        self,
        x: tf.Tensor,
        layer_idx: int,
        n_filters: int,
        filter_size: int
    ) -> tf.Tensor:
        """Create a downsampling block.

        Args:
            x: Input tensor
            layer_idx: Layer index
            n_filters: Number of filters
            filter_size: Filter size

        Returns:
            Processed tensor
        """
        with tf.compat.v1.name_scope(f'downsc_conv{layer_idx}'):
            # Convolution
            x = Conv1D(
                filters=n_filters,
                kernel_size=filter_size,
                dilation_rate=DILATION_RATE,
                activation=None,
                padding='same',
                kernel_initializer=Orthogonal()
            )(x)
            
            # Pooling
            x = MaxPooling1D(
                pool_size=self.pool_size,
                padding='valid',
                strides=self.strides
            )(x)
            x = LeakyReLU(DEFAULT_LEAKY_RELU_ALPHA)(x)

            # Normalization
            nb = 128 / (2**layer_idx)
            x_norm = self._make_normalizer(x, n_filters, nb)
            x = self._apply_normalizer(x, x_norm, n_filters, nb)

            print('D-Block: ', x.get_shape())
            return x

    def _create_bottleneck_block(self, x: tf.Tensor, layer_idx: int) -> tf.Tensor:
        """Create the bottleneck block.

        Args:
            x: Input tensor
            layer_idx: Layer index

        Returns:
            Processed tensor
        """
        with tf.compat.v1.name_scope('bottleneck_conv'):
            x = Conv1D(
                filters=DEFAULT_FILTERS[-1],
                kernel_size=DEFAULT_FILTER_SIZES[-1],
                dilation_rate=DILATION_RATE,
                activation=None,
                padding='same',
                kernel_initializer=Orthogonal()
            )(x)
            
            x = MaxPooling1D(
                pool_size=self.pool_size,
                padding='valid',
                strides=self.strides
            )(x)
            x = Dropout(rate=DEFAULT_DROPOUT_RATE)(x)
            x = LeakyReLU(DEFAULT_LEAKY_RELU_ALPHA)(x)

            nb = 128 / (2**layer_idx)
            x_norm = self._make_normalizer(x, DEFAULT_FILTERS[-1], nb)
            x = self._apply_normalizer(x, x_norm, DEFAULT_FILTERS[-1], nb)
            
            return x

    def _create_upsampling_blocks(
        self,
        x: tf.Tensor,
        layer_idx: int,
        downsampling_layers: List[tf.Tensor]
    ) -> tf.Tensor:
        """Create upsampling blocks.

        Args:
            x: Input tensor
            layer_idx: Layer index
            downsampling_layers: List of downsampling layer outputs

        Returns:
            Processed tensor
        """
        for l, nf, fs, l_in in reversed(list(zip(
            range(layer_idx),
            DEFAULT_FILTERS,
            DEFAULT_FILTER_SIZES,
            downsampling_layers
        ))):
            with tf.compat.v1.name_scope(f'upsc_conv{l}'):
                x = Conv1D(
                    filters=2*nf,
                    kernel_size=fs,
                    dilation_rate=DILATION_RATE,
                    activation=None,
                    padding='same',
                    kernel_initializer=Orthogonal()
                )(x)

                x = Dropout(rate=DEFAULT_DROPOUT_RATE)(x)
                x = Activation('relu')(x)
                x = SubPixel1D(x, r=2)

                nb = 128 / (2**l)
                x_norm = self._make_normalizer(x, nf, nb)
                x = self._apply_normalizer(x, x_norm, nf, nb)
                x = Concatenate()([x, l_in])
                print('U-Block: ', x.get_shape())

        return x

    def _create_final_conv_block(self, x: tf.Tensor, X: tf.Tensor) -> tf.Tensor:
        """Create the final convolution block.

        Args:
            x: Input tensor
            X: Original input tensor

        Returns:
            Processed tensor
        """
        with tf.compat.v1.name_scope('lastconv'):
            x = Conv1D(
                filters=2,
                kernel_size=9,
                activation=None,
                padding='same',
                kernel_initializer=RandomNormal(stddev=1e-3)
            )(x)
            x = SubPixel1D(x, r=2)
            return Add()([x, X])

    def _make_normalizer(self, x_in: tf.Tensor, n_filters: int, n_block: int) -> tf.Tensor:
        """Create a normalizer using LSTM.

        Args:
            x_in: Input tensor
            n_filters: Number of filters
            n_block: Block size

        Returns:
            Normalized tensor
        """
        x_in_down = MaxPooling1D(pool_size=int(n_block), padding='valid')(x_in)
        return LSTM(units=n_filters, return_sequences=True)(x_in_down)

    def _apply_normalizer(
        self,
        x_in: tf.Tensor,
        x_norm: tf.Tensor,
        n_filters: int,
        n_block: int
    ) -> tf.Tensor:
        """Apply normalization to input tensor.

        Args:
            x_in: Input tensor
            x_norm: Normalization tensor
            n_filters: Number of filters
            n_block: Block size

        Returns:
            Normalized tensor
        """
        x_shape = tf.shape(input=x_in)
        n_steps = x_shape[1] / int(n_block)

        x_in = tf.reshape(x_in, shape=(-1, n_steps, int(n_block), n_filters))
        x_norm = tf.reshape(x_norm, shape=(-1, n_steps, 1, n_filters))
        x_out = x_norm * x_in
        return tf.reshape(x_out, shape=x_shape)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data.

        Args:
            X: Input data

        Returns:
            Predicted output
        """
        assert len(X) == 1
        x_sp = spline_up(X, self.r)
        x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(self.layers+1)))]
        X = x_sp.reshape((1, len(x_sp), 1))
        feed_dict = self.load_batch((X, X), train=False)
        return self.sess.run(self.predictions, feed_dict=feed_dict)


def spline_up(x_lr: np.ndarray, r: int) -> np.ndarray:
    """Upsample input using spline interpolation.

    Args:
        x_lr: Low resolution input
        r: Upsampling ratio

    Returns:
        Upsampled output
    """
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f)

    return x_sp
