

from keras.engine.topology import Layer
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf

class RoiPoolingConv(Layer):

    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):

        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = K.cast(rois[0, roi_idx, 0], 'int32')
            y = K.cast(rois[0, roi_idx, 1], 'int32')
            w = K.cast(rois[0, roi_idx, 2], 'int32')
            h = K.cast(rois[0, roi_idx, 3], 'int32')

            img_pro = img[:, y:y+h, x:x+w, :]
            rs = tf.image.resize_images(img_pro, (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        return final_output
    
    
    def get_config(self):
        config = { 'pool_size': self.pool_size,
                   'num_rois': self.num_rois }

        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
