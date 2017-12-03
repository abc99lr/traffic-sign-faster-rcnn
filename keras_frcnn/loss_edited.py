from keras import backend as K
from keras.objectives import categorical_crossentropy

if K.image_dim_ordering() == 'tf':
	import tensorflow as tf

"""

Four Losses: 
1. RPN classify loss: object / not object 
2. RPN regress box coordinates 
3. Final classification score (object classes)
4. Final box coordinates 

For RPN losses: 
L({pi}, {ti}) = Loss_cls + Loss_reg = SUM(L_cls(pi, pi*)) / N_cls + 
									  lambda * SUM(L_reg(ti, ti*)) / N_reg, 
where L_cls is log loss and L_reg is robust loss (smooth L1 in this case).

"""

def rpn_loss_regr(num_anchors):
	def rpn_loss_regr_fixed_num(y_true, y_pred):

		d = y_true[:, :, :, 4 * num_anchors:] - y_pred
		p_star = K.cast(K.less_equal(K.abs(d), 1.0), tf.float32)

		n_reg = K.sum(1e-4 + y_true[:, :, :, :4 * num_anchors])

		if p_star == 1: 
			l1_smooth = 0.5 * d * d
		else: 
			l1_smooth = K.abs(d) - 0.5

		tot_reg_loss = K.sum(y_true[:, :, :, :4 * num_anchors] * l1_smooth)

		rpn_reg_loss = tot_reg_loss / n_reg
		
		return rpn_reg_loss

	return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
	def rpn_loss_cls_fixed_num(y_true, y_pred):
		
		n_cls = K.sum(1e-4 + y_true[:, :, :, :num_anchors])

		tot_cls_loss = K.sum(y_true[:, :, :, :num_anchors] * \
			K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:]))

		rpn_cls_loss = tot_cls_loss / n_cls

		return rpn_cls_loss

	return rpn_loss_cls_fixed_num

"""

For classification losses: 
Regression Loss is the bounding coordinates 
Classification Loss is the final cross-entropy score 

"""

def class_loss_regr(num_classes):
	def class_loss_regr_fixed_num(y_true, y_pred):
		d = y_true[:, :, 4 * num_classes:] - y_pred
		p_star = K.cast(K.less_equal(K.abs(d), 1.0), tf.float32)

		n_reg = K.sum(1e-4 + y_true[:, :, :, :4 * num_anchors])

		if p_star == 1: 
			l1_smooth = 0.5 * d * d
		else: 
			l1_smooth = K.abs(d) - 0.5
			
		tot_reg_loss = K.sum(y_true[:, :, :, :4 * num_anchors] * l1_smooth)

		class_reg_loss = tot_reg_loss / n_reg

		return class_reg_loss
	return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
	class_cls_loss = K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))

	return class_cls_loss
