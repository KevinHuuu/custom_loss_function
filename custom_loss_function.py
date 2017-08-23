from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.recurrent import SimpleRNN
from keras import backend as K
from keras.layers.wrappers import TimeDistributed
import numpy as np

'''
test for custom_bce()
notice the masking value of y_true should be -1
'''
def custom_bce(y_true, y_pred):
    b = K.not_equal(y_true, -K.ones_like(y_true))
    b = K.cast(b, dtype='float64')
    losses =K.binary_crossentropy(target = y_true, output = y_pred)*b/K.sum(b)
    print('custom_bce loss ',losses.sum().eval())
    return  losses.sum()

train_pred = np.array([0.1, 0.4, 0.3, 0.1, 0.8, 0.2])

# without masking value
train_true = np.array([0, 0, 1, 0, 1, 1])
# with masking value
# train_true = np.array([0,0,1,0,-1,-1])

train_pred_1 = K.variable(value=train_pred, dtype='float64')
train_true_1 = K.variable(value=train_true, dtype='float64')


custom_bce(y_true = train_true_1, y_pred = train_pred_1)
bce_loss = K.binary_crossentropy(target = train_true_1, output = train_pred_1).mean().eval()
print('bce loss ',bce_loss)

'''
test for custom_cce()
notice the masking value of y_true should be -1
'''
def custom_cce(y_true, y_pred):
    b = K.not_equal(y_true, -K.ones_like(y_true))
    b = K.cast(b, dtype='float64')
    losses = K.categorical_crossentropy(target = y_true, output = y_pred) * K.mean(b,axis = -1)
    return  losses

train_pred = np.array([ [ [0.1, 0.2, 0.7], [0.1, 0.4, 0.5], [0.8, 0.1, 0.1] ],
            [ [0.2, 0.5, 0.3], [0.5,0.2,0.3], [0.3,0.1, 0.6] ],
            [ [0.4, 0.1, 0.5], [0.4,0.6, 0.0], [0.5,0.1,0.4] ] ])

# without masking value
train_true = np.array([ [ [0, 0, 1], [1, 0, 0], [1, 0, 0] ],
            [ [1, 0, 0], [0, 1, 0], [0, 0, 1] ],
            [ [0, 1, 0], [0, 0, 1], [0, 1, 0] ] ])

# with masking value
# train_true = np.array([ [ [0, 0, 1], [1, 0, 0], [1, 0, 0] ],
#             [ [1, 0, 0], [0, 1, 0], [-1, -1, -1] ],
#             [ [0, 1, 0], [-1, -1, -1], [-1, -1, -1] ] ])

train_pred_1 = K.variable(value=train_pred, dtype='float64')
train_true_1 = K.variable(value=train_true, dtype='float64')
custom_cce(y_true = train_true_1, y_pred = train_pred_1)
cce_loss = K.categorical_crossentropy(target = train_true_1, output = train_pred_1).mean().eval()
print('cce loss ',cce_loss)
