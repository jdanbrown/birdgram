import caffe
import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint

from bubo.util import caffe_root, shell, plt_show

model_id      = 'bvlc_reference_caffenet' # A variant of alexnet
#model_id     = 'bvlc_googlenet' # TODO
#model_id     = 'bvlc_alexnet'
model_def     = '%(caffe_root)s/models/%(model_id)s/deploy.prototxt'         % locals()
model_weights = '%(caffe_root)s/models/%(model_id)s/%(model_id)s.caffemodel' % locals()

if os.path.isfile(model_weights):
    print 'Found model[%(model_id)s]' % locals()
else:
    print 'Downloading model[%(model_id)s]...' % locals()
    shell('%(caffe_root)s/scripts/download_model_binary.py %(caffe_root)s/models/%(model_id)s' % locals())

caffe.set_mode_cpu()
net = caffe.Net(
    model_def,      # defines the structure of the model
    model_weights,  # contains the trained weights
    caffe.TEST,     # use test mode (e.g., don't perform dropout)
)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('%(caffe_root)s/python/caffe/imagenet/ilsvrc_2012_mean.npy' % locals())
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
# with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(
    50,        # batch size
    3,         # 3-channel (BGR) images
    227, 227,  # image size is 227x227
    #224, 224, # image size is 227x227 [TODO Make googlenet work]
)

# Pick your image
image_path = '%(caffe_root)s/examples/images/cat.jpg' % locals()
#image_path = 'spectrograms/PC1_20090705_070000_0040.bmp'
image = caffe.io.load_image(image_path)
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)
plt_show()

###

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image
# perform classification (slow, ~secs)
output = net.forward()
output_prob = output['prob'][0] # the output probability vector for the first image in the batch
print 'predicted class is:', output_prob.argmax()

# load ImageNet labels
labels_file = '%(caffe_root)s/data/ilsvrc12/synset_words.txt' % locals()
if not os.path.exists(labels_file):
    shell('%(caffe_root)s/data/ilsvrc12/get_ilsvrc_aux.sh' % locals())
labels = np.loadtxt(labels_file, str, delimiter='\t')
print 'probabilities and labels:'
top_inds = output_prob.argsort()[::-1][:20] # top k predictions from softmax output
pprint(zip(output_prob[top_inds], labels[top_inds]))

# For each layer, show the shapes of the activations and params:
#   blob       activations  (batch_size, channel_dim, height, width) -- typically, but not always
#   params[0]  weights      (output_channels, input_channels, filter_height, filter_width)
#   params[1]  biases       (output_channels,)
def show_shape(shape, name, fields):
    return '%s(%s)' % (name, ', '.join(['%s=%s' % (d,s) for (s,d) in zip(shape, fields)]))
for layer_name, blob in net.blobs.iteritems():
    [param_weights, param_biases] = net.params.get(layer_name, [None, None])
    print '%-52s %-29s %-31s %s' % (
        layer_name,
        show_shape(blob.data.shape, 'act', ('b', 'c', 'h', 'w')),
        param_weights and show_shape(param_weights.data.shape, 'weight', ('o', 'i', 'h', 'w')) or '',
        param_biases  and show_shape(param_biases.data.shape,  'bias',   ('o', 'i', 'h', 'w')) or '',
    )

###

def norm(data):
    return (data - data.min()) / (data.max() - data.min())

def tile(w, data):

    n = data.shape[0]
    w = int(np.ceil(w(n)))
    h = int(np.ceil(n / float(w)))
    padding = (
        ((0, h*w-n), (0, 1), (0, 1))  # add some space between filters
        + ((0, 0),) * (data.ndim - 3) # don't pad the last dimension (if there is one)
    )
    data = np.pad(data, padding, mode='constant', constant_values=1)# pad with ones (white)

    # tile the filters into an image
    data = data.reshape((h,w) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((h * data.shape[1], w * data.shape[3]) + data.shape[4:])
    return data

def vis_pretiled(data):
    plt.imshow(data)
    plt.axis('off')
    plt_show() # So we can have multiple plt.imshow's from the same cell

def vis_square(data):
    'data: an array of shape (n, height, width) or (n, height, width, 3)'
    vis_pretiled(tile(np.sqrt, norm(data)))

def tile_tiles(data):
    print 'tile_tile: data.shape[%s]' % (data.shape,)

    # TODO Make plots fill the figure:
    # - Twiddle 400, np.sqrt
    # - Use figure_wh_ratio = 10.5 / 12.24

    # Calculate our various widths (w_*)
    w_pixels_per_figure = 400
    (n_out, n_in, h_pixels_per_filter, w_pixels_per_filter) = data.shape
    w_filters_per_tile = int(np.ceil(np.sqrt(n_in)))
    w_tiles_per_figure = w_pixels_per_figure / w_filters_per_tile / w_pixels_per_filter

    return tile(
        lambda n: w_tiles_per_figure,
        np.array(map(
            lambda xs: tile(lambda n: w_filters_per_tile, xs),
            np.array(map(norm, data)),
        )),
    )

#vis_pretiled(tile_tiles(net.params['conv1'][0].data))

###

#vis_pretiled(tile(lambda n: 7, net.params['conv2'][0].data[0]))
vis_pretiled(tile_tiles(net.params['conv1'][0].data))
vis_pretiled(tile_tiles(net.params['conv2'][0].data))
vis_pretiled(tile_tiles(net.params['conv3'][0].data))
vis_pretiled(tile_tiles(net.params['conv4'][0].data))
vis_pretiled(tile_tiles(net.params['conv5'][0].data))

###

for layer_name, blob in net.blobs.iteritems():

    [param_weights, param_biases] = net.params.get(layer_name, [None, None])
    print '\nlayer[%s]\n- blob.data.shape[%s]\n- param_weights.data.shape[%s]\n- param_biases.data.shape[%s]\n' % (
        layer_name,
        blob.data.shape,
        param_weights and param_weights.data.shape,
        param_biases  and param_biases.data.shape,
    )

    try:
        vis_square(blob.data[0])
    except Exception, e:
        print '    Error:', e

    if not param_weights:
        print 'layer[%s], param_weights[%s]' % (layer_name, param_weights)
    else:
        print 'layer[%s], param_weights.data.shape[%s]' % (layer_name, param_weights.data.shape)
        try:
            vis_square(param_weights.data.transpose(0, 2, 3, 1)) # e.g. (96,3,11,11) -> (96,11,11,3)
        except Exception, e:
            print '    Error:', e

###

# The first fully connected layer, `fc6` (rectified)
# - We show the output values and the histogram of the positive values
plt.subplot(3, 1, 1); plt.plot(net.blobs['fc6'].data[0].flat)
plt.subplot(3, 1, 2); plt.hist(net.blobs['fc6'].data[0].flat[net.blobs['fc6'].data[0].flat > 0], bins=100); None
# The final probability output, `prob`
# - Note the cluster of strong predictions; the labels are sorted semantically
# - The top peaks correspond to the top predicted labels, as shown above
plt.subplot(3, 1, 3); plt.plot(net.blobs['prob'].data[0].flat)
plt_show()
