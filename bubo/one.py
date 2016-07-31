import caffe
from contextlib import contextmanager
import ipdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint

import bubo.util;            reload(bubo.util)            # XXX dev: Reload in case we're in repl
import bubo.mpl_backend_xee; reload(bubo.mpl_backend_xee) # XXX dev: Reload in case we're in repl

from bubo.util import caffe_root, plot_image, show_shapes
from bubo.util import shell, singleton, puts

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
plot_image(image)

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
    show_shapes('tile.pre', data)

    pad = lambda data, padding: np.pad(data, padding, mode='constant', constant_values=1) # Pad with 1's (white)

    # Pad the right and bottom of each tile
    n    = data.shape[0]
    w    = int(np.round(w(n)))
    h    = int(np.ceil(n / float(w)))
    data = pad(data, (
        (0, h*w-n), # Blank tiles to fill out rectangle
        (0, 1),     # Blank col (row?) on the right of each tile
        (0, 1),     # Blank row (col?) on the bottom of each tile
        (0, 0),     # Don't pad the rgb dim
    )[:data.ndim])  # In case rgb dim isn't present

    # Tile the filters into an image
    data = data.reshape((h,w) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((h * data.shape[1], w * data.shape[3]) + data.shape[4:])

    # Pad the left and top of tiled figure
    data = pad(data, (
        (1, 0),    # Blank col (row?) on the left of tiled figure
        (1, 0),    # Blank row (col?) on the top of tiled figure
        (0, 0),    # Don't pad the rgb dim
    )[:data.ndim]) # In case rgb dim isn't present

    show_shapes('tile.post', data)
    return data

def tile_tiles(data):

    # Calculate our various widths (w_*)
    (n_out, n_in, h_pixels_per_filter, w_pixels_per_filter) = data.shape
    [w_figure_in, h_figure_in] = plt.rcParams['figure.figsize']
    figure_wh_ratio            = w_figure_in / float(h_figure_in)
    w_filters_per_tile         = max(3, int(np.ceil(np.sqrt(n_in))))
    w_tiles_per_figure         = np.sqrt(n_out * figure_wh_ratio)

    show_shapes('tile_tiles.pre', data)
    data = tile(
        lambda n: w_tiles_per_figure,
        # Use dim 1 as color channel if it has size 3
        norm(data).transpose(0,2,3,1) if data.shape[1] == 3 else ( # (b,3,h,w) -> (b,h,w,3)
            np.array(map(
                lambda xs: tile(lambda n: w_filters_per_tile, xs),
                np.array(map(norm, data)),
            ))
        )
    )
    return show_shapes('tile_tiles.post', data)

def vis_layer_params(net, layer_name):
    data = net.params[layer_name][0].data # [param_weights, param_biases]
    with show_shapes.bracketing('vis_layer_params[%s]' % layer_name, data):
        plot_image(tile_tiles(data))

vis_layer_params(net, 'conv1')
vis_layer_params(net, 'conv2')
vis_layer_params(net, 'conv3')
vis_layer_params(net, 'conv4')
vis_layer_params(net, 'conv5')

###

# TODO You are here (currently produces lots of junk)

def vis_square(data):
    'data: an array of shape (n, height, width) or (n, height, width, 3)'
    plot_image(tile(np.sqrt, norm(data)))

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
plt.tight_layout()
plt.show()
