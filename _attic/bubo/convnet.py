import caffe
from contextlib import contextmanager
import ggplot as gg
import ipdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pprint import pprint

# XXX dev: repl workflow
exec '; '.join(['import %s; reload(%s)' % (m, m) for m in [
    'bubo.util',
    'potoo.dynvar',
    'potoo.mpl_backend_xee',
]])

from bubo.util import caffe_root, plot_img, plot_gg, show_shapes, show_tuple_tight
from bubo.util import gg_layer, gg_xtight, gg_ytight, gg_tight
from bubo.util import shell, singleton, puts

#
# Setup
#

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
    model_def,      # Defines the structure of the model
    model_weights,  # Contains the trained weights
    caffe.TEST,     # Use test mode (e.g. don't perform dropout)
)

# Load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('%(caffe_root)s/python/caffe/imagenet/ilsvrc_2012_mean.npy' % locals())
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# Set the size of the input (we can skip this if we're happy with the default; we can also change it later, e.g., for
# different batch sizes)
net.blobs['data'].reshape(
    50,        # batch size
    3,         # 3-channel (BGR) images
    227, 227,  # image size is 227x227
    #224, 224, # image size is 227x227 [TODO Make googlenet work]
)

#
# For each layer, show the shapes of the activations and params:
#   blob       activations  (batch_size, channel_dim, height, width) -- typically, but not always
#   params[0]  weights      (output_channels, input_channels, filter_height, filter_width)
#   params[1]  biases       (output_channels,)
#

def show_shape(shape, name, fields):
    return '%s(%s)' % (name, ', '.join(['%s=%s' % (d,s) for (s,d) in zip(shape, fields)]))

for layer, blob in net.blobs.iteritems():
    [param_weights, param_biases] = net.params.get(layer, [None, None])
    print '%-52s %-29s %-31s %s' % (
        layer,
        show_shape(blob.data.shape, 'act', ('b', 'c', 'h', 'w')),
        param_weights and show_shape(param_weights.data.shape, 'weight', ('o', 'i', 'h', 'w')) or '',
        param_biases  and show_shape(param_biases.data.shape,  'bias',   ('o', 'i', 'h', 'w')) or '',
    )

#
# Plot params: defs (fast)
#

def norm(data):
    return (data - data.min()) / (data.max() - data.min())

def w_from_figure_wh_ratio(n_out):
    [w_figure_in, h_figure_in] = plt.rcParams['figure.figsize']
    figure_wh_ratio            = w_figure_in / float(h_figure_in)
    w_tiles_per_figure         = np.sqrt(n_out * figure_wh_ratio)
    return w_tiles_per_figure

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

# Projections from layer params: net.params[layer] = [weights, biases]
just_weights                     = lambda (ws,bs): ws.data
just_weights_sorted_by_bias      = lambda (ws,bs): np.array(map(lambda (w,b): w, _sorted_by_bias((ws,bs))))
weights_plus_bias                = lambda (ws,bs): np.array(map(lambda (x,y): x+y, zip(ws.data, bs.data)))
weights_plus_bias_sorted_by_bias = lambda (ws,bs): np.array(map(lambda (w,b): w+b, _sorted_by_bias((ws,bs))))
_sorted_by_bias                  = lambda (ws,bs): sorted(zip(ws.data, bs.data), key = lambda (w,b): -b)

def plot_layer_params_weights(net, layer, data_f, get_weights = just_weights):
    data = get_weights(net.params[layer])
    with show_shapes.bracketing('plot_layer_params_weights[%s]' % layer, data, disable=True):
        plot_img(
            data_f(data),
            'layer-params-weights-%s-%s' % (layer, show_tuple_tight(data.shape)),
        )

#
# Plot params: go (slow)
#

conv_layers = filter(lambda (layer, (weights, biases)): len(weights.data.shape) == 4, net.params.items())
fc_layers   = filter(lambda (layer, (weights, biases)): len(weights.data.shape) != 4, net.params.items())

# Plot layer weights
for layer, params in conv_layers:
    plot_layer_params_weights(net, layer, tile_tiles)
for layer, params in fc_layers:
    # Very slow, very big, very hardly insightful...
    plot_layer_params_weights(net, layer, lambda x: x.transpose(1,0))

# Plot layer biases
plot_gg(gg_layer(
    gg.ggplot(gg.aes(x='layer', y='bias'),
        pd.DataFrame([
            {'bias': bias, 'layer': layer}
            for layer, (weights, biases) in net.params.items()
            for bias in biases.data
        ])
    ),
    gg.geom_violin(),
    gg.ggtitle('layer params biases'),
))

#
# Image to classify
#

# Pick image
#img_path  = '%(caffe_root)s/examples/images/cat.jpg' % locals()
#img_path  = 'data/img/cat-pizza.jpg'
#img_path = 'data/MLSP 2013/mlsp_contest_dataset/supplemental_data/spectrograms/PC1_20090705_070000_0040.bmp'
img_path = 'data/MLSP 2013/mlsp_contest_dataset/supplemental_data/spectrograms/PC1_20100705_050001_0040.bmp'
#img_path = ... # TODO

# Load image
img_desc                    = 'img-%s' % os.path.basename(os.path.splitext(img_path)[0])
img                         = caffe.io.load_image(img_path)
transformed_img             = transformer.preprocess('data', img)
net.blobs['data'].data[...] = transformed_img # Copy the image data onto the data layer

plot_img(img, img_desc)

#
# Run net forward
#

# Perform classification (slow, ~secs)
output      = net.forward()
output_prob = output['prob'][0] # The output probability vector for the first image in the batch
print 'Predicted class: %s' % output_prob.argmax()

# Show probs with ImageNet labels
labels_file = '%(caffe_root)s/data/ilsvrc12/synset_words.txt' % locals()
if not os.path.exists(labels_file):
    shell('%(caffe_root)s/data/ilsvrc12/get_ilsvrc_aux.sh' % locals())
labels = np.loadtxt(labels_file, str, delimiter='\t')
top_inds = output_prob.argsort()[::-1][:20] # Top k predictions from softmax output
pprint(zip(output_prob[top_inds], labels[top_inds]))

#
# Plot activations
#   - FIXME Make gg_tight work with facet('free') [https://github.com/yhat/ggplot/issues/516]
#

# Only plot first image in batch (50), for now
batch_i = 0

def plot_conv_acts(layer, acts):
    data = acts.data[batch_i]
    plot_img(
        tile(w_from_figure_wh_ratio, norm(data)),
        '%s-layer-acts-%s-%s-(i=%s)' % (img_desc, layer, show_tuple_tight(data.shape), batch_i),
    )

conv_layers = filter(lambda (layer, acts): len(acts.data.shape) == 4, net.blobs.items())
fc_layers   = filter(lambda (layer, acts): len(acts.data.shape) != 4, net.blobs.items())

# Plot conv acts
for layer, acts in conv_layers:
    plot_conv_acts(layer, acts)

# Plot fc acts
df = pd.concat([
    pd.DataFrame({'act': acts.data[batch_i], 'layer': layer}).reset_index()
    for layer, acts in fc_layers
])
plot_gg(gg_layer(
    gg.ggplot(df, gg.aes(y='act', x='index')),
    gg.geom_point(alpha=.5),
    gg.facet_wrap(x='layer', scales='free'),
    gg.ggtitle('%s layer acts fc/prob points (i=%s)' % (img_desc, batch_i)),
))
plot_gg(gg_layer(
    gg.ggplot(df, gg.aes(x='act')),
    gg.geom_histogram(bins=25, size=0),
    gg.facet_wrap(x='layer', scales='free'),
    gg.scale_y_log(),
    gg.ylim(low=0.1),
    gg.ggtitle('%s layer acts fc/prob histo (i=%s)' % (img_desc, batch_i)),
))
