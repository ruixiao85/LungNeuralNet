'''
Credit goes to Shaohui Ruan
https://github.com/eragonruan/refinenet-image-segmentation
'''


import tensorflow as tf
from tensorflow.contrib import slim
from refinenet import resnet_v1
FLAGS = tf.app.flags.FLAGS

def unpool(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])


def get_labels_from_annotation(annotation_tensor, class_labels):
    """Returns tensor of size (width, height, num_classes) derived from annotation tensor.
    The function returns tensor that is of a size (width, height, num_classes) which
    is derived from annotation tensor with sizes (width, height) where value at
    each position represents a class. The functions requires a list with class
    values like [0, 1, 2 ,3] -- they are used to derive labels. Derived values will
    be ordered in the same way as the class numbers were provided in the list. Last
    value in the aforementioned list represents a value that indicate that the pixel
    should be masked out. So, the size of num_classes := len(class_labels) - 1.

    Parameters
    ----------
    annotation_tensor : Tensor of size (width, height)
        Tensor with class labels for each element
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.

    Returns
    -------
    labels_2d_stacked : Tensor of size (width, height, num_classes).
        Tensor with labels for each pixel.
    """

    # Last value in the classes list should show
    # which number was used in the annotation to mask out
    # the ambigious regions or regions that should not be
    # used for training.
    valid_entries_class_labels=class_labels[:-1]

    # Stack the binary masks for each class
    labels_2d=map(lambda x:tf.equal(annotation_tensor, x),
                  valid_entries_class_labels)

    # Perform the merging of all of the binary masks into one matrix
    labels_2d_stacked=tf.stack(labels_2d, axis=2)

    # Convert tf.bool to tf.float
    # Later on in the labels and logits will be used
    # in tf.softmax_cross_entropy_with_logits() function
    # where they have to be of the float type.
    labels_2d_stacked_float=tf.to_float(labels_2d_stacked)

    return labels_2d_stacked_float


def get_labels_from_annotation_batch(annotation_batch_tensor, class_labels):
    """Returns tensor of size (batch_size, width, height, num_classes) derived
    from annotation batch tensor. The function returns tensor that is of a size
    (batch_size, width, height, num_classes) which is derived from annotation tensor
    with sizes (batch_size, width, height) where value at each position represents a class.
    The functions requires a list with class values like [0, 1, 2 ,3] -- they are
    used to derive labels. Derived values will be ordered in the same way as
    the class numbers were provided in the list. Last value in the aforementioned
    list represents a value that indicate that the pixel should be masked out.
    So, the size of num_classes len(class_labels) - 1.

    Parameters
    ----------
    annotation_batch_tensor : Tensor of size (batch_size, width, height)
        Tensor with class labels for each element
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.

    Returns
    -------
    batch_labels : Tensor of size (batch_size, width, height, num_classes).
        Tensor with labels for each batch.
    """

    batch_labels=tf.map_fn(fn=lambda x:get_labels_from_annotation(annotation_tensor=x, class_labels=class_labels),
                           elems=annotation_batch_tensor,
                           dtype=tf.float32)

    return batch_labels


def get_valid_entries_indices_from_annotation_batch(annotation_batch_tensor, class_labels):
    """Returns tensor of size (num_valid_eintries, 3).
    Returns tensor that contains the indices of valid entries according
    to the annotation tensor. This can be used to later on extract only
    valid entries from logits tensor and labels tensor. This function is
    supposed to work with a batch input like [b, w, h] -- where b is a
    batch size, w, h -- are width and height sizes. So the output is
    a tensor which contains indexes of valid entries. This function can
    also work with a single annotation like [w, h] -- the output will
    be (num_valid_eintries, 2).

    Parameters
    ----------
    annotation_batch_tensor : Tensor of size (batch_size, width, height)
        Tensor with class labels for each batch
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.

    Returns
    -------
    valid_labels_indices : Tensor of size (num_valid_eintries, 3).
        Tensor with indices of valid entries
    """

    # Last value in the classes list should show
    # which number was used in the annotation to mask out
    # the ambigious regions or regions that should not be
    # used for training.
    mask_out_class_label=class_labels[-1]

    # Get binary mask for the pixels that we want to
    # use for training. We do this because some pixels
    # are marked as ambigious and we don't want to use
    # them for trainig to avoid confusing the model
    valid_labels_mask=tf.not_equal(annotation_batch_tensor,
                                   mask_out_class_label)

    valid_labels_indices=tf.where(valid_labels_mask)

    return tf.to_int32(valid_labels_indices)


def get_valid_logits_and_labels(annotation_batch_tensor, logits_batch_tensor, class_labels):
    """Returns two tensors of size (num_valid_entries, num_classes).
    The function converts annotation batch tensor input of the size
    (batch_size, height, width) into label tensor (batch_size, height,
    width, num_classes) and then selects only valid entries, resulting
    in tensor of the size (num_valid_entries, num_classes). The function
    also returns the tensor with corresponding valid entries in the logits
    tensor. Overall, two tensors of the same sizes are returned and later on
    can be used as an input into tf.softmax_cross_entropy_with_logits() to
    get the cross entropy error for each entry.
    Parameters
    ----------
    annotation_batch_tensor : Tensor of size (batch_size, width, height)
        Tensor with class labels for each batch
    logits_batch_tensor : Tensor of size (batch_size, width, height, num_classes)
        Tensor with logits. Usually can be achived after inference of fcn network.
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.
    Returns
    -------
    (valid_labels_batch_tensor, valid_logits_batch_tensor) : Two Tensors of size (num_valid_eintries, num_classes).
        Tensors that represent valid labels and logits.    """

    labels_batch_tensor=get_labels_from_annotation_batch(annotation_batch_tensor=annotation_batch_tensor, class_labels=class_labels)
    valid_batch_indices=get_valid_entries_indices_from_annotation_batch(annotation_batch_tensor=annotation_batch_tensor, class_labels=class_labels)
    valid_labels_batch_tensor=tf.gather_nd(params=labels_batch_tensor, indices=valid_batch_indices)
    valid_logits_batch_tensor=tf.gather_nd(params=logits_batch_tensor, indices=valid_batch_indices)
    return valid_labels_batch_tensor, valid_logits_batch_tensor


def ResidualConvUnit(inputs,features=256,kernel_size=3):
    net=tf.nn.relu(inputs)
    net=slim.conv2d(net, features, kernel_size)
    net=tf.nn.relu(net)
    net=slim.conv2d(net,features,kernel_size)
    net=tf.add(net,inputs)

    return net

def MultiResolutionFusion(high_inputs=None,low_inputs=None,up0=2,up1=1,n_i=256):
    g0 = unpool(slim.conv2d(high_inputs, n_i, 3), scale=up0)
    if low_inputs is None:
        return g0
    g1=unpool(slim.conv2d(low_inputs,n_i,3),scale=up1)
    return tf.add(g0,g1)

def ChainedResidualPooling(inputs,n_i=256):
    net_relu=tf.nn.relu(inputs)
    net=slim.max_pool2d(net_relu, [5, 5],stride=1,padding='SAME')
    net=slim.conv2d(net,n_i,3)
    return tf.add(net,net_relu)

def RefineBlock(high_inputs=None,low_inputs=None):
    if low_inputs is not None:
        print(high_inputs.shape)
        rcu_high=ResidualConvUnit(high_inputs,features=256)
        rcu_low=ResidualConvUnit(low_inputs,features=256)
        fuse=MultiResolutionFusion(rcu_high,rcu_low,up0=2,up1=1,n_i=256)
        fuse_pooling=ChainedResidualPooling(fuse,n_i=256)
        output=ResidualConvUnit(fuse_pooling,features=256)
        return output
    else:
        rcu_high = ResidualConvUnit(high_inputs, features=256)
        fuse = MultiResolutionFusion(rcu_high, low_inputs=None, up0=1,  n_i=256)
        fuse_pooling = ChainedResidualPooling(fuse, n_i=256)
        output = ResidualConvUnit(fuse_pooling, features=256)
        return output


def model(images, weight_decay=1e-5, is_training=True):
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_101(images, is_training=is_training, scope='resnet_v1_101')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))

            g = [None, None, None, None]
            h = [None, None, None, None]

            for i in range(4):
                h[i]=slim.conv2d(f[i], 256, 1)
            for i in range(4):
                print('Shape of h_{} {}'.format(i, h[i].shape))

            g[0]=RefineBlock(h[0])
            g[1]=RefineBlock(g[0],h[1])
            g[2]=RefineBlock(g[1],h[2])
            g[3]=RefineBlock(g[2],h[3])
            g[3]=unpool(g[3],scale=4)
            F_score = slim.conv2d(g[3], 21, 1, activation_fn=tf.nn.relu, normalizer_fn=None)

    return F_score


def mean_image_subtraction(images, means=None):
    means=means or [123.68, 116.78, 103.94]
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def loss(annotation_batch,upsampled_logits_batch,class_labels):
    valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(
        annotation_batch_tensor=annotation_batch,
        logits_batch_tensor=upsampled_logits_batch,
        class_labels=class_labels)

    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                              labels=valid_labels_batch_tensor)

    cross_entropy_sum = tf.reduce_mean(cross_entropies)
    tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)

    return cross_entropy_sum
