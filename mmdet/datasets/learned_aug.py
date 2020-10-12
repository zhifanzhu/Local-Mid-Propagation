import inspect

import cv2
import numpy as np
from numpy import random

"""
Implementation of policies learned in AutoAugmentation.
See: Barret Zoph, Learning Data Augmentation Strategies for Object Detection.

According to paper:
    The most commonly used operation in good policies is Rotate.
    Two other operations that are commonly used are Equalize and BBox Only TranslateY.

Available policies:
    equalize(image)
    rotate_with_bbox(image, bboxes, degrees)
    translate_bbox(image, bboxes, pixels, shift_horizontal)
    {equalize,rotate,translate}_only_bboxes: apply augmentation to box area only.

Design Requirement:
    when a function takes as input bboxes, it must return bboxes(optionally modified) as well.
    In *_only_bboxes policies, prob is independent for each bbox.

TODO: remove wrapper since pickle don't like'em. Clean-up the code(also in extra_aug)
"""


EQUALIZE_METHOD = 'tf'


def equalize(image):
    """
        Result of 'cv2' is slightly different from TF's open source implementation.

    :param image: np array of [H, W, C]
    :return: same as image
    """
    def scale_channel_tf(im, c):
        im = im[:, :, c].astype(np.uint8)
        # Compute the histogram of the image channel.
        histo, _ = np.histogram(im.flatten(), bins=256, range=[0, 255])
        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = np.where(histo !=  0)
        nonzero_histo = np.reshape(histo[nonzero], [-1])
        step = (np.sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (np.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = np.concatenate([[0], lut[:-1]], 0)
            return np.clip(lut, 0, 255)
        if step == 0:
            result = im
        else:
            result = build_lut(histo, step)[im]
        return result.astype(np.uint8)

    def scale_channel_cv2(im, c):
        im = im[:, :, c].astype(np.uint8)
        histo, _ = np.histogram(im.flatten(), bins=256, range=[0, 255])
        cdf = histo.cumsum()
        cdf = (cdf - cdf.min()) * 254 / (cdf[-1] - cdf.min())
        return cdf[im].astype(np.uint8)

    method = EQUALIZE_METHOD
    assert method in ('tf', 'cv2')
    if method == 'tf':
        scale_channel = scale_channel_tf
    else:
        scale_channel = scale_channel_cv2
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    return np.stack([s1, s2, s3], 2)


def rotate_with_bboxes(image, bboxes, degrees):
    image = _rotate(image, degrees)
    image_height = image.shape[0]
    image_width = image.shape[1]
    bboxes = np.asarray([
        _rotate_bbox(b, image_height, image_width, degrees)
        for b in bboxes])
    return image, bboxes


def _rotate(image, degrees):
    """ Rotate image, fill border value with image mean."""
    fill = np.mean(image, (0, 1))
    fill = [int(v) for v in fill]
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, degrees, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1],
                            flags=cv2.INTER_LINEAR, borderValue=fill)
    return result


def _rotate_bbox(bbox, image_height, image_width, degrees):
    """

    :param bbox: 1D array [x1, y1, x2, y2]
    :param image_height:  int
    :param image_width: int
    :param degrees: float, Anticlockwise if positive
    :return:  array of same shape as bbox
    """
    degrees_to_radians = np.pi / 180.0
    radians = degrees * degrees_to_radians
    # Translate the bbox to the center of the image
    # Y coordinates are made negative as the y axis of images goes down with
    # increasing pixel values, so we negate to make sure x axis and y axis points
    # are in the traditionally positive direction.
    min_x = (bbox[0] - image_width * 0.5).astype(np.int32)
    min_y = - (bbox[1] - image_height * 0.5).astype(np.int32)
    max_x = (bbox[2] - image_width * 0.5).astype(np.int32)
    max_y = - (bbox[3] - image_height * 0.5).astype(np.int32)
    coordinates = np.stack(
        [[min_y, min_x], [min_y, max_x], [max_y, min_x], [max_y, max_x]]
    ).astype(np.float32)
    rotation_matrix = np.stack(
        [[np.cos(radians), np.sin(radians)],
         [-np.sin(radians), np.cos(radians)]])
    new_coords = np.matmul(rotation_matrix, np.transpose(coordinates)).astype(np.int32)
    min_y = - new_coords[0, :].max() + image_height * 0.5
    min_x = new_coords[1, :].min() + image_width * 0.5
    max_y = - new_coords[0, :].min() + image_height * 0.5
    max_x = new_coords[1, :].max() + image_width * 0.5

    # Clip
    x1, y1, x2, y2 = _clip_bbox(min_x, min_y, max_x, max_y, image_width, image_height)
    x1, y1, x2, y2 = _check_bbox_area(x1, y1, x2, y2, image_width, image_height)
    return np.stack([x1, y1, x2, y2])


def _clip_bbox(x1, y1, x2, y2, xlim, ylim):
    """ Clip bbox with abs pixels"""
    x1 = np.clip(x1, 0, xlim)
    y1 = np.clip(y1, 0, ylim)
    x2 = np.clip(x2, 0, xlim)
    y2 = np.clip(y2, 0, ylim)
    return x1, y1, x2, y2


def _check_bbox_area(x1, y1, x2, y2, xlim, ylim, delta=0.05):
    height = y2 - y1
    width = x2 - x1

    def _adjust_bbox_boundaries(min_coord, max_coord, lim, _delta):
        _delta = lim * _delta
        max_coord = np.maximum(max_coord, 0.0 + _delta)
        min_coord = np.minimum(min_coord, lim - _delta)
        return min_coord, max_coord
    if width == 0.0:
        x1, x2 = _adjust_bbox_boundaries(x1, x2, xlim, delta)
    if height == 0.0:
        y1, y2 = _adjust_bbox_boundaries(y1, y2, ylim, delta)
    return x1, y1, x2, y2


def translate_bbox(image, bboxes, pixels, shift_horizontal):
    """
    :param image: 3D array
    :param bboxes: 2D array
    :param pixels: int
    :param shift_horizontal: Boolean
    :return: A tuple containing a 3D uint8 Tensor and shifted 2D bboxes.
    """
    if shift_horizontal:
        image = translate_x(image, pixels)
    else:
        image = translate_y(image, pixels)

    image_height, image_width, _ = image.shape
    bboxes = np.asarray([
        _shift_bbox(b, image_height, image_width, pixels, shift_horizontal)
        for b in bboxes
    ])
    return image, bboxes


def translate_x(image, pixels):
    return _translate_img(image, pixels, 0)


def translate_y(image, pixels):
    return _translate_img(image, 0, pixels)


def _translate_img(image, x_pixels, y_pixels):
    height, width, _ = image.shape
    fill = image.mean((0, 1))
    fill = [int(v) for v in fill]
    M = np.float32([
        [1, 0, -x_pixels],
        [0, 1, -y_pixels],
    ])
    dst = cv2.warpAffine(image, M, (width, height), borderValue=fill)
    return dst


def _shift_bbox(bbox, image_height, image_width, pixels, shift_horizontal):
    """ Shifts the bbox coordinated by pixels

    :param bbox 1d array [x1, y1, x2, y2], in abs value
    :param image_height & image_width: int
    :param pixels: int
    :param shift_horizontal: Boolean, True shift in X else in Y
    """
    pixels = np.int32(pixels)
    x1, y1, x2, y2 = np.int32(bbox[..., :4])
    if shift_horizontal:
        x1 = np.maximum(0, x1 - pixels)
        x2 = np.minimum(image_width, x2 - pixels)
    else:
        y1 = np.maximum(0, y1 - pixels)
        y2 = np.minimum(image_height, y2 - pixels)

    x1, y1, x2, y2 = _clip_bbox(x1, y1, x2, y2, xlim=image_width, ylim=image_height)
    x1, y1, x2, y2 = _check_bbox_area(x1, y1, x2, y2, image_width, image_height)
    return np.stack([x1, y1, x2, y2])


def _bbox_only_aug(image, bbox, augmentation_func, *args):
    """
        Apply augmentations_func in single box area.

    :param image: 3D array
    :param bbox: 1D [x1, y1, x2, y2]
    :param augmentation_func: function that will be applied to subsection of image.
    :param args: args that will be passed into augmentations_func
    :return: a modified version of image.
    """
    bbox = np.int32(bbox)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    ind_sel = slice(y1, y2+1), slice(x1, x2+1)
    bbox_content = image[ind_sel]
    augmented_bbox_content = augmentation_func(bbox_content, *args)
    image[ind_sel] = augmented_bbox_content
    return image


def _bboxes_only_aug(image, bboxes, augmentation_func, *args):
    """
        Apply augmentation_func in multiple bboxes area

    :param image: 3D array
    :param bboxes: 2D [N,4] of [x1, y1, x2, y2]
    :param augmentation_func: function that will be applied to subsection of image.
    :param args: args that will be passed into augmentations_func
    :return: a modified version of image.
    """
    image = np.copy(image)
    prob = args[0]
    for bbox in bboxes:
        apply = random.choice([True, False], p=[prob, 1 - prob])
        aug_args = []
        if len(args) > 1:
            magnitude_lim = args[1]
            mag = random.uniform(-magnitude_lim, magnitude_lim)
            aug_args.append(mag)
        if apply:
            image = _bbox_only_aug(image, bbox, augmentation_func, *aug_args)
        else:
            image = image
    return image


"""
    BBox only ops.
"""


def equalize_only_bboxes(image, bboxes, prob, *args):
    args = [prob]
    return _bboxes_only_aug(image, bboxes, equalize, *args), bboxes


def rotate_only_bboxes(image, bboxes, prob, mag, *args):
    args = [prob, mag]
    return _bboxes_only_aug(image, bboxes, _rotate, *args), bboxes


def translate_x_only_bboxes(image, bboxes, prob, mag, *args):
    args = [prob, mag]
    return _bboxes_only_aug(image, bboxes, translate_x, *args), bboxes


def translate_y_only_bboxes(image, bboxes, prob, mag, *args):
    args = [prob, mag]
    return _bboxes_only_aug(image, bboxes, translate_y, *args), bboxes


"""
    Randomized version.
"""


NAME_TO_FUNC = {
    'Equalize': equalize,
    'Rotate_With_BBox': rotate_with_bboxes,
    'TranslateX_BBox': lambda image, bboxes, pixels: translate_bbox(
        image, bboxes, pixels, shift_horizontal=True),
    'TranslateY_BBox': lambda image, bboxes, pixels: translate_bbox(
        image, bboxes, pixels, shift_horizontal=False),
    'Rotate_Only_BBoxes': rotate_only_bboxes,
    'Equalize_Only_BBoxes': equalize_only_bboxes,
    'TranslateX_Only_BBoxes': translate_x_only_bboxes,
    'TranslateY_Only_BBoxes': translate_y_only_bboxes,
}


def bbox_wrapper(func):
    """ Adds a bboxes funciton argument to func and returns unchanged bboxes"""
    def wrapper(images, bboxes, *args, **kwargs):
        return func(images, *args, **kwargs), bboxes
    return wrapper


def prob_wrapper(func, prob, mag):
    """ For *_only_bboxes ops """
    def wrapper(images, bboxes, *args, **kwargs):
        return func(images, bboxes, prob, mag, *args, **kwargs)
    return wrapper


def mono_prob_wrpaaer(func, prob, mag):
    """ Apply whole func with monolithic prob."""
    def wrapper(images, bboxes):
        apply = random.choice([True, False], p=[prob, 1 - prob])
        if apply:
            args = []
            mag_value = np.random.uniform(-mag, mag)
            if 'pixels' in inspect.getfullargspec(func)[0]:
                args.append(mag_value)
            if 'degrees' in inspect.getfullargspec(func)[0]:
                args.append(mag_value)
            return func(images, bboxes, *args)
        else:
            return images, bboxes
    return wrapper


def wrap_with_prob_and_mag(name, prob, magnitude):
    """
        This function takes in a str, a float prob, a int magnitude range,
    and produces a randomized function.
    """
    try:
        func = NAME_TO_FUNC[name]
    except KeyError:
        raise ValueError('Provided augmentation method not found.')

    # make sure func has 3 args, if not, append positional args.
    if 'bboxes' not in inspect.getfullargspec(func)[0]:
        func = bbox_wrapper(func)

    # functions like *_only_bboxes take in prob as argument (independent prob for each box)
    # import pdb;pdb.set_trace()
    if 'prob' in inspect.getfullargspec(func)[0]:
        func = prob_wrapper(func, prob, magnitude)
    else:
        func = mono_prob_wrpaaer(func, prob, magnitude)

    def labels_wrapper(img, bboxes, labels):
        return func(img, bboxes) + (labels, )

    return labels_wrapper


def build_policies(policies=None):
    """ Build policies

    :param policies: list of tuple (name, prob, magitude), e.g.
        policies = [
        ('Rotate_With_BBox', 0.6, 10),
        ('Equalize', 0.8, 10),
        ('TranslateY_Only_BBoxes', 0.6, 6),
        ('TranslateX_BBox', 0.6, 4),
        ('TranslateY_BBox', 0.6, 6),
        ('Rotate_Only_BBoxes', 0.6, 6),
        ('Equalize_Only_BBoxes', 0.8, 10),
        ('TranslateX_Only_BBoxes', 0.6, 6)]
    The most used three mentioned in paper are:
        policies = [
        ('Rotate_With_BBox', 0.6, 10),
        ('Equalize', 0.8, 10),
        ('TranslateY_Only_BBoxes', 0.6, 6)]

    :returns list of augmentation function.
        each function has interface (img, bboxes, labels)->(img, bboxes, labels)
    """
    def _no_op(img, bboxes, labels):
        return img, bboxes, labels
    if policies is None:
        return [_no_op]

    aug_func_list = []
    for name, prob, magnitude in policies:
        wrapped_func = wrap_with_prob_and_mag(name, prob, magnitude)
        aug_func_list.append(wrapped_func)
    return aug_func_list
