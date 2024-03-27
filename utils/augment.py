import tensorflow as tf

IMG_SIZE = 28


def apply_noise(image, label):
    noisy_image = image + tf.cast(tf.random.normal(tf.shape(image), mean=0.0, stddev=0.2), tf.float64)

    noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)

    return noisy_image, label


def augment(image_label, seed):
    image, label = image_label
    image, label = apply_noise(image, label)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
    # Make a new seed.
    new_seed = tf.random.split(seed, num=1)[0, :]
    # Random crop back to the original size.
    image = tf.image.stateless_random_crop(
        image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
    # Random brightness.
    image = tf.image.stateless_random_brightness(
        image, max_delta=0.5, seed=new_seed)
    image = tf.clip_by_value(image, 0, 1)
    return image, label
