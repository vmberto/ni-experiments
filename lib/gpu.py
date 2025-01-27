def set_memory_growth(tf):
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_memory_growth(physical_devices[1], True)


def reset_tensorflow_keras_backend(tf, gc, K):
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()
