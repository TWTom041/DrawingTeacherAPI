import tensorflow as tf


def get_nextline(outline, drew, prev):
    hub_module = tf.saved_model.load(r".\tf_models\get_step\nfirst\normal")
    outputs = hub_module(outline, drew, prev)
    output = outputs[0]
    return output


def get_firstline(outline):
    hub_module = tf.saved_model.load(r".\tf_models\get_step\first\normal")
    outputs = hub_module(outline)
    output = outputs[0]
    return output
