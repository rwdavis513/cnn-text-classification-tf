#! /usr/bin/env python

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time
import datetime
from data_helpers import load_data, get_num_classes, batch_iter
from text_cnn import TextCNN
from config import FLAGS


def session_setup():
    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", cnn.loss)
    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    return train_op, global_step, train_summary_op, train_summary_writer, dev_summary_writer, dev_summary_op, \
           checkpoint_prefix, saver, out_dir


def step(x_batch, y_batch, train_or_dev='train', writer=None):
    """
    Evaluates model on a dev set
    """
    if train_or_dev == 'train':
        step_list = [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy]
        drop_keep_prob = FLAGS.dropout_keep_prob
    elif train_or_dev == 'dev':
        step_list = [global_step, dev_summary_op, cnn.loss, cnn.accuracy]
        drop_keep_prob = 1.0
    else:
        raise Exception("train_or_dev must be equal to train or dev only. train_or_dev={}".format(train_or_dev))

    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: drop_keep_prob
    }
    if train_or_dev == 'train':
        _, step_num, summaries, loss, accuracy = sess.run(step_list, feed_dict)
    elif train_or_dev == 'dev':
        step_num, summaries, loss, accuracy = sess.run(step_list, feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step_num, loss, accuracy))
    if writer:
        writer.add_summary(summaries, step_num)


# Training
# ==================================================

if __name__ == '__main__':
    (x_train, x_dev, y_train, y_dev), vocab_processor = load_data()

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=get_num_classes(y_train),
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            train_op, global_step, train_summary_op, train_summary_writer, dev_summary_writer, dev_summary_op, \
                checkpoint_prefix, saver, out_dir  = session_setup()

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Generate batches
            batches = batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                step(x_batch, y_batch, train_or_dev='train')
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    step(x_dev, y_dev, train_or_dev='dev', writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
