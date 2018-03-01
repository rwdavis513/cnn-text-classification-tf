from train import session_setup
import tensorflow as tf
import os
from text_cnn import TextCNN
from config import FLAGS
from data_helpers import batch_iter, load_data
from train import get_num_classes, step


def test_session_setup():
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
    objs = session_setup(sess, cnn)
    assert len(objs) == 7


def test_training():
    print(FLAGS.accounting_data_file)
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
                checkpoint_prefix, saver, out_dir  = session_setup(sess, cnn)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Generate batches
            batches = batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for _, batch in enumerate(batches):
                if _ > 11:
                    break
                x_batch, y_batch = zip(*batch)
                time_str, step_num, loss, accuracy = step(sess, cnn, x_batch, y_batch, train_op, train_summary_op, dev_summary_op, global_step,  train_or_dev='train')
                current_step = tf.train.global_step(sess, global_step)
                if step_num % 10 == 0:
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step_num, loss, accuracy))
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    time_str, step_num, loss, accuracy = step(sess, cnn, x_batch, y_batch, train_op, train_summary_op, dev_summary_op, global_step, train_or_dev='dev', writer=dev_summary_writer)
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step_num, loss, accuracy))
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

if __name__ == '__main__':
    test_training()
