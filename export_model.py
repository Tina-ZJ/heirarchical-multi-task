# -*- coding: utf-8 -*-
#import sys
import tensorflow as tf
import numpy as np
from HAN_model import HierarchicalAttention
from data_util import create_term
import os
import codecs


from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.estimator.export.export_output import PredictOutput

SIGNATURE_NAME='hierarchical_han'
#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("attention_unit_size",100,"transition unit")
tf.app.flags.DEFINE_integer("num_cpu_threads",8,"number of cpu")
tf.app.flags.DEFINE_float("alpha",0.5,"global and local weight")
tf.app.flags.DEFINE_float("keep_prob",1.0,"keep_prob")
tf.app.flags.DEFINE_float("threshold",0.3,"predict threshold")
tf.app.flags.DEFINE_integer("num_classes",4939,"number of label")
tf.app.flags.DEFINE_integer("num_classes_first",586,"number of cid")
tf.app.flags.DEFINE_integer("num_classes_second",5933,"number of cid2")
tf.app.flags.DEFINE_integer("num_classes_product",64725,"number of product")
tf.app.flags.DEFINE_integer("num_classes_brand",168671,"number of brand")
tf.app.flags.DEFINE_integer("num_classes_third",4939,"number of cid3")
tf.app.flags.DEFINE_integer("num_tags",10,"number of tags")
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 24000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_integer("sequence_length",8,"max sentence length")
tf.app.flags.DEFINE_integer("first_length",5,"max first y length")
tf.app.flags.DEFINE_integer("second_length",10,"max second y length")
tf.app.flags.DEFINE_integer("third_length",12,"max third y length")
tf.app.flags.DEFINE_integer("embed_size",200,"embedding size")  #200
tf.app.flags.DEFINE_integer("hidden_size",200,"hidden size")   #128
tf.app.flags.DEFINE_integer("fc_size",512,"hidden size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",3,"number of epochs to run.")
tf.app.flags.DEFINE_string("train_sample_file","data/train_sample.tfrecord","path of traning data.")
tf.app.flags.DEFINE_string("dev_sample_file","data/dev_sample.tfrecord","path of dev data.")
tf.app.flags.DEFINE_string("term_index_path","data/term_index.txt",'path of term')
tf.app.flags.DEFINE_string("cid3_path","data/cid3_name.txt",'path of cid3')
tf.app.flags.DEFINE_string("cid2_path","data/cid2_name.txt",'path of cid2')
tf.app.flags.DEFINE_string("cid_path","data/cid_name.txt",'path of cid')
tf.app.flags.DEFINE_integer("train_sample_num",195635677,"train sample num")
tf.app.flags.DEFINE_integer("dev_sample_num",1327682,"dev sample num")
tf.app.flags.DEFINE_boolean("do_train",True,"whether to run training")
tf.app.flags.DEFINE_boolean("do_eval",False,"whether to run eval on the dev")
tf.app.flags.DEFINE_boolean("do_predict",False,"whether to run model in inference")
tf.app.flags.DEFINE_string("output_dir","./output/checkpoint",'save model path')
tf.app.flags.DEFINE_integer("save_checkpoints_steps",5000, "how many steps to make estimator call")





def model_fn_builder(vocab_size):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info(" name = %s, shape = %s" % (name, features[name].shape))
        input_x = features["input_x"]
        #input_y_first = features["input_y_first"]
        input_y_first = tf.ones(tf.shape(input_x), dtype=tf.float32)
        input_y_second = tf.ones(tf.shape(input_x), dtype=tf.float32)
        input_product = tf.ones(tf.shape(input_x), dtype=tf.float32)
        input_brand = tf.ones(tf.shape(input_x), dtype=tf.float32)
        #input_y_second = features["input_y_second"]
        model = HierarchicalAttention( FLAGS.keep_prob, input_x, input_y_first, input_y_second, input_product, input_brand, 
                                       FLAGS.fc_size, FLAGS.attention_unit_size, FLAGS.alpha, FLAGS.threshold,
                                       FLAGS.num_classes_first, FLAGS.num_classes_second, FLAGS.num_classes_third,
                                       FLAGS.num_classes_product, FLAGS.num_classes_brand, FLAGS.num_tags, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                                       FLAGS.decay_rate, FLAGS.sequence_length, vocab_size, FLAGS.embed_size, FLAGS.hidden_size, FLAGS.is_training)

        total_loss, f1, recall, precision = model.loss_val, model.cid_f1, model.cid_recall, model.cid_precision
        if mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={"predictions": model.predictions,
                                 "product_predictions": model.product_predictions,
                                 "brand_predictions": model.brand_predictions},
                                  export_outputs={SIGNATURE_NAME: PredictOutput({"predictions": model.predictions,
                                                                                 "product_predictions": model.product_predictions,
                                                                                  "brand_predictions": model.brand_predictions})})
        return output_spec
    return model_fn


def serving_input_receiver_fn():
    input_x = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_x')
    receiver_tensors = {'input_x': input_x}
    features = {'input_x': input_x}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


 
        
if __name__=='__main__':
    
    #1. load vocabulary
    vocabulary_word2index, vocabulary_index2word= create_term(FLAGS.term_index_path)
    vocab_size = len(vocabulary_index2word)
    print("vocab_size:",vocab_size)
    cp_file = tf.train.latest_checkpoint(FLAGS.output_dir)
    model_fn = model_fn_builder(
            vocab_size=vocab_size)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.log_device_placement = False
    batch_size = 1
    export_dir = FLAGS.output_dir 
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.output_dir, config=RunConfig(session_config=config),
                                                params={'batch_size': batch_size})
    estimator.export_saved_model(export_dir, serving_input_receiver_fn, checkpoint_path=cp_file) 
