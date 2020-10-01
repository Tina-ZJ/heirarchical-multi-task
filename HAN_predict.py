# -*- coding: utf-8 -*-
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.predict
import sys
import tensorflow as tf
import time
tf.reset_default_graph()
import numpy as np
from data_util import load_test,create_term,create_label, load_cid, load_tag
from tflearn.data_utils import pad_sequences
import os
import codecs
import time
from HAN_model import HierarchicalAttention
from pathlib import Path
from tensorflow.contrib import predictor

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("attention_unit_size",100,"transition unit")
tf.app.flags.DEFINE_float("alpha",0.5,"global and local weight")
tf.app.flags.DEFINE_integer("num_classes",4939,"number of label")
tf.app.flags.DEFINE_integer("num_classes_first",563,"number of cid")
tf.app.flags.DEFINE_integer("num_classes_second",5573,"number of cid2")
tf.app.flags.DEFINE_integer("num_classes_third",4939,"number of cid3")
tf.app.flags.DEFINE_integer("num_tags",10,"number of tags")
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_string("ckpt_dir","output/checkpoint","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",8,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",200,"embedding size")
tf.app.flags.DEFINE_integer("hidden_size",200,"hidden size")
tf.app.flags.DEFINE_integer("fc_size",512,"fc size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",7,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("predict_target_file","./test.txt","target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file",'./test.han',"target file path for final prediction")
tf.app.flags.DEFINE_string("label_index_path","data/cid2_name.txt",'path of cid3')
tf.app.flags.DEFINE_string("product_index_path","data/product_name.txt",'path of cid3')
tf.app.flags.DEFINE_string("brand_index_path","data/brand_name.txt",'path of cid3')
tf.app.flags.DEFINE_string("label_name_path","data/valid_cids.name",'path of cid3')
tf.app.flags.DEFINE_string("product_name_path","data/product_name2id.txt",'path of product')
tf.app.flags.DEFINE_string("brand_name_path","data/brand_id2name.txt",'path of brand')
tf.app.flags.DEFINE_float("threshold", 0.02, "test threshold")
tf.app.flags.DEFINE_string("tags_index_path","data/tags_index.txt",'path of tags')
tf.app.flags.DEFINE_string("term_index_path","data/term_index.txt",'path of term')




def get_result(index, scores, index2label, label2name, topN=5):
    label_list, label_names = [], []
    count = 0
    for i, s in zip(index, scores):
        if (s > FLAGS.threshold and len(label_list)<15) or count<topN:
            label = index2label[i]
            name = label2name.get(label, "无结果")
            label_score = label+':'+str(s)
            name_score = name+':'+str(s)
            label_list.append(label_score)
            label_names.append(name_score)
            count+=1
    return label_list, label_names

def predict():
    # 1.load data with vocabulary of words and labels
    vocabulary_word2index, vocabulary_index2word= create_term(FLAGS.term_index_path)
    vocab_size = len(vocabulary_word2index)
    testX, lines = load_test(FLAGS.predict_target_file, vocabulary_word2index)
    f = open(FLAGS.predict_source_file,'w')

    # for tags index
    #tags_index2word = load_tag(FLAGS.tags_index_path)


    # id2name
    cid2name = load_cid(FLAGS.label_name_path)
    brand2name = load_cid(FLAGS.brand_name_path)
    product2name = load_cid(FLAGS.product_name_path)

    # id2index
    cid2index, index2cid = create_label(FLAGS.label_index_path)
    product2index, index2product = create_label(FLAGS.product_index_path)
    brand2index, index2brand = create_label(FLAGS.brand_index_path)

    #path 
    subdirs = [x for x in Path(FLAGS.ckpt_dir).iterdir() if x.is_dir() and 'temp' not in str(x)]
    model_pb = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(model_pb) 

    #evl
    for i, x in enumerate(testX):
        print(x)
        feed_dict = {'input_x': [x]}
        result = predict_fn(feed_dict)
        batch_predictions = result['predictions']
        batch_product_predictions = result['product_predictions']
        batch_brand_predictions = result['brand_predictions']
        for predictions, product_predictions, brand_predictions in zip(batch_predictions, batch_product_predictions, batch_brand_predictions):
            predictions_sorted = sorted(predictions, reverse=True)
            product_predictions_sorted = sorted(product_predictions, reverse=True)
            brand_predictions_sorted = sorted(brand_predictions, reverse=True)

            index_sorted = np.argsort(-predictions)
            product_index_sorted = np.argsort(-product_predictions)
            brand_index_sorted = np.argsort(-brand_predictions)

            label_list, label_name = get_result(index_sorted, predictions_sorted, index2cid, cid2name)
            product_list, product_name = get_result(product_index_sorted, product_predictions_sorted, index2product, product2name)
            brand_list, brand_name = get_result(brand_index_sorted, brand_predictions_sorted, index2brand, brand2name)

            f.write(lines[i]+'\t'+','.join(label_list)+'\t'+','.join(label_name)+'\t'+','.join(product_list)+'\t'+','.join(product_name)+'\t'+','.join(brand_list)+'\t'+','.join(brand_name)+'\n') 
    f.flush()
if __name__ == "__main__":
   predict() 
