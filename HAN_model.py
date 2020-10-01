# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import numpy as np
import tensorflow.contrib as tf_contrib
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import rnn

class HierarchicalAttention:
    def __init__(self, keep_prob, input_x, input_y_first, input_y_second, input_product, input_brand, input_tag, fc_size, attention_unit_size, alpha, threshold, num_classes_first, num_classes_second, num_classes_third, num_classes_product, num_classes_brand, num_tags, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size, hidden_size, is_training, initializer=initializers.xavier_initializer(), clip_gradients=5.0):
        """init all hyperparameter here"""
        # set hyperparamter
        self.fc_size = fc_size
        self.attention_unit_size = attention_unit_size
        self.alpha = alpha
        self.threshold = threshold
        self.num_classes_first = num_classes_first
        self.num_classes_second = num_classes_second
        self.num_classes_third = num_classes_third
        self.num_classes_product = num_classes_product
        self.num_classes_brand = num_classes_brand
        self.num_tags = num_tags
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.95)
        self.initializer = initializer
        self.clip_gradients=clip_gradients

        #self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        self.input_x = input_x
        # sequence true length
        self.length = tf.reduce_sum(tf.sign(self.input_x), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32)

        self.input_mask = tf.cast(tf.not_equal(self.input_x, 0), tf.float32)

        self.input_y_first = input_y_first
        self.input_y_second = input_y_second
        self.input_product = input_product
        self.input_brand = input_brand
        self.input_tag = input_tag
    
        self.dropout_keep_prob = keep_prob
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        # init model
        self.instantiate_weights()
        
        # lstm output
        self.gru_out, self.gru_out_pool = self.BiGRU()

        # for tag head
        self.tag_logits, self.tag_logits_ = self.Ner(self.gru_out)
        tag_predictions = tf.nn.softmax(self.tag_logits, axis=-1) 
        tag_predictionss = tf.nn.softmax(self.tag_logits_, axis=-1)
        self.tag_softmax = tf.identity(tag_predictionss, name='tag_predicitions') 
        self.tag_predictions = tf.argmax(tag_predictions, axis=-1) 
        
        # first level
        self.first_att_weight, self.first_att_out = self.attention(self.gru_out, self.num_classes_first, name="first-")
        self.first_local_input = tf.concat([self.gru_out_pool, self.first_att_out], axis=1)
        self.first_local_fc_out = self.fc_layer(self.first_local_input, name="first-local-")
        self.first_logits, self.first_scores, self.first_visual = self.local_layer(
            self.first_local_fc_out, self.first_att_weight, self.num_classes_first, name="first-")

        # second level
        self.second_att_input = tf.multiply(self.gru_out, tf.expand_dims(self.first_visual, -1))
        self.second_local_input = tf.reduce_sum(self.second_att_input, axis=1)

        self.fc_out = self.fc_layer(self.second_local_input)
       
        # dropout 
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.fc_out, self.dropout_keep_prob)
        # global output
        self.second_logits, self.second_scores = self.global_layer(self.num_classes_second, 'cid_class')
        self.product_logits, self.product_scores = self.global_layer(self.num_classes_product, 'product_class')
        self.brand_logits, self.brand_scores = self.global_layer(self.num_classes_brand, 'brand_class')
        
        self.predictions = tf.reshape(self.second_scores,[-1, self.num_classes_second], name='prediction')
        self.product_predictions = tf.reshape(self.product_scores,[-1, self.num_classes_product], name='product_prediction')
        self.brand_predictions = tf.reshape(self.brand_scores,[-1, self.num_classes_brand], name='brand_prediction')
       
        self.cid_precision, self.cid_recall, self.cid_f1 = self.metric(self.predictions, self.input_y_second, 'cid_') 
        self.product_precision, self.product_recall, self.product_f1 = self.metric(self.product_predictions, self.input_product, 'product_') 
        self.brand_precision, self.brand_recall, self.brand_f1 = self.metric(self.brand_predictions, self.input_brand, 'brand_') 
        

        # for tag acc
        correct = tf.cast(tf.equal(tf.cast(self.tag_predictions, tf.int64) - self.input_tag, 0), tf.float32)
        correct = tf.reduce_sum(self.input_mask * correct) 
        self.tag_acc = tf.div(correct, tf.cast(tf.reduce_sum(self.length), tf.float32))
            
       
        # multi task loss 
        self.loss_val = self.multi_task_loss()

        if not is_training:
            return

        # optimizer
        self.train_op = self.train()
      

    def metric(self, predictions, labels, name):
        tp = tf.reduce_sum(tf.cast(tf.greater(predictions, self.threshold), tf.float32) * tf.cast(tf.equal(labels, 1), tf.float32))
        tn = tf.reduce_sum(tf.cast(tf.less(predictions, self.threshold), tf.float32) * tf.cast(tf.equal(labels, 0), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.greater(predictions, self.threshold), tf.float32) * tf.cast(tf.equal(labels, 0), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.less(predictions, self.threshold), tf.float32) * tf.cast(tf.equal(labels, 1), tf.float32))
        precision = tf.div(tp, tp+fp, name=name+'precision')
        recall = tf.div(tp, tp+fn, name=name+'recall')
        f1 = tf.div(2*precision*recall, precision+recall, name=name+'f1')
        return precision, recall, f1
 
    def global_layer(self,  num_class,  name):
        #dropout
        with tf.name_scope(name):
            num_units = self.h_drop.get_shape().as_list()[-1]
            W = tf.Variable(tf.truncated_normal(shape=[num_units, num_class],
                                                stddev=0.1, dtype=tf.float32), name="W_"+name)
            b = tf.Variable(tf.constant(value=0.1, shape=[num_class],dtype=tf.float32), name="b_"+name)
            global_logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            global_scores = tf.sigmoid(global_logits, name="scores")
        return global_logits, global_scores

    
    def BiGRU(self):
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)
        gru_fw_cell = rnn.GRUCell(self.hidden_size)
        gru_bw_cell = rnn.GRUCell(self.hidden_size)
        outputs,_ = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell, gru_bw_cell, self.embedded_words, sequence_length=self.length, dtype=tf.float32)
        gru_out = tf.concat(outputs, axis=2)

        # mask
        mask = tf.cast(self.input_mask, tf.float32)
        gru_out = tf.multiply(gru_out, tf.expand_dims(mask,-1))
        gru_out_pool = tf.reduce_sum(gru_out, axis=1)
        return gru_out, gru_out_pool  
       
    def Ner(self, gru_out):
        batch_size = gru_out.get_shape().as_list()[0]
        gru_out_ = tf.reshape(gru_out, shape=[-1, self.hidden_size*2])
        tag_drop = tf.nn.dropout(gru_out_, keep_prob=self.dropout_keep_prob)
        tag_logit_ = tf.matmul(tag_drop, self.W_tag) + self.b_tag
        tag_logit = tf.reshape(tag_logit_,[self.batch_size, -1, self.num_tags])
        return tag_logit, tag_logit_
 
    def attention(self, input_x, num_classes, name=""):
        num_units = input_x.get_shape().as_list()[-1]
        with tf.name_scope(name + "attention"):
            W_transition = tf.Variable(tf.truncated_normal(shape=[self.attention_unit_size, num_units],
                                                           stddev=0.1, dtype=tf.float32), name="W_transition")        
            W_context = tf.Variable(tf.truncated_normal(shape=[num_classes, self.attention_unit_size],
                                                           stddev=0.1, dtype=tf.float32), name="W_context")        
            
            attention_matrix = tf.tanh(tf.matmul(input_x, tf.transpose(W_transition)))
            attention_matrix = tf.transpose(attention_matrix, perm=[0,2,1])
            attention_matrix = tf.matmul(W_context,attention_matrix) #256*9*8

            attention_weight = tf.nn.softmax(attention_matrix, name="attention")
            attention_out = tf.matmul(attention_weight, input_x)
            attention_out = tf.reduce_mean(attention_out, axis=1)
        return attention_weight, attention_out


    def local_layer(self, input_x, input_att_weight, num_classes, name=""):
        with tf.name_scope(name+"output"):
            num_units = input_x.get_shape().as_list()[-1]
            W = tf.Variable(tf.truncated_normal(shape=[num_units, num_classes],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[num_classes], dtype=tf.float32), name='b')
            logits = tf.nn.xw_plus_b(input_x, W, b, name="logits")
            scores = tf.sigmoid(logits, name="scores")
            
            # shape of visual: [batch_size, sequence_length]
            visual = tf.multiply(input_att_weight, tf.expand_dims(scores, -1))
            # mask
            mask = (1.0 - tf.cast(self.input_mask, tf.float32))*-10000.0
            visual = visual + tf.expand_dims(mask, 1) 
            visual = tf.nn.softmax(visual)
            visual = tf.reduce_mean(visual, axis=1, name="visual")
        return logits, scores, visual
    

    def fc_layer(self, input_x, name=""):
        with tf.name_scope(name + "fc"):
            num_units = input_x.get_shape().as_list()[-1]
            W = tf.Variable(tf.truncated_normal(shape=[num_units, self.fc_size],
                                                stddev=0.1, dtype=tf.float32, name="W"))
            b = tf.Variable(tf.constant(value=0.1, shape=[self.fc_size],dtype=tf.float32), name="b")
            fc = tf.nn.xw_plus_b(input_x, W, b)
            fc_out = tf.nn.relu(fc)
        return fc_out
 
    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def tag_loss(self):
        log_probs = tf.nn.log_softmax(self.tag_logits, axis=-1)
        one_hot_labels = tf.one_hot(self.input_tag, depth=self.num_tags, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss_tag = tf.reduce_mean(per_example_loss)
        return loss_tag

    def class_loss(self, labels, logits, name):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits= logits)
        #losses = tf.reduce_mean(tf.reduce_sum(losses,axis=1), name=name + "losses")
        losses = tf.reduce_mean(losses, name=name + "losses")
        return losses
        
    def l2_loss(self, l2_lambda=0.0001):
        l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables() if 'bias' not in v.name], name="l2_loss") * l2_lambda
        return l2_losses


    def multi_task_loss(self):
        losse_first = self.class_loss(labels=self.input_y_first, logits=self.first_logits, name="first_")
        losse_second = self.class_loss(labels=self.input_y_second, logits=self.second_logits, name="second_")
        losse_product = self.class_loss(labels=self.input_product, logits=self.product_logits, name="product_")
        losse_brand = self.class_loss(labels=self.input_brand, logits=self.brand_logits, name="brand_")
        tag_losses = self.tag_loss() 
        # sum
        loss = tf.add_n([losse_first, losse_second, losse_product, losse_brand, tag_losses], name="loss")
        return loss
 
    def train(self):
        """based on the loss, use SGD to update parameter"""
        #self.decay_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        self.decay_learning_rate = tf.train.exponential_decay(self.learning_rate, tf.train.get_global_step(), self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=None,
                                                   learning_rate=self.decay_learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op



    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding_projection"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)

        with tf.name_scope("tag_projection"):
            self.W_tag = tf.get_variable("W_tag", shape=[self.hidden_size*2, self.num_tags], initializer=self.initializer)
            self.b_tag = tf.get_variable("b_tag", shape=[self.num_tags])
