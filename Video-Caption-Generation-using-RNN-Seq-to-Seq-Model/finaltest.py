import numpy as np
from data import dataset
import tensorflow as tf
import pickle
import os
import math
from testbase import S2VT
import sys

# Parameter Setting
input_num = 4096
hidden_num = 256
frame_num = 80
batch_size = 256
epoch_num = 100
test_dir = sys.argv[1]

def test():
    data_test = dataset(batch_size,test_dir,'./testing_label.json')
    data_test.load_token()
    data_test.new_process_data()
    print('Finish data loading')

    test_graph = tf.Graph()
    with test_graph.as_default():
        model = S2VT(input_num,hidden_num,frame_num)
        model.load_vocab()
        feat = tf.placeholder(tf.float32, [None, frame_num, input_num], name='features')
        _, _, pred_op, logit_list_op = model.modelBuilding(feat, isTrain=False)
        saver = tf.train.Saver(max_to_keep=3)
    sess_test = tf.Session(graph=test_graph)
    
    cur_dict = os.getcwd()
    print('Restoring model from:')
    print(cur_dict,'/save/\n')
    latest_checkpoint = tf.train.latest_checkpoint(cur_dict+'/save/')
    saver.restore(sess_test, latest_checkpoint)
    print('Model restoration completed.')
    
    txt = open(cur_dict+'MLDS_hw2_1_dataoutput_testset.txt', 'w')
    batch_num = int(math.ceil(data_test.dataSize/batch_size))
    eos = model.token.texts_to_sequences(['<EOS>'])[0][0]
    eos_idx = model.max_caption_len
    for i in range(batch_num):
        id_batch, feat_batch = data_test.next_feature_batch()
        prediction = sess_test.run(pred_op,feed_dict={feat:feat_batch})
        for j in range(len(feat_batch)):
            for k in range(model.max_caption_len):
                if prediction[j][k]== eos:
                    eos_idx = k
                    break
            cap_output = model.token.sequences_to_texts([prediction[j][0:eos_idx]])[0]
            txt.write(id_batch[j] + "," + str(cap_output) + "\n")
    txt.close()
    print('Testing Output Generated.')
    
    
if __name__ == '__main__':
    test()
    
    