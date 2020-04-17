import numpy as np 
import pandas as pd 
import tensorflow as tf
import sys
import importlib
import collections

sys.path.append('/home/bsucm/NQ/')

import src.utils.bert_utils as bert_utils
from src.utils.utils import *
import src.utils.modeling as modeling
import src.utils.tokenization as tokenization
from src.utils.preprocessing import preprocess

import json
import argparse
import os

MODELS_DIR = '/home/bsucm/NQ/models/'

def create_name(name, ext=''):

    if not os.path.exists(name + '.' + ext):
        return name + '.' + ext

    i = 2
    while(os.path.exists(name + str(i) + '.' + ext)):
        i += 1

    return name + str(i) + '.' + ext


parser = argparse.ArgumentParser(description='Answers question after analyzing the text')
parser.add_argument('--model', type=str, help='Dir with model maker and bin file')
parser.add_argument('--data', type=str, help='Jsonl file to measure quality and performance')
parser.add_argument('--use_feats', type=str, default='', help='Path to existing features')
parser.add_argument('--outdir', type=str, default='/home/bsucm/NQ/data/outputs/')
args = parser.parse_args()

model_pack = 'models.' + args.model
model = importlib.import_module('.model', model_pack)
model_path = MODELS_DIR + args.model + '/'

gpu = bool(tf.config.experimental.list_physical_devices('GPU'))

# Loading model
curr_model = model.mk_model()
cpkt = tf.train.Checkpoint(model=curr_model)
cpkt.restore(model_path + 'model_cpkt-1').assert_consumed()

# Data preprocessing
if not args.use_feats:
    feat_records = args.outdir + args.model

    feat_records = create_name(feat_records, 'tfrecords')

    if gpu:
        with tf.device('/GPU:0',):
            preprocess(args.data, feat_records, model_path + 'vocab.txt')
    else:
        preprocess(args.data, feat_records, model_path + 'vocab.txt')
else:
    feat_records = args.use_feats


raw_ds = tf.data.TFRecordDataset(feat_records)
token_map_ds = raw_ds.map(decode_tokens)
decoded_ds = raw_ds.map(decode_record)
ds = decoded_ds.batch(batch_size=32, drop_remainder=False) 

if gpu:
    with tf.device('/GPU:0',):
        result = curr_model.predict_generator(ds,verbose=1)
else:
    result = curr_model.predict_generator(ds,verbose=1)


np.savez_compressed(create_name(args.outdir + args.model + '-output') + '.npz',
                    **dict(zip(['uniqe_id','start_logits','end_logits','answer_type_logits'],
                               result)))

# result = []
# with np.load('/home/bsucm/NQ/data/outputs/bert_joint_baseline-output.npz') as data:
#     result.append(data["uniqe_id"])
#     result.append(data["start_logits"])
#     result.append(data["end_logits"])
#     result.append(data["answer_type_logits"])

all_results = [bert_utils.RawResult(*x) for x in zip(*result)]
    
print ("Going to candidates file")

candidates_dict = read_candidates(args.data)

print ("setting up eval features")

eval_features = list(token_map_ds)

print ("compute_pred_dict")
# For every example dict in appropriate form with answers is returned

nq_pred_dict = compute_pred_dict(candidates_dict,
                                 eval_features,
                                 all_results,
                                 tqdm=None)

predictions_json = {"predictions": list(nq_pred_dict.values())}

print ("writing json")

with tf.io.gfile.GFile(create_name(args.outdir +  args.model + '_predictions', 'json'), "w") as f:
    json.dump(predictions_json, f, indent=4)
print('done!')


