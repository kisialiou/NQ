import numpy as np
import tensorflow as tf
import sys
import subprocess

import json
import argparse
import os

import mlflow
import time
from tqdm import tqdm

sys.path.append('/home/bsucm/NQ/')
import src.utils.bert_utils as bert_utils
from src.utils.utils import *


parser = argparse.ArgumentParser(description='Generates prediction with given features an BERT outptut')
parser.add_argument('--run_id', type=str, help='Unique id for current run')
parser.add_argument('--data', type=str, help='Jsonl file to measure quality and performance')
parser.add_argument('--model_output', type=str)
parser.add_argument('--use_feats', type=str, default='', help='Path to existing features')
parser.add_argument('--outdir', type=str, default='/home/bsucm/NQ/data/outputs/')
args = parser.parse_args()

feat_records = args.use_feats
raw_ds = tf.data.TFRecordDataset(feat_records)
token_map_ds = raw_ds.map(decode_tokens)
decoded_ds = raw_ds.map(decode_record)

result = []
with np.load(args.model_output) as data:
    result.append(data["uniqe_id"])
    result.append(data["start_logits"])
    result.append(data["end_logits"])
    result.append(data["answer_type_logits"])

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
                                 tqdm=tqdm)

predictions_json = {"predictions": list(nq_pred_dict.values())}

print ("writing json")

prediction_name = args.outdir +  args.run_id + '_predictions' + '.json'

with tf.io.gfile.GFile(prediction_name, "w") as f:
    json.dump(predictions_json, f, indent=4)
print('done!')
