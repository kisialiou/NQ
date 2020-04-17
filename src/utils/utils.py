import collections
import tensorflow as tf

import numpy as np
import gzip
import json

class DummyObject:
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)

FLAGS=DummyObject(skip_nested_contexts=True, #True
                  max_position=50,
                  max_contexts=48,
                  max_query_length=64,
                  max_seq_length=512, #512
                  doc_stride=128,
                  include_unknowns=0.02, #0.02
                  n_best_size=5, #20
                  max_answer_length=30, #30
                  
                  warmup_proportion=0.1,
                  learning_rate=1e-5,
                  num_train_epochs=3.0,
                  train_batch_size=32,
                  num_train_steps=100000,
                  num_warmup_steps=10000,
                  max_eval_steps=100,
                  use_tpu=False,
                  eval_batch_size=8, 
                  max_predictions_per_seq=20)


seq_length = FLAGS.max_seq_length #config['max_position_embeddings']
name_to_features = {
      "unique_id": tf.io.FixedLenFeature([], tf.int64),
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
  }

def decode_record(record, name_to_features=name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(serialized=record, features=name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if name != 'unique_id': #t.dtype == tf.int64:
            t = tf.cast(t, dtype=tf.int64)
        example[name] = t

    return example

def decode_tokens(record):
    return tf.io.parse_single_example(serialized=record, 
                                      features={
                                          "unique_id": tf.io.FixedLenFeature([], tf.int64),
                                          "token_map" :  tf.io.FixedLenFeature([seq_length], tf.int64)
                                      })


Span = collections.namedtuple("Span", ["start_token_idx", "end_token_idx", "score"])


class ScoreSummary(object):
  def __init__(self):
    self.predicted_label = None
    self.short_span_score = None
    self.cls_token_score = None
    self.answer_type_logits = None


class EvalExample(object):
  """Eval data available for a single example."""
  def __init__(self, example_id, candidates):
    self.example_id = example_id
    self.candidates = candidates
    self.results = {}
    self.features = {}


def get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(
      enumerate(logits[1:], 1), key=lambda x: x[1], reverse=True)
  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def top_k_indices(logits,n_best_size,token_map):
    indices = np.argsort(logits[1:])+1
    indices = indices[token_map[indices]!=-1]
    return indices[-n_best_size:]


def remove_duplicates(span):
    start_end = []
    for s in span:
        cont = 0
        if not start_end:
            start_end.append(Span(s[0], s[1], s[2]))
            cont += 1
        else:
            for i in range(len(start_end)):
                if start_end[i][0] == s[0] and start_end[i][1] == s[1]:
                    cont += 1
        if cont == 0:
            start_end.append(Span(s[0], s[1], s[2]))
            
    return start_end


def get_short_long_span(predictions, example):
    
    sorted_predictions = sorted(predictions, reverse=True)
    short_span = []
    long_span = []
    for prediction in sorted_predictions:
        score, _, summary, start_span, end_span = prediction
        # get scores > zero
        if score > 0:
            short_span.append(Span(int(start_span), int(end_span), float(score)))

    short_span = remove_duplicates(short_span)

    for s in range(len(short_span)):
        for c in example.candidates:
            start = short_span[s].start_token_idx
            end = short_span[s].end_token_idx
            ## print(c['top_level'],c['start_token'],start,c['end_token'],end)
            if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
                long_span.append(Span(int(c["start_token"]), int(c["end_token"]), float(short_span[s].score)))
                break
    long_span = remove_duplicates(long_span)
    
    if not long_span:
        long_span = [Span(-1, -1, -10000.0)]
    if not short_span:
        short_span = [Span(-1, -1, -10000.0)]
        
    
    return short_span, long_span


def compute_predictions(example):
    """Converts an example into an NQEval object for evaluation."""
    predictions = []
    n_best_size = FLAGS.n_best_size
    max_answer_length = FLAGS.max_answer_length
    i = 0
    for unique_id, result in example.results.items():
        if unique_id not in example.features:
            raise ValueError("No feature found with unique_id:", unique_id)
        token_map = np.array(example.features[unique_id]["token_map"]) #.int64_list.value
        start_indexes = top_k_indices(result.start_logits,n_best_size,token_map)
        if len(start_indexes)==0:
            continue
        end_indexes   = top_k_indices(result.end_logits,n_best_size,token_map)
        if len(end_indexes)==0:
            continue
        # sophisticated way of making cartesian product
        indexes = np.array(list(np.broadcast(start_indexes[None],end_indexes[:,None])))
        # select only valid pairs  
        indexes = indexes[(indexes[:,0]<indexes[:,1])*(indexes[:,1]-indexes[:,0]<max_answer_length)]
        for _, (start_index,end_index) in enumerate(indexes):  
            summary = ScoreSummary()
            summary.short_span_score = (
                result.start_logits[start_index] +
                result.end_logits[end_index])
            summary.cls_token_score = (
                result.start_logits[0] + result.end_logits[0])
            summary.answer_type_logits = result.answer_type_logits-result.answer_type_logits.mean()
            start_span = token_map[start_index]
            end_span = token_map[end_index] + 1

            # Span logits minus the cls logits seems to be close to the best.
            score = summary.short_span_score - summary.cls_token_score
            predictions.append((score, i, summary, start_span, end_span))
            i += 1 # to break ties

    # Default empty prediction.
    #score = -10000.0
    default_span  = Span(-1, -1, -10000.0)
    summary    = ScoreSummary()

    if predictions:
        short_spans, long_spans = get_short_long_span(predictions, example)
    else:
        short_spans, long_spans = [default_span], [default_span]
    
    # selecting only one long span
    # heuristic is taken from notebook
    long_span = max(long_spans, key = lambda span: span.score)
    long_span = long_span if long_span.score > 3 else default_span

    summary.predicted_label = {
        "example_id": int(example.example_id),
        "long_answer": {
          "start_token": long_span.start_token_idx,
          "end_token": long_span.end_token_idx, 
          "start_byte": -1, "end_byte": -1
        },
        "long_answer_score": long_span.score
       }

    # selecting suitable short spans
    # heuristic is taken from notebook
    summary.predicted_label["short_answers"] = []
    short_answers_score = 0
    for short_ans in short_spans:
        if short_ans.score > 8:
            short_answers_score += short_ans.score
            summary.predicted_label["short_answers"].append( 
                {
                    "start_byte": -1, "end_byte": -1,
                    "start_token":short_ans.start_token_idx,
                    "end_token":short_ans.end_token_idx
                })
    if not summary.predicted_label["short_answers"]:
        summary.predicted_label["short_answers"].append(
               {
                    "start_byte": -1, "end_byte": -1,
                    "start_token":-1,
                    "end_token":-1
                })
    summary.predicted_label["short_answers_score"] = short_answers_score / len(summary.predicted_label["short_answers"])
    summary.predicted_label["yes_no_answer"] = "NONE"
    
    return summary


def compute_pred_dict(candidates_dict, dev_features, raw_results,tqdm=None):
    """Computes official answer key from raw logits."""
    raw_results_by_id = [(int(res.unique_id),1, res) for res in raw_results]

    examples_by_id = [(int(k),0,v) for k, v in candidates_dict.items()]
  
    features_by_id = [(int(d['unique_id']),2,d) for d in dev_features] 
  
    # Join examples with features and raw results.
    examples = []
    print('merging examples...')
    merged = sorted(examples_by_id + raw_results_by_id + features_by_id)
    print('done.')
    for idx, type_, datum in merged:
        if type_==0: #isinstance(datum, list):
            examples.append(EvalExample(idx, datum))
        elif type_==2: #"token_map" in datum:
            examples[-1].features[idx] = datum
        else:
            examples[-1].results[idx] = datum

    # Construct prediction objects.
    print('Computing predictions...')
   
    nq_pred_dict = {}
    #summary_dict = {}
    if tqdm is not None:
        examples = tqdm(examples)
    for e in examples:
        summary = compute_predictions(e)
        #summary_dict[e.example_id] = summary
        nq_pred_dict[e.example_id] = summary.predicted_label
    return nq_pred_dict


def read_candidates_from_one_split(input_path):
  """Read candidates from a single jsonl file."""
  candidates_dict = {}
  print("Reading examples from: %s" % input_path)
  if input_path.endswith(".gz"):
    with gzip.GzipFile(fileobj=tf.io.gfile.GFile(input_path, "rb")) as input_file:
      for index, line in enumerate(input_file):
        e = json.loads(line)
        candidates_dict[e["example_id"]] = e["long_answer_candidates"]
        
  else:
    with tf.io.gfile.GFile(input_path, "r") as input_file:
      for index, line in enumerate(input_file):
        e = json.loads(line)
        candidates_dict[e["example_id"]] = e["long_answer_candidates"] # testar juntando com question_text
  return candidates_dict


def read_candidates(input_pattern):
  """Read candidates with real multiple processes."""
  input_paths = tf.io.gfile.glob(input_pattern)
  final_dict = {}
  for input_path in input_paths:
    final_dict.update(read_candidates_from_one_split(input_path))
  return final_dict



 # {long score > 2, cont = 5 | short score > 2, cont = 5} = 0.18
# { long score > 2, cont = 5 | short score > 6, cont = 5}
# { long score > 2, cont = 1 | short score > 6, cont = 5}

def df_long_index_score(df):
    answers = []
    cont = 0
    for e in df['long_answers']['tokens_and_score']:
        # if score > 2
        if e[2] > 3: 
            index = {}
            index['start'] = e[0]
            index['end'] = e[1]
            index['score'] = e[2]
            answers.append(index)
            cont += 1
        # number of answers
        if cont == 1:
            break
            
    return answers

def df_short_index_score(df):
    answers = []
    cont = 0
    for e in df['short_answers']['tokens_and_score']:
        # if score > 2
        if e[2] > 8:
            index = {}
            index['start'] = e[0]
            index['end'] = e[1]
            index['score'] = e[2]
            answers.append(index)
            cont += 1
        # number of answers
        if cont == 1:
            break
            
    return answers

def df_example_id(df):
    return df['example_id']


def create_answer(entry):
    answer = []
    for e in entry:
        answer.append(str(e['start']) + ':'+ str(e['end']))
    if not answer:
        answer = ""
    return ", ".join(answer)
