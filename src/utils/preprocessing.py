from .bert_utils import *
import os
from .tokenization import *
from tqdm import tqdm



def preprocess(dataset, outfile, vocab):
    eval_writer = FeatureWriter(
            filename=os.path.join(outfile),
            is_training=False)

    tokenizer = FullTokenizer(vocab_file=vocab, 
                                        do_lower_case=True)

    features = []
    convert = ConvertExamples2Features(tokenizer=tokenizer,
                                                is_training=False,
                                                output_fn=eval_writer.process_feature,
                                                collect_stat=False)

    n_examples = 0
    for examples in nq_examples_iter(input_file=dataset, 
                                        is_training=False,
                                        tqdm=tqdm):
        for example in examples:
            n_examples += convert(example)

    eval_writer.close()
    print('number of test examples: %d, written to file: %d' % (n_examples,eval_writer.num_features))