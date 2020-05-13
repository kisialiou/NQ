import tensorflow as tf
import sys

sys.path.append('/home/bsucm/NQ/')
import src.utils.modeling as modeling

config = {'attention_probs_dropout_prob':0.1,
'hidden_act':'gelu', # 'gelu',
'hidden_dropout_prob':0.1,
'hidden_size':1024,
'initializer_range':0.02,
'intermediate_size':4096,
'max_position_embeddings':512,
'num_attention_heads':16,
'num_hidden_layers':24,
'type_vocab_size':2,
'vocab_size':30522}

class TDense(tf.keras.layers.Layer):
    def __init__(self,
                 output_size,
                 kernel_initializer=None,
                 bias_initializer="zeros",
                **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
    def build(self,input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
          raise TypeError("Unable to build `TDense` layer with "
                          "non-floating point (and non-complex) "
                          "dtype %s" % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
          raise ValueError("The last dimension of the inputs to "
                           "`TDense` should be defined. "
                           "Found `None`.")
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.output_size,last_dim],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        self.bias = self.add_weight(
            "bias",
            shape=[self.output_size],
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True)
        super(TDense, self).build(input_shape)
    def call(self,x):
        return tf.matmul(x,self.kernel,transpose_b=True)+self.bias
    
def mk_model():
    
     # Enable XLA.
    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(True)

    seq_len = config['max_position_embeddings']
    unique_id  = tf.keras.Input(shape=(1,),dtype=tf.int64,name='unique_id')
    input_ids   = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_ids')
    input_mask  = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_mask')
    segment_ids = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='segment_ids')
    BERT = modeling.BertModel(config=config,name='bert')
    pooled_output, sequence_output = BERT(input_word_ids=input_ids,
                                          input_mask=input_mask,
                                          input_type_ids=segment_ids)
    
    logits = TDense(2,name='logits')(sequence_output)
    start_logits,end_logits = tf.split(logits,axis=-1,num_or_size_splits= 2,name='split')
    start_logits = tf.squeeze(start_logits,axis=-1,name='start_squeeze')
    end_logits   = tf.squeeze(end_logits,  axis=-1,name='end_squeeze')
    
    ans_type      = TDense(5,name='ans_type')(pooled_output)
    return tf.keras.Model([input_ for input_ in [unique_id,input_ids,input_mask,segment_ids] 
                           if input_ is not None],
                          [unique_id,start_logits,end_logits,ans_type],
                          name='bert-baseline')