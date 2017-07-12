# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/data/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import pdb
import tac_kbp
import numpy as np
import tensorflow as tf
import tac_kbp
import reader
import pandas

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


parser = argparse.ArgumentParser(description="Machine")
parser.add_argument('-m', help='foo help')
args = parser.parse_args()
pdb.set_trace()
if str(args.m) == "remote":
        defaultDataFolder = "/home/khalife/ai-lab/data/"
elif str(args.m) == "local":
        defaultDataFolder = "/Users/sammy/Documents/phd-2016/lab/ai-lab/data/"
else:   
        print("Which machine?")
        sys.exit()      




def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32



from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple

# Thanks to 'initializers_enhanced.py' of Project RNN Enhancement:
# https://github.com/nicolas-ivanov/Seq2Seq_Upgrade_TensorFlow/blob/master/rnn_enhancement/initializers_enhanced.py
def orthogonal_initializer(scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if partition_info is not None:
            ValueError(
                "Do not know what to do with partition_info in BN_LSTMCell")
        flat_shape = (shape[0], np.prod(shape[1:]))
        #a = np.random.normal(0.0, 1.0, flat_shape)
        a = np.zeros(flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype)
    return _initializer


# Thanks to https://github.com/OlavHN/bnlstm
def batch_norm(inputs, name_scope, is_training, epsilon=1e-3, decay=0.99):
    with tf.variable_scope(name_scope):
        size = inputs.get_shape().as_list()[1]

        scale = tf.get_variable(
            'scale', [size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', [size])

        population_mean = tf.get_variable(
            'population_mean', [size],
            initializer=tf.zeros_initializer(), trainable=False)
        population_var = tf.get_variable(
            'population_var', [size],
            initializer=tf.ones_initializer(), trainable=False)
        batch_mean, batch_var = tf.nn.moments(inputs, [0])

        # The following part is based on the implementation of :
        # https://github.com/cooijmanstim/recurrent-batch-normalization
        train_mean_op = tf.assign(
            population_mean,
            population_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(
            population_var, population_var * decay + batch_var * (1 - decay))

        if is_training is True:
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(
                    inputs, batch_mean, batch_var, offset, scale, epsilon)
        else:
            return tf.nn.batch_normalization(
                inputs, population_mean, population_var, offset, scale,
                epsilon)


class BNLstmCell(RNNCell):
    """LSTM cell with Recurrent Batch Normalization.
    This implementation is based on:
         http://arxiv.org/abs/1603.09025
    This implementation is also based on:
         https://github.com/OlavHN/bnlstm
         https://github.com/nicolas-ivanov/Seq2Seq_Upgrade_TensorFlow
    """

    def __init__(self, num_units, is_training,
                 use_peepholes=False, cell_clip=None,
                 initializer=orthogonal_initializer(),
                 num_proj=None, proj_clip=None,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=tf.tanh):
        """Initialize the parameters for an LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          is_training: bool, set True when training.
          use_peepholes: bool, set True to enable diagonal/peephole
            connections.
          cell_clip: (optional) A float value, if provided the cell state
            is clipped by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight
            matrices.
          num_proj: (optional) int, The output dimensionality for
            the projection matrices.  If None, no projection is performed.
          forget_bias: Biases of the forget gate are initialized by default
            to 1 in order to reduce the scale of forgetting at the beginning of
            the training.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.
          activation: Activation function of the inner states.
        """
        if not state_is_tuple:
            tf.logging.log_first_n(
                tf.logging.WARN,
                "%s: Using a concatenated state is slower and "
                " will soon be deprecated.  Use state_is_tuple=True.", 1, self)

        self.num_units = num_units
        self.is_training = is_training
        self.use_peepholes = use_peepholes
        self.cell_clip = cell_clip
        self.num_proj = num_proj
        self.proj_clip = proj_clip
        self.initializer = initializer
        self.forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self.state_is_tuple = state_is_tuple
        self.activation = activation

        if num_proj:
            self._state_size = (
                LSTMStateTuple(num_units, num_proj)
                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (
                LSTMStateTuple(num_units, num_units)
                if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):

        num_proj = self.num_units if self.num_proj is None else self.num_proj

        if self._state_is_tuple:
            (c_prev, h_prev) = state
        else:
            c_prev = tf.slice(state, [0, 0], [-1, self.num_units])
            h_prev = tf.slice(state, [0, self.num_units], [-1, num_proj])

        dtype = inputs.dtype
        #input_size = inputs.get_shape().with_rank(2)[1]
        #input_size = inputs.get_shape().with_rank()
        input_size = inputs.get_shape()[1]
        with tf.variable_scope(scope or type(self).__name__):
            if input_size.value is None:
                raise ValueError(
                    "Could not infer input size from inputs.get_shape()[-1]")

            W_xh = tf.get_variable(
                'W_xh',
                [input_size, 4 * self.num_units],
                initializer=self.initializer)
            W_hh = tf.get_variable(
                'W_hh',
                [num_proj, 4 * self.num_units],
                initializer=self.initializer)
            bias = tf.get_variable('B', [4 * self.num_units])
            try:
              #print(inputs)
              #print(W_xh)
              xh = tf.matmul(inputs, W_xh)
              hh = tf.matmul(h_prev, W_hh)
            except:
              print("dimension mismatch") 
              #pdb.set_trace() 
            bn_xh = batch_norm(xh, 'xh', self.is_training)
            bn_hh = batch_norm(hh, 'hh', self.is_training)

            # i:input gate, j:new input, f:forget gate, o:output gate
            lstm_matrix = tf.nn.bias_add(tf.add(bn_xh, bn_hh), bias)
            i, j, f, o = tf.split(
                value=lstm_matrix, num_or_size_splits=4, axis=1)

            # Diagonal connections
            if self.use_peepholes:
                w_f_diag = tf.get_variable(
                    "W_F_diag", shape=[self.num_units], dtype=dtype)
                w_i_diag = tf.get_variable(
                    "W_I_diag", shape=[self.num_units], dtype=dtype)
                w_o_diag = tf.get_variable(
                    "W_O_diag", shape=[self.num_units], dtype=dtype)

            if self.use_peepholes:
                c = c_prev * tf.sigmoid(f + self.forget_bias +
                                        w_f_diag * c_prev) + \
                    tf.sigmoid(i + w_i_diag * c_prev) * self.activation(j)
            else:
                c = c_prev * tf.sigmoid(f + self.forget_bias) + \
                    tf.sigmoid(i) * self.activation(j)

            if self.cell_clip is not None:
                c = tf.clip_by_value(c, -self.cell_clip, self.cell_clip)

            bn_c = batch_norm(c, 'cell', self.is_training)

            if self.use_peepholes:
                h = tf.sigmoid(o + w_o_diag * c) * self.activation(bn_c)
            else:
                h = tf.sigmoid(o) * self.activation(bn_c)

            if self.num_proj is not None:
                w_proj = tf.get_variable(
                    "W_P", [self.num_units, num_proj], dtype=dtype)

                h = tf.matmul(h, w_proj)
                if self.proj_clip is not None:
                    h = tf.clip_by_value(h, -self.proj_clip, self.proj_clip)
            
            new_state = (LSTMStateTuple(c, h)
                         if self.state_is_tuple else tf.concat(1, [c, h]))

            return h, new_state

 

class BNRModel(object):
  """TheBNR Model"""
  def __init__(self, is_training, config, input_):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def bnlstm_cell():
      # With the latest TensorFlow source code (as of Mar 27, 2017),
      # the BasicLSTMCell will need a reuse parameter which is unfortunately not
      # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
      # an argument check here:
      
      #if 'reuse' in inspect.getargspec(
      #    tf.contrib.rnn.BasicLSTMCell.__init__).args:
      #  return tf.contrib.rnn.BasicLSTMCell(
      #      size, forget_bias=0.0, state_is_tuple=True,
      #      reuse=tf.get_variable_scope().reuse)
      #else:
      #  return tf.contrib.rnn.BasicLSTMCell(
      #      size, forget_bias=0.0, state_is_tuple=True)
      return BNLstmCell(size, is_training)
    
    attn_cell = bnlstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            bnlstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())

    #with tf.device("/cpu:0"):
    #  embedding = tf.get_variable(
    #      "embedding", [vocab_size, size], dtype=data_type())
    #  inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    # previous inputs : 
	# input_ is a PTBInput, and input_.input_data is a <tf.Tensor 'TrainInput/StridedSlice:0' shape=(20, 20) dtype=int32>
	# inputs is a <tf.Tensor 'embedding_lookup:0' shape=(20, 20, 200) dtype=float32>
    inputs = input_.input_data


    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        #(cell_output, state) = cell(inputs[:, time_step, :], state)
        (cell_output, state) = cell(inputs[:, time_step], state)
        outputs.append(cell_output)

    self.state = state
    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
    self.output = output
    self._final_state = state
 
    if not is_training:
      return
  
  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op



class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)




class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False):
#def run_epoch(session, model, loss, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state) # gets the initial state of a BNLSTM cell
  cost = tf.reduce_sum(model.loss) / model.input.batch_size
  fetches = {
      #"cost": model.cost,
      "cost": cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

def nel_producer(raw_data, config, name=None):
  """Iterate on the raw PTB data.
  This chunks up raw_data into batches of examples and returns Tensors that are drawn from these batches.
  Args:
  raw_data: one of the raw data outputs from ptb_raw_data.
  batch_size: int, the batch size.
  num_steps: int, the number of unrolls.
  name: the name of this operation (optional).
  Returns:
  A pair of Tensors, each shaped [batch_size, num_steps]. The second element
  of the tuple is the same data time-shifted to the right by one.
  Raises:
  tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  batch_size = config.batch_size
  num_steps = config.num_steps
  dimension = config.dimension

  with tf.name_scope(name, "NELProducer", [raw_data, batch_size, num_steps]):
    #raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data")
    data_len = tf.size(raw_data) # equal to total size : in case of nel : width * length
    #pdb.set_trace()
    batch_len = raw_data.shape[0].value // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len, raw_data.shape[1].value])
    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(epoch_size, message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps, dimension])
    return x 

class nelConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 50
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 5 
  vocab_size = 10000
  dimension = 500



print("Getting data")
#_, knowledgeDataFrame = tac_kbp.loadKnowledgeBase()
#queriesName, mentionsDataFrame = tac_kbp.loadMentions()
#embeddings = tac_kbp.loadEmbeddings()
#entityToEmbeddings, knowledgeDataFrame, mentionsDataFrame = tac_kbp.crossMapNel(knowledgeDataFrame, mentionsDataFrame, embeddings)
#mentionsDataFrame = tac_kbp.generateGoldAndCorruptedEntities(mentionsDataFrame, entityToEmbeddings)
jsonFolder = "/home/khalife/ai-lab/data/LDC2015E19_TAC_KBP_English_Entity_Linking_Comprehensive_Training_and_Evaluation_Data_2009-2013/json/"
mentionsDataFrame = pandas.read_pickle(jsonFolder + "mentionsWithEmbeddings.pickle")


train_data = {}
train_data["mentions_embeddings"] = mentionsDataFrame["embeddings"].values.tolist()
train_data["gold_embeddings"] = mentionsDataFrame["gold_embeddings"].values.tolist()
train_data["corrupted_embeddings"] = mentionsDataFrame["corrupted_embeddings"].values.tolist()
raw_data = train_data["mentions_embeddings"]
nel_config = nelConfig()
train_input = {}
#train_input["mentions_embeddings"] = nel_producer(train_data["mentions_embeddings"], nel_config)
#train_input["gold_embeddings"] = nel_producer(train_data["gold_embeddings"], nel_config)
#train_input["corrupted_embeddings"] = nel_producer(train_data["corrupted_embeddings"], nel_config)


class NELInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data = nel_producer(data, config)



# Train
initializer = tf.random_uniform_initializer(-nel_config.init_scale, nel_config.init_scale)
with tf.variable_scope("nel") as scope:
  mentions_embeddings = NELInput(config=nel_config, data=train_data["mentions_embeddings"], name="TestInput")
  mention_model = BNRModel(is_training=True, config=nel_config, input_=mentions_embeddings) 
  scope.reuse_variables()
  gold_entities_embeddings = NELInput(config=nel_config, data=train_data["gold_embeddings"], name="TestInput")
  gold_entity_model = BNRModel(is_training=False, config=nel_config, input_=gold_entities_embeddings)
  corrupted_entities_embeddings = NELInput(config=nel_config, data=train_data["corrupted_embeddings"], name="TestInput")
  corrupted_entity_model = BNRModel(is_training=False, config=nel_config, input_=corrupted_entities_embeddings)
  pdb.set_trace()


#
#
## Valid
#with tf.variable_scope("mentions") as scope:
#  scope.reuse_variables()
#  mentions_embeddings = PTBInput(config=config, data=valid_data["mentions_embeddings"], name="TestInput")
#  mention_model = BNRModel(is_training=False, config=config, input_=mentions_embeddings) 
#
#
#with tf.variable_scope("entity") as scope: 
#  scope.reuse_variables()
#  gold_entities_embeddings = PTBInput(config=config, data=valid_data["gold_entities_embeddings"], name="TestInput")
#  gold_entity_model = BNRModel(is_training=False, config=config, input_=gold_entities_embeddings)
#  corrupted_entities_embeddings = PTBInput(config=config, data=train_data["corrupted_entities_embeddings"], name="TestInput")
#  corrupted_entity_model = BNRModel(is_training=False, config=config, input_=corrupted_entities_embeddings)
#
## Test
#with tf.variable_scope("mentions") as scope:
#  scope.reuse_variables()
#  mentions_embeddings = PTBInput(config=eval_config, data=test_data["mentions_embeddings"], name="TestInput")
#  mention_model = BNRModel(is_training=False, config=eval_config, input_=mentions_embeddings) 
#
#
#with tf.variable_scope("entity") as scope:
#  scope.reuse_variables()
#  gold_entities_embeddings = PTBInput(config=eval_config, data=test_data["gold_entities_embeddings"], name="TestInput")
#  gold_entity_model = BNRModel(is_training=False, config=eval_config, input_=gold_entities_embeddings)
#  scope.reuse_variables()
#  corrupted_entities_embeddings = PTBInput(config=eval_config, data=test_data["corrupted_entities_embeddings"], name="TestInput")
#  corrupted_entity_model = BNRModel(is_training=False, config=eval_config, input_=corrupted_entities_embeddings)
#
#
#normalize_mention = tf.nn.l2_normalize(mention_model, 0)        
#normalize_gold_entity = tf.nn.l2_normalize(gold_entity_model, 0)
#normalize_corrupted_entity = tf.nn.normalize(corrupted_entity_model, 0)
#cosine_similarity = tf.reduce_sum(tf.multiply(normalize_mention,normalize_gold_entity))
#cosine_corrupted_similarities = tf.reduce_sum(tf.multiply(normalize_mention,normalize_corrupted_entity))
#
#distance = tf.max(0, 1 - cosine_similarity + cosine_corrupted_similarities)
#loss = tf.reduce_sum(distance)/(batch_size)
#optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)

#batch_size = 128
#with tf.Session() as session:
#for epoch in range(30):
#  avg_loss = 0.
#  avg_acc = 0.
#  total_batch = int(X_train.shape[0]/batch_size)
#  start_time = time.time()
#  # Loop over all batches
#  for i in range(total_batch):
#      s  = i * batch_size
#      e = (i+1) *batch_size
#      # Fit training using batch data
#      input1,input2,y =next_batch(s,e,tr_pairs,tr_y)
#      _,loss_value,predict=sess.run([optimizer,loss,distance], feed_dict={images_L:input1,images_R:input2 ,labels:y,dropout_f:0.9})
#      #feature1=model1.eval(feed_dict={images_L:input1,dropout_f:0.9})
#      #feature2=model2.eval(feed_dict={images_R:input2,dropout_f:0.9})
#      tr_acc = compute_accuracy(predict,y)
#      #print(model1.)
#      #print(model2.
#      pdb.set_trace()
#      if math.isnan(tr_acc) and epoch != 0:
#          print('tr_acc %0.2f' % tr_acc)
#          pit db.set_trace()
#      avg_loss += loss_value
#      avg_acc +=tr_acc*100




#	
#
#distance  = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(model1,model2),2),1,keep_dims=True))
#loss = contrastive_loss(labels,distance)


####################################################################################################
######################################  Entity linking    ##########################################
with tf.variable_scope("nel") as scope:
  # state = mention_model._initial_state
  batch_size = nel_config.batch_size 
  num_steps = nel_config.num_steps
  mention_output = mention_model.output
  mention_state = mention_model.state
  gold_output = gold_entity_model.output
  gold_state = gold_entity_model.state
  corrupted_output = corrupted_entity_model.output
  corrupted_state = corrupted_entity_model.state

  normalize_mention = tf.nn.l2_normalize(mention_output, 0)               
  normalize_gold_entity = tf.nn.l2_normalize(gold_output, 0)
  normalize_corrupted_entity = tf.nn.l2_normalize(corrupted_output, 0)
  cosine_similarity = tf.reduce_sum(tf.multiply(normalize_mention,normalize_gold_entity))
  cosine_corrupted_similarities = tf.reduce_sum(tf.multiply(normalize_mention,normalize_corrupted_entity))
  pdb.set_trace() 
  distance = tf.maximum(0.0, 1 - cosine_similarity + cosine_corrupted_similarities)
  cost = tf.reduce_sum(distance)/(batch_size)
  pdb.set_trace()
  nel_optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)
  #nel_optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001).minimize(cost)

  pdb.set_trace() 
  
  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
  optimizer = tf.train.GradientDescentOptimizer(mention_model._lr)
  train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=tf.contrib.framework.get_or_create_global_step())
  
  sv = tf.train.Supervisor(logdir=FLAGS.save_path)
  with sv.managed_session() as session:
    tvars[0].eval(session)
  #softmax_w = tf.get_variable(
  #    "softmax_w", [size, vocab_size], dtype=data_type())
  #softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
   
  #mention_model._cost = cost = tf.reduce_sum(loss) / batch_size
  #mention_model._final_state = state
  #mention_model.softmax_w = softmax_w
  #mention_model.softmax_b = softmax_b


print("Tout va bien je vais bien")
######################### Epoch for NEL ############################################
####################################################################################

sv = tf.train.Supervisor(logdir=FLAGS.save_path)
print("Starting session ...")
with sv.managed_session() as session:                                                                         
  start_time = time.time()
  i = 0 
  lr_decay = nel_config.lr_decay ** max(i + 1 - nel_config.max_epoch, 0.0)
  # mention_model.assign_lr(session, config.learning_rate * lr_decay)
  costs = 0.0
  iters = 0
  verbose = True
  fetches = {
      #"cost": model.cost,
      "cost": cost,
      "final_mentions_state": mention_model._final_state,
      "final_entities_state": gold_entity_model._final_state
  }
  #eval_op = mention_model._train_op
  fetches["eval_op"] = nel_optimizer 
  
  print("beginning steps ...")
  for step in range(mention_model.input.epoch_size):
    vals = session.run(fetches)
    cost = vals["cost"]
    pdb.set_trace()
    mentionsState = vals["final_mentions_state"]
    goldState = vals["final_entities_state"]
    costs += cost
    iters += mention_model.input.num_steps
    print("For parameters")
    #pdb.set_trace()
    #if verbose and step % (mention_model.input.epoch_size // 10) == 10:
    #  print("%.3f perplexity: %.3f speed: %.0f wps" %
    #        (step * 1.0 / mention_model.input.epoch_size, np.exp(costs / iters),
    #         iters * mention_model.input.batch_size / (time.time() - start_time)))
  
    print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / mention_model.input.epoch_size, np.exp(costs / iters),
        iters * mention_model.input.batch_size / (time.time() - start_time)))
  

print(np.exp(costs / iters))


####################################################################################
####################################################################################

pdb.set_trace()

def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data
  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1
  
  size = config.hidden_size
  vocab_size = config.vocab_size
  
  ############################ 
  print("loading data")
  train_input = PTBInput(config=config, data=train_data, name="TrainInput")
  



  #initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
  initializer = tf.random_uniform_initializer(0.5,0.5)


  with tf.variable_scope("Model", reuse=None, initializer=initializer):      
    mention_model = BNRModel(is_training=True, config=config, input_=train_input)
    # state = mention_model._initial_state
    batch_size = config.batch_size 
    num_steps = config.num_steps
    output = mention_model.output
    state = mention_model.state
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(mention_model._input.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    mention_model._cost = cost = tf.reduce_sum(loss) / batch_size
    mention_model._final_state = state
    mention_model.softmax_w = softmax_w
    mention_model.softmax_b = softmax_b
   
    ###################################################################################
    ######################## Learning : painful version  ##############################
    mention_model._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(mention_model._lr)
    mention_model._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())
                                                                                 
    mention_model._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    mention_model._lr_update = tf.assign(mention_model._lr, mention_model._new_lr)   
    ####################################################################################
    ####################################################################################
    optimizer1 = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)
    print("For output")






  ########### Run epoch    ##############
  #######################################
  #with tf.Graph().as_default():
  sv = tf.train.Supervisor(logdir=FLAGS.save_path)
  print("Starting session ...")
  with sv.managed_session() as session:                                                                         
    start_time = time.time()
    i = 0 
    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
    # mention_model.assign_lr(session, config.learning_rate * lr_decay)
    costs = 0.0
    iters = 0
    verbose = True
    state = session.run(mention_model.initial_state)
    fetches = {
        #"cost": model.cost,
        "cost": cost,
        "final_state": mention_model.final_state,
    }
    eval_op = mention_model._train_op
    fetches["eval_op"] = eval_op
    
    print("beginning steps ...")
    for step in range(mention_model.input.epoch_size):
      feed_dict = {}
      for i, (c, h) in enumerate(mention_model.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h
      
      print(state[0][0][0][:4])                                                                       
      vals = session.run(fetches, feed_dict)
      ############################################        
      pdb.set_trace()
      #session.run(optimizer1,feed_dict)
      ############################################
      cost = vals["cost"]
      state = vals["final_state"]
                                                                             
      costs += cost
      iters += mention_model.input.num_steps
      print("For parameters")
      if verbose and step % (mention_model.input.epoch_size // 10) == 10:
        print("%.3f perplexity: %.3f speed: %.0f wps" %
              (step * 1.0 / mention_model.input.epoch_size, np.exp(costs / iters),
               iters * mention_model.input.batch_size / (time.time() - start_time)))
  print(np.exp(costs / iters))
  pdb.set_trace()

####################################### Initial code working ###########################################
########################################################################################################

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        #m = PTBModel(is_training=True, config=config, input_=train_input)
        m = BNRModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
        #mvalid = BNRModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)
        #mtest = BNRModel(is_training=False, config=eval_config, input_=test_input)
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
        
        #train_perplexity = run_epoch(session, m, loss, eval_op=m.train_op, verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        #valid_perplexity = run_epoch(session, mvalid, loss)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      #test_perplexity = run_epoch(session, mtest)
      test_perplexity = run_epoch(session, mtest, loss)
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

##############################################################################################
##############################################################################################

if __name__ == "__main__":
  mentionsDataFrame, knowledgeDataFrame = parse_tac_kbp()
  tf.app.run()
