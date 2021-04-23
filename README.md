# NTM
TensorFlow implementation of Neural Turing Machines (NTM); 




## Prerequisites

* Python 3.5
* TensorFlow 1.2.0
* NumPy

## Implementation of NTM

### Paper

Graves, Alex, Greg Wayne, and Ivo Danihelka. "[Neural turing machines.](https://arxiv.org/abs/1410.5401)" _arXiv preprint arXiv:1410.5401_ (2014).

### Usage

#### Class NTMCell()
All the model avalible in this repository are ready to use, see the folder /ntm --- they are encapsulated into classes `NTMCell`, and the usage is similar to `LSTMCell` in TensorFlow, so you can apply these models easily in other programs. The sample code is also provided

The usage of class `NTMCell` in `ntm/ntm_cell.py` is similar to `tf.contrib.rnn.BasicLSTMCell` in TensorFlow. The basic pseudocode is as follows:

```python
import ntm.ntm_cell as ntm_cell
cell = ntm_cell.NTMCell(
    rnn_size=200,           # Size of hidden states of controller 
    memory_size=128,        # Number of memory locations (N)
    memory_vector_dim=20,   # The vector size at each location (M)
    read_head_num=1,        # # of read head
    write_head_num=1,       # # of write head
    addressing_mode='content_and_location', # Address Mechanisms, 'content_and_location' or 'content'
    reuse=False,            # Whether to reuse the variable in the model (if the length of sequence is not fixed, you might need to build more than one model using the same variable, and this will be useful)
)
state = cell.zero_state(batch_size, tf.float32)
output_list = []
for t in range(seq_length):
    output, state = cell(input[i], state)
    output_list.append(output)
```

#### Train and Test

For Train the model use the phython console, and type following.

```
python copy_task.py
```
You can specify training options including parameters to the model via flags, such as `--model` (default is NTM), `--batch_size` and so on. See code for more detail in the slides.

To test the model use the following command into the python console:

```
python copy_task.py --mode test
```

You can specify testing options via flags such as `--test_seq_length`.

## Sample Outputs

Below are some sample outputs on the Copy and Associative Recall tasks. We replicated the hyperparameters from the [original paper](https://arxiv.org/abs/1410.5401) for the 2 tasks:

- Memory Size: 128 X 20
- Controller: LSTM - 100 units
- Optimizer: RMSProp - learning rate = 10^-4

The Copy task network was trained on sequences of length sampled from Uniform(1,20) with 8-dimensional random bit vectors. The Associative Recall task network was trained on sequences with the number of items sampled from Uniform(2,6) each item consisted of 3 6-dimensional random bit vectors.
