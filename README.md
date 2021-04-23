# NTM
TensorFlow implementation of Neural Turing Machines (NTM); 

The models are ready to use -- they are encapsulated into classes `NTMCell` and `MANNCell`, and the usage is similar to `LSTMCell` in TensorFlow, so you can apply these models easily in other programs. The sample code is also provided

## Prerequisites

* Python 3.5
* TensorFlow 1.2.0
* NumPy

## Implementation of NTM

### Paper

Graves, Alex, Greg Wayne, and Ivo Danihelka. "[Neural turing machines.](https://arxiv.org/abs/1410.5401)" _arXiv preprint arXiv:1410.5401_ (2014).

### Usage

#### Class NTMCell()

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

To train the model, run:

```
python copy_task.py
```
You can specify training options including parameters to the model via flags, such as `--model` (default is NTM), `--batch_size` and so on. See code for more detail.

To test the model, run:

```
python copy_task.py --mode test
```

You can specify testing options via flags such as `--test_seq_length`.

### Result (Copy task)

![](images/copy_task_head.png) | ![](images/copy_task_loss.png)
---|---
Vector of weighting (left: read vector; right: write vector; shift range: 1) | Training loss
