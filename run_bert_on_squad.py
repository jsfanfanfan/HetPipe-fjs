# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Run BERT on SQuAD."""

from functools import partial

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.run_squad import squad_data
from megatron.model import BertModel
from megatron.model.bert_model import bert_extended_attention_mask, bert_position_ids
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group


def SQuAd_post_language_model_processing(lm_output, binary_head, lm_labels):
    binary_logits = None
    if binary_head is not None:
        binary_logits = binary_head(lm_output)
    start_logits, end_logits = binary_logits.split(1, dim=-1)

    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)
    return start_logits, end_logits, lm_labels
    

class BertForQuestionAnswering(BertModel):
    
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.

    Outputs:
         Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
         position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    
    def __init__(self,
                 num_tokentypes=2,
                 add_binary_head=True,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        super(BertForQuestionAnswering, self).__init__(num_tokentypes, add_binary_head, parallel_output, pre_process, post_process)
    
    def forward(self, bert_model_input, attention_mask,
                tokentype_ids=None, lm_labels=None):

        extended_attention_mask = bert_extended_attention_mask(attention_mask)
        input_ids = bert_model_input
        position_ids = bert_position_ids(input_ids)

        lm_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            tokentype_ids=tokentype_ids
        )

        if self.post_process and self.add_binary_head:
            lm_output, pooled_output = lm_output
        else:
            pooled_output = None

        if self.post_process:
            return SQuAd_post_language_model_processing(lm_output, self.binary_head,
                                                  lm_labels)
        else:
            return lm_output

    def get_linear_layer(rows, columns, init_method):
        """Simple linear layer with weight initialization."""
        layer = torch.nn.Linear(rows, columns)
        init_method(layer.weight)
        with torch.no_grad():
            layer.bias.zero_()
        return layer

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building BERT model ...')

    args = get_args()
    num_tokentypes = 2 if args.bert_binary_head else 0
    model = BertForQuestionAnswering(
        num_tokentypes=num_tokentypes,
        add_binary_head=True,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process)

    return model


def get_batch(data_iterator):
    """Build the batch."""

    # Items and their type.
    '''
    input:
        input_ids: 
        segment_ids
        input_mask
    labels:
        start_position
        end_position
    '''
    keys = ['input_ids', 'segment_ids', 'start_position', 'end_position', 'input_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    input_ids = data_b['input_ids'].long()
    segment_ids = data_b['segment_ids'].long()
    start_position = data_b['start_position'].long()
    end_position = data_b['end_position'].long()
    input_mask = data_b['input_mask'].long()
   
    return input_ids, segment_ids, start_position, end_position, input_mask


def loss_func(output_tensor):

    #####---------------------loss-----------------------#####
    #                                                    #####
    #         SQuAd loss = start_loss + end_loss         #####
    ####-------------------------------------------------#####
    start_logits, end_logits, lm_labels = output_tensor
    start_pos, end_pos = lm_labels
    start_loss = F.cross_entropy(start_logits, start_pos)
    end_loss = F.cross_entropy(end_logits , end_pos)
    lm_loss = (start_loss + end_loss) / 2

    loss = lm_loss
    averaged_losses = average_losses_across_data_parallel_group(
        [lm_loss])
    return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()
    # Get the batch.
    timers('batch-generator').start()
    # tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = get_batch(
    #     data_iterator)
    input_ids, segment_ids, start_position, end_position, input_mask= get_batch(
        data_iterator)
    timers('batch-generator').stop()

    if not args.bert_binary_head:
        types = None

    # Forward pass through the model.
    output_tensor = model(input_ids, input_mask, tokentype_ids=segment_ids,
                          lm_labels=(start_position, end_position))


    return output_tensor, partial(loss_func)


def train_valid_test_datasets_provider(train_val_test_num_samples,seed=1234):
    args = get_args()
   
    """Build train, valid, and test datasets."""
    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ...')
    train_ds, valid_ds, test_ds = squad_data(train_file=args.data_path[0],cache_dir=args.cache_dir[0],vocab_file=args.vocab_file,seed=seed)
    print_rank_0("> finished creating BERT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
