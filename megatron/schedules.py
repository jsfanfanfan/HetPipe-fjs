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

from contextlib import contextmanager
from matplotlib.pyplot import cool
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_num_microbatches
from megatron import get_timers
from megatron import mpu
from megatron import p2p_communication
from megatron.utils import unwrap_model
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module

#-------Add Rng Restore -----------------#
class Rng_store:

    def __init__(self, device=torch.device('cpu')):
        self.device = device
        # # devices list for `fork_rng` method
        self.devices = [self.device] if self.device.type == 'cuda' else []
    def stash_rng_state(self):
        """ Stash RNG state """
        self.cpu_rng_state = torch.get_rng_state()
        with torch.cuda.device(self.device):
            self.gpu_rng_state = torch.cuda.get_rng_state()
        # if self.device.type == 'cuda':
        #     with torch.cuda.device(self.device):
        #         self.gpu_rng_state = torch.cuda.get_rng_state()
        # else:
        #     self.gpu_rng_state = None
        # self.state[micro_batch_index] = (cpu_rng_state, gpu_rng_state)
    def restore_rng_state(self):
        cpu_rng_state, gpu_rng_state = self.cpu_rng_state, self.gpu_rng_state
        torch.set_rng_state(cpu_rng_state)
        if not (gpu_rng_state is None):
            torch.cuda.set_rng_state(gpu_rng_state, self.device)
            
    def clear_state(self):
        self.state.clear()

#---------End----------------------------#

def get_forward_backward_func():
    args = get_args()
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if args.virtual_pipeline_model_parallel_size is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
            assert get_num_microbatches() % args.pipeline_model_parallel_size == 0, \
                'number of microbatches is not divisible by pipeline-parallel ' \
                'size when using interleaved schedule'
        else:
            # forward_backward_func = DDP_WFBP
            # forward_backward_func = schedule_following_stage
            # forward_backward_func = forward_backward_pipelining_with_WFBP_without_interleaving
            if mpu.get_pipeline_model_parallel_world_size() == 8:
                forward_backward_func = forward_backward_pipelining_with_PreRecompute_batch8_WFBP_stage8
            elif args.sprecompute and mpu.get_pipeline_model_parallel_world_size() != 8:
                if get_num_microbatches() == 8:
                    if args.DDP_impl == 'torch':
                        forward_backward_func = forward_backward_pipelining_with_PreRecompute_batch8_WFBP
                    else:
                        forward_backward_func = forward_backward_pipelining_with_PreRecompute_batch8
                else:
                    forward_backward_func = forward_backward_pipelining_with_speedup_forward
                # if get_num_microbatches == 8:
                #     forward_backward_func = forward_backward_pipelining_with_PreRecompute_batch8
                # else:
                #     Schedule = PipeSchedule(get_num_microbatches(), mpu.get_pipeline_model_parallel_world_size(), mpu.get_pipeline_model_parallel_rank())
                #     # forward_backward_func = forward_backward_pipelining_with_speedup_forward
                #     forward_backward_func = Schedule.steps

            else:
                forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def forward_step(forward_step_func, data_iterator, model, input_tensor, losses_reduced, recompute=False):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    timers = get_timers()
    if mpu.get_tensor_model_parallel_rank() == 0 and not recompute:
        timers('pipeline-stage-{}-forward'.format(mpu.get_pipeline_model_parallel_rank())).start()
        timers('pipeline-stage-{}'.format(mpu.get_pipeline_model_parallel_rank())).start()

    timers('forward-compute').start()
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)
    output_tensor, loss_func = forward_step_func(data_iterator, model, recompute)
    if mpu.is_pipeline_last_stage():
        # print("output_tensor in forward_step",output_tensor)
        output_tensor = loss_func(output_tensor)
        loss, loss_reduced = output_tensor
        output_tensor = loss / get_num_microbatches()
        losses_reduced.append(loss_reduced)
        # print("output_tensor in after forward_step",output_tensor.shape)
    timers('forward-compute').stop()
    
    if mpu.get_tensor_model_parallel_rank() == 0 and not recompute:
        timers('pipeline-stage-{}-forward'.format(mpu.get_pipeline_model_parallel_rank())).stop()
        timers('pipeline-stage-{}'.format(mpu.get_pipeline_model_parallel_rank())).stop()

    return output_tensor


def backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""
    args = get_args()

    timers = get_timers()
    if mpu.get_tensor_model_parallel_rank() == 0:
        timers('pipeline-stage-{}-backward'.format(mpu.get_pipeline_model_parallel_rank())).start()   
        timers('pipeline-stage-{}'.format(mpu.get_pipeline_model_parallel_rank())).start() 

    timers('backward-compute').start()

    # Retain the grad on the input_tensor.
    if input_tensor is not None:
        input_tensor.retain_grad()

    # Backward pass.
    if output_tensor_grad is None:
        output_tensor = optimizer.scale_loss(output_tensor)
    torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)

    # Collect the grad of the input_tensor.
    input_tensor_grad = None
    if input_tensor is not None:
        input_tensor_grad = input_tensor.grad

    timers('backward-compute').stop()
    
    if mpu.get_tensor_model_parallel_rank() == 0:
        timers('pipeline-stage-{}-backward'.format(mpu.get_pipeline_model_parallel_rank())).stop()    
        timers('pipeline-stage-{}'.format(mpu.get_pipeline_model_parallel_rank())).stop()
    return input_tensor_grad


@contextmanager
def dummy_handler():
    try:
        yield
    finally:
        pass


def forward_backward_no_pipelining(forward_step_func, data_iterator, model,
                                   optimizer, timers, forward_only):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses."""
    assert len(model) == 1
    model = model[0]

    context_handler = dummy_handler
    if isinstance(model, torchDDP):
        context_handler = model.no_sync

    losses_reduced = []
    input_tensor, output_tensor_grad = None, None
    with context_handler():
        for i in range(get_num_microbatches() - 1):
            output_tensor = forward_step(forward_step_func, data_iterator, model,
                                         input_tensor, losses_reduced)
            if not forward_only:
                backward_step(optimizer, input_tensor, output_tensor,
                              output_tensor_grad)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = forward_step(forward_step_func, data_iterator, model,
                                 input_tensor, losses_reduced)
    if not forward_only:
        backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad)

    return losses_reduced


def forward_backward_pipelining_with_interleaving(forward_step_func, data_iterator, model,
                                                  optimizer, timers, forward_only):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    losses_reduced = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = mpu.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = mpu.get_pipeline_model_parallel_rank()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    num_microbatches = get_num_microbatches() * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if get_num_microbatches() == pipeline_parallel_size:
            num_warmup_microbatches = num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = \
                (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (
                num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches,
                                          num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = (num_model_chunks - model_chunk_id - 1)
        return model_chunk_id

    def forward_step_helper(microbatch_id):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        mpu.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # forward step
        if mpu.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == \
                    len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        output_tensor = forward_step(forward_step_func,
                                     data_iterator[model_chunk_id],
                                     model[model_chunk_id],
                                     input_tensor, losses_reduced)
        output_tensors[model_chunk_id].append(output_tensor)

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        mpu.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if mpu.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = \
            backward_step(optimizer,
                          input_tensor,
                          output_tensor,
                          output_tensor_grad)

        return input_tensor_grad

    # Run warmup forward passes.
    mpu.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(
        p2p_communication.recv_forward(timers=timers))
    for k in range(num_warmup_microbatches):
        output_tensor = forward_step_helper(k)

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k+1, forward=True)
        recv_prev = True
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if mpu.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if k == (num_warmup_microbatches - 1) and not forward_only and \
                not all_warmup_microbatches:
            input_tensor_grad = None
            recv_next = True
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                recv_next = False
            input_tensor, output_tensor_grad = \
                p2p_communication.send_forward_backward_recv_forward_backward(
                        output_tensor, input_tensor_grad,
                        recv_prev=recv_prev, recv_next=recv_next,
                        timers=timers)
            output_tensor_grads[num_model_chunks-1].append(output_tensor_grad)
        else:
            input_tensor = \
                p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, timers=timers)
        input_tensors[next_forward_model_chunk_id].append(input_tensor)

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches
        output_tensor = forward_step_helper(forward_k)

        # Backward pass.
        backward_k = k
        input_tensor_grad = backward_step_helper(backward_k)

        # Send output_tensor and input_tensor_grad, receive input_tensor
        # and output_tensor_grad.

        # Determine if current stage has anything to send in either direction,
        # otherwise set tensor to None.
        forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        mpu.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
        if mpu.is_pipeline_last_stage():
            output_tensor = None

        backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
        mpu.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
        if mpu.is_pipeline_first_stage():
            input_tensor_grad = None

        # Determine if peers are sending, and where in data structure to put
        # received tensors.
        recv_prev = True
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            # First stage is ahead of last stage by (pipeline_parallel_size - 1).
            next_forward_model_chunk_id = get_model_chunk_id(
                forward_k - (pipeline_parallel_size - 1), forward=True)
            if next_forward_model_chunk_id == (num_model_chunks - 1):
                recv_prev = False
            next_forward_model_chunk_id += 1
        else:
            next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1,
                                                             forward=True)

        recv_next = True
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
            next_backward_model_chunk_id = get_model_chunk_id(
                backward_k - (pipeline_parallel_size - 1), forward=False)
            if next_backward_model_chunk_id == 0:
                recv_next = False
            next_backward_model_chunk_id -= 1
        else:
            next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1,
                                                              forward=False)

        # If last iteration, don't receive; we already received one extra
        # before the start of the for loop.
        if k == (num_microbatches_remaining - 1):
            recv_prev = False

        # Communicate tensors.
        input_tensor, output_tensor_grad = \
            p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor, input_tensor_grad,
                    recv_prev=recv_prev, recv_next=recv_next,
                    timers=timers)

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(
                output_tensor_grad)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks-1].append(
                p2p_communication.recv_backward(timers=timers))
        for k in range(num_microbatches_remaining, num_microbatches):
            input_tensor_grad = backward_step_helper(k)
            next_backward_model_chunk_id = get_model_chunk_id(k+1, forward=False)
            recv_next = True
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (num_microbatches - 1):
                recv_next = False
            output_tensor_grads[next_backward_model_chunk_id].append(
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, timers=timers))

    return losses_reduced


def forward_backward_pipelining_without_interleaving(forward_step_func, data_iterator,
                                                     model, optimizer, timers,
                                                     forward_only):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    timers = get_timers()

    assert len(model) == 1
    model = model[0]

    # Compute number of warmup microbatches.
    num_microbatches = get_num_microbatches()
    num_warmup_microbatches = \
        (mpu.get_pipeline_model_parallel_world_size() -
         mpu.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    losses_reduced = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        input_tensor = p2p_communication.recv_forward(timers=timers)
        output_tensor = forward_step(forward_step_func, data_iterator, model,
                                     input_tensor, losses_reduced)
        p2p_communication.send_forward(output_tensor, timers=timers)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communication.recv_forward(timers=timers)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = (i == (num_microbatches_remaining - 1))

        output_tensor = forward_step(forward_step_func, data_iterator, model,
                                     input_tensor, losses_reduced)
        if forward_only:
            p2p_communication.send_forward(output_tensor, timers=timers)

            if not last_iteration:
                input_tensor = p2p_communication.recv_forward(timers=timers)

        else:
            output_tensor_grad = \
                p2p_communication.send_forward_recv_backward(output_tensor,
                                                             timers=timers)

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            input_tensor_grad = \
                backward_step(optimizer, input_tensor, output_tensor,
                              output_tensor_grad)

            if last_iteration:
                input_tensor = None
                p2p_communication.send_backward(input_tensor_grad, timers=timers)
            else:
                input_tensor = \
                    p2p_communication.send_backward_recv_forward(
                        input_tensor_grad, timers=timers)

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = p2p_communication.recv_backward(timers=timers)

            input_tensor_grad = \
                backward_step(optimizer, input_tensor, output_tensor,
                              output_tensor_grad)

            p2p_communication.send_backward(input_tensor_grad, timers=timers)

    return losses_reduced

##————————————————————speed up the forward of recomputation -------------------##
def forward_backward_pipelining_with_speedup_forward(forward_step_func, data_iterator,
                                                     model, optimizer, timers,
                                                     forward_only):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    args = get_args()
    timers = get_timers()

    assert len(model) == 1
    model = model[0]

    # Compute number of warmup microbatches.
    num_microbatches = get_num_microbatches()
    num_warmup_microbatches = \
        (mpu.get_pipeline_model_parallel_world_size() -
         mpu.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
        rng_states = []
    losses_reduced = []
    context_handler = dummy_handler
    if isinstance(model, torchDDP):
        context_handler = model.no_sync
    # Run warmup forward passes.
    with context_handler():
        for i in range(num_warmup_microbatches):
            input_tensor = p2p_communication.recv_forward(timers=timers)
            if mpu.is_pipeline_first_stage() and i >= 1:
                # rng_state = Rng_store(torch.device("cuda", torch.cuda.current_device()))
                # rng_state.stash_rng_state()
                # if not forward_only:
                #     rng_states.append(rng_state)
                # if torch.distributed.get_rank() == 0:
                    # print("#--#"*10, flush=True)
                    # print("warmup_forward",rng_state.gpu_rng_state, flush=True)
                    # print("#--#"*10, flush=True) 
                output_tensor = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, losses_reduced, recompute=args.sprecompute)
                # print("*--*"*10)
                # print("warmup, the stage: {}, the microbatch is {}, recompute".format( mpu.get_pipeline_model_parallel_rank(), i))
                # print("*--*"*10)
            else:
                output_tensor = forward_step(forward_step_func, data_iterator, model,
                                            input_tensor, losses_reduced)
                # print("*--*"*10)
                # print("warmup, the stage: {}, the microbatch is {}, no_recompute".format( mpu.get_pipeline_model_parallel_rank(), i))
                # print("*--*"*10)
            p2p_communication.send_forward(output_tensor, timers=timers)

            if not forward_only:
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)
                

        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_microbatches_remaining > 0:
            input_tensor = p2p_communication.recv_forward(timers=timers)

        # Run 1F1B in steady state.
        for i in range(num_microbatches_remaining):
            last_iteration = (i == (num_microbatches_remaining - 1))

            ####   add recompute ######
            if mpu.is_pipeline_last_stage():
                output_tensor = forward_step(forward_step_func, data_iterator, model,
                                            input_tensor, losses_reduced)
                # print("*--*"*10)
                # print("1F1B, the stage: {}, the microbatch is {}, no_recompute".format( mpu.get_pipeline_model_parallel_rank(), i))
                # print("*--*"*10)
            elif mpu.get_pipeline_model_parallel_rank() == 2 and not last_iteration:
                output_tensor = forward_step(forward_step_func, data_iterator, model,
                                            input_tensor, losses_reduced)
                # print("*--*"*10)
                # print("1F1B, the stage: {}, the microbatch is {}, no_recompute".format( mpu.get_pipeline_model_parallel_rank(), i))
                # print("*--*"*10)
            else:
                # rng_state = Rng_store(torch.device("cuda", torch.cuda.current_device()))
                # rng_state.stash_rng_state()
                # rng_states.append(rng_state)
                # if torch.distributed.get_rank() == 0:
                #     print("#--#"*10, flush=True)
                #     print("1F1B_forward",rng_state.gpu_rng_state, flush=True)
                #     print("#--#"*10, flush=True) 
                output_tensor = forward_step(forward_step_func, data_iterator, model,
                                            input_tensor, losses_reduced, recompute=args.sprecompute)
                # print("*--*"*10)
                # print("1F1B, the stage: {}, the microbatch is {}, recompute".format( mpu.get_pipeline_model_parallel_rank(), i))
                # print("*--*"*10)
            ####  end  #########
            if forward_only:
                p2p_communication.send_forward(output_tensor, timers=timers)

                if not last_iteration:
                    input_tensor = p2p_communication.recv_forward(timers=timers)

            else:
                output_tensor_grad = \
                    p2p_communication.send_forward_recv_backward(output_tensor,
                                                                timers=timers)

                # Add input_tensor and output_tensor to end of list.
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)
                # Pop input_tensor and output_tensor from the start of the list for
                # the backward pass.
                input_tensor = input_tensors.pop(0)
                output_tensor = output_tensors.pop(0)

                input_tensor_grad = \
                    backward_step(optimizer, input_tensor, output_tensor,
                                output_tensor_grad)

                if last_iteration:
                    input_tensor = None
                    p2p_communication.send_backward(input_tensor_grad, timers=timers)
                else:
                    input_tensor = \
                        p2p_communication.send_backward_recv_forward(
                            input_tensor_grad, timers=timers)

    # Run cooldown backward passes.
    if not forward_only:
        with context_handler():
            for i in range(num_warmup_microbatches-1):
                input_tensor = input_tensors.pop(0)
                # true_output_tensor = output_tensors.pop(0)
                # rng_state = rng_states.pop(0)
                # if torch.distributed.get_rank() == 0:
                #     print("#--#"*10, flush=True)
                #     print("cooldown", rng_state.gpu_rng_state, flush=True)
                #     print("#--#"*10, flush=True) 
                # with torch.random.fork_rng(devices=rng_state.device):
                #     rng_state.restore_rng_state()
                #     output_tensor = forward_step(forward_step_func, data_iterator, model,
                #                                 input_tensor, losses_reduced)
                # rng_state.restore_rng_state()
                output_tensor = forward_step(forward_step_func, data_iterator, model,
                                            input_tensor, losses_reduced)
                # print("*-*-"*10)    
                # print(torch.allclose(true_output_tensor,output_tensor))
                # print(true_output_tensor,output_tensor)
                # print("*-*-"*10)                              
                output_tensor_grad = p2p_communication.recv_backward(timers=timers)

                input_tensor_grad = \
                    backward_step(optimizer, input_tensor, output_tensor,
                                output_tensor_grad)

                p2p_communication.send_backward(input_tensor_grad, timers=timers)
        if not mpu.is_pipeline_last_stage():
            input_tensor = input_tensors.pop(0)
            # true_output_tensor = output_tensors.pop(0)
            # rng_state = rng_states.pop(0)
            # # with torch.random.fork_rng(devices=rng_state.device):
            # rng_state.restore_rng_state()
            output_tensor = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, losses_reduced)                           
            # print("*-*-"*10)
            # print(torch.allclose(true_output_tensor,output_tensor))
            # print(true_output_tensor,output_tensor)
            # print("*-*-"*10)
            # if torch.distributed.get_rank() == 0:
            #     print("#--#"*10, flush=True)
            #     print("cooldown",rng_state.gpu_rng_state, flush=True)
            #     print("#--#"*10, flush=True)     
            output_tensor_grad = p2p_communication.recv_backward(timers=timers)

            input_tensor_grad = \
                backward_step(optimizer, input_tensor, output_tensor,
                            output_tensor_grad)

            p2p_communication.send_backward(input_tensor_grad, timers=timers)

    return losses_reduced

def DDP_WFBP(forward_step_func, data_iterator,
                                        model, optimizer, timers,
                                        forward_only):
    timers = get_timers()

    assert len(model) == 1
    model = model[0]
    schedule_guide=[]
    num_microbatches = get_num_microbatches()
    for i in range(mpu.get_pipeline_model_parallel_world_size()):
        num_warmup_microbatches = \
        min((mpu.get_pipeline_model_parallel_world_size() - i - 1), num_microbatches)
        num_1f1b_microbatches = num_microbatches - num_warmup_microbatches
        schedule_guide.append({"warm_up":num_warmup_microbatches, "1f1b":num_1f1b_microbatches})
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    losses_reduced = []

    context_handler = dummy_handler
    if isinstance(model, torchDDP):
        context_handler = model.no_sync

    stage_guide = schedule_guide[mpu.get_pipeline_model_parallel_rank()]
    with context_handler():
        # print(context_handler)
        for i in range(stage_guide["warm_up"]):
            input_tensor = p2p_communication.recv_forward(timers=timers)
            output_tensor = forward_step(forward_step_func, data_iterator, model,
                                            input_tensor, losses_reduced)
            p2p_communication.send_forward(output_tensor, timers=timers)
            if not forward_only:
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)
        
        if stage_guide["1f1b"] > 0:
            input_tensor = p2p_communication.recv_forward(timers=timers)
            

        for i in range(stage_guide["1f1b"]):
            last_iteration = (i == (stage_guide["1f1b"] - 1))
            output_tensor = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, losses_reduced)
            if forward_only:
                p2p_communication.send_forward(output_tensor, timers=timers)
                if not last_iteration:
                    input_tensor = p2p_communication.recv_forward(timers=timers)
                # input_tensor = p2p_communication.recv_forward(timers=timers)
            else:
                output_tensor_grad = \
                    p2p_communication.send_forward_recv_backward(output_tensor,
                                                                timers=timers)

                # Add input_tensor and output_tensor to end of list.
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)

                # Pop input_tensor and output_tensor from the start of the list for
                # the backward pass.
                input_tensor = input_tensors.pop(0)
                output_tensor = output_tensors.pop(0)

                input_tensor_grad = \
                    backward_step(optimizer, input_tensor, output_tensor,
                                output_tensor_grad)

                if last_iteration:
                    input_tensor = None
                    p2p_communication.send_backward(input_tensor_grad, timers=timers)
                else:
                    input_tensor = \
                        p2p_communication.send_backward_recv_forward(
                            input_tensor_grad, timers=timers)
            # input_tensor = \
            #     p2p_communication.send_backward_recv_forward(
            #         input_tensor_grad, timers=timers)

    # if forward_only:
    #         p2p_communication.send_forward(output_tensor, timers=timers)
    #         # if not last_iteration:
    #         #     input_tensor = p2p_communication.recv_forward(timers=timers)
    # else:
    #     output_tensor = forward_step(forward_step_func, data_iterator, model,
    #                                     input_tensor, losses_reduced)
    #     output_tensor_grad = \
    #             p2p_communication.send_forward_recv_backward(output_tensor,
    #                                                          timers=timers)

    #     # Add input_tensor and output_tensor to end of list.
    #     input_tensors.append(input_tensor)
    #     output_tensors.append(output_tensor)

    #     # Pop input_tensor and output_tensor from the start of the list for
    #     # the backward pass.
    #     output_tensor = output_tensors.pop(0)
    #     input_tensor = input_tensors.pop(0)
    #     input_tensor_grad = \
    #         backward_step(optimizer, input_tensor, output_tensor,
    #                         output_tensor_grad)
    #     p2p_communication.send_backward(input_tensor_grad, timers=timers)

    # Run cooldown backward passes.
    if not forward_only:
        # with context_handler():
            # print(context_handler,"print(context_handler)")
        for i in range(stage_guide["warm_up"]):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            if mpu.is_pipeline_first_stage() and i == stage_guide["warm_up"]-1:
                model.reducer.prepare_for_backward([])    
            output_tensor_grad = p2p_communication.recv_backward(timers=timers)

            input_tensor_grad = \
                backward_step(optimizer, input_tensor, output_tensor,
                            output_tensor_grad)

            p2p_communication.send_backward(input_tensor_grad, timers=timers)


    return losses_reduced



##————————————————————PreRecompute -------------------##
def forward_backward_pipelining_with_PreRecompute(forward_step_func, data_iterator,
                                                     model, optimizer, timers,
                                                     forward_only):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    args = get_args()
    timers = get_timers()

    assert len(model) == 1
    model = model[0]

    # Compute number of warmup microbatches.
    num_microbatches = get_num_microbatches()
    num_warmup_microbatches = \
        (mpu.get_pipeline_model_parallel_world_size() -
         mpu.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
        rng_states = []
    losses_reduced = []
    context_handler = dummy_handler
    if isinstance(model, torchDDP):
        context_handler = model.no_sync
    # Run warmup forward passes.
    with context_handler():
        for i in range(num_warmup_microbatches):
            input_tensor = p2p_communication.recv_forward(timers=timers)
            if mpu.is_pipeline_first_stage() and i >= 1:
                # rng_state = Rng_store(torch.device("cuda", torch.cuda.current_device()))
                # rng_state.stash_rng_state()
                # if not forward_only:
                #     rng_states.append(rng_state)
                # if torch.distributed.get_rank() == 0:
                #     print("#--#"*10, flush=True)
                #     print("warmup_forward",rng_state.gpu_rng_state, flush=True)
                #     print("#--#"*10, flush=True) 
                output_tensor = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, losses_reduced, recompute=args.sprecompute)
                # print("*--*"*10)
                # print("warmup, the stage: {}, the microbatch is {}, recompute".format( mpu.get_pipeline_model_parallel_rank(), i))
                # print("*--*"*10)
            else:
                output_tensor = forward_step(forward_step_func, data_iterator, model,
                                            input_tensor, losses_reduced)
                # print("*--*"*10)
                # print("warmup, the stage: {}, the microbatch is {}, no_recompute".format( mpu.get_pipeline_model_parallel_rank(), i))
                # print("*--*"*10)
            p2p_communication.send_forward(output_tensor, timers=timers)

            if not forward_only:
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)
                

        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_microbatches_remaining > 0:
            input_tensor = p2p_communication.recv_forward(timers=timers)

        # Run 1F1B in steady state.
        for i in range(num_microbatches_remaining):
            last_iteration = (i == (num_microbatches_remaining - 1))

            ####   add recompute ######
            if mpu.is_pipeline_last_stage():
                output_tensor = forward_step(forward_step_func, data_iterator, model,
                                            input_tensor, losses_reduced)
                # print("*--*"*10)
                # print("1F1B, the stage: {}, the microbatch is {}, no_recompute".format( mpu.get_pipeline_model_parallel_rank(), i))
                # print("*--*"*10)
            elif mpu.get_pipeline_model_parallel_rank() == 2 and not last_iteration:
                output_tensor = forward_step(forward_step_func, data_iterator, model,
                                            input_tensor, losses_reduced)
                # print("*--*"*10)
                # print("1F1B, the stage: {}, the microbatch is {}, no_recompute".format( mpu.get_pipeline_model_parallel_rank(), i))
                # print("*--*"*10)
            else:
                # rng_state = Rng_store(torch.device("cuda", torch.cuda.current_device()))
                # rng_state.stash_rng_state()
                # rng_states.append(rng_state)
                # if torch.distributed.get_rank() == 0:
                #     print("#--#"*10, flush=True)
                #     print("1F1B_forward",rng_state.gpu_rng_state, flush=True)
                #     print("#--#"*10, flush=True) 
                output_tensor = forward_step(forward_step_func, data_iterator, model,
                                            input_tensor, losses_reduced, recompute=args.sprecompute)
                # print("*--*"*10)
                # print("1F1B, the stage: {}, the microbatch is {}, recompute".format( mpu.get_pipeline_model_parallel_rank(), i))
                # print("*--*"*10)
            ####  end  #########
            if forward_only:
                p2p_communication.send_forward(output_tensor, timers=timers)

                if not last_iteration:
                    input_tensor = p2p_communication.recv_forward(timers=timers)

            else:
                output_tensor_grad = \
                    p2p_communication.send_forward_recv_backward(output_tensor,
                                                                timers=timers)

                # Add input_tensor and output_tensor to end of list.
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)
                # Pop input_tensor and output_tensor from the start of the list for
                # the backward pass.
                input_tensor = input_tensors.pop(0)
                output_tensor = output_tensors.pop(0)

                input_tensor_grad = \
                    backward_step(optimizer, input_tensor, output_tensor,
                                output_tensor_grad)

                if last_iteration:
                    input_tensor = None
                    p2p_communication.send_backward(input_tensor_grad, timers=timers)
                else:
                    input_tensor = \
                        p2p_communication.send_backward_recv_forward(
                            input_tensor_grad, timers=timers)

    # Run cooldown backward passes.
    if not forward_only:
        with context_handler():
            for i in range(num_warmup_microbatches-1):
                input_tensor = input_tensors.pop(0)
                # true_output_tensor = output_tensors.pop(0)
                # rng_state = rng_states.pop(0)
                # if torch.distributed.get_rank() == 0:
                #     print("#--#"*10, flush=True)
                #     print("cooldown", rng_state.gpu_rng_state, flush=True)
                #     print("#--#"*10, flush=True) 
                # with torch.random.fork_rng(devices=rng_state.device):
                #     rng_state.restore_rng_state()
                #     output_tensor = forward_step(forward_step_func, data_iterator, model,
                #                                 input_tensor, losses_reduced)
                # rng_state.restore_rng_state()
                output_tensor = forward_step(forward_step_func, data_iterator, model,
                                            input_tensor, losses_reduced)
                # print("*-*-"*10)    
                # print(torch.allclose(true_output_tensor,output_tensor))
                # print(true_output_tensor,output_tensor)
                # print("*-*-"*10)                              
                output_tensor_grad = p2p_communication.recv_backward(timers=timers)

                input_tensor_grad = \
                    backward_step(optimizer, input_tensor, output_tensor,
                                output_tensor_grad)

                p2p_communication.send_backward(input_tensor_grad, timers=timers)
        if not mpu.is_pipeline_last_stage():
            input_tensor = input_tensors.pop(0)
            # true_output_tensor = output_tensors.pop(0)
            # rng_state = rng_states.pop(0)
            # # with torch.random.fork_rng(devices=rng_state.device):
            # rng_state.restore_rng_state()
            output_tensor = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, losses_reduced)                           
            # print("*-*-"*10)
            # print(torch.allclose(true_output_tensor,output_tensor))
            # print(true_output_tensor,output_tensor)
            # print("*-*-"*10)
            # if torch.distributed.get_rank() == 0:
            #     print("#--#"*10, flush=True)
            #     print("cooldown",rng_state.gpu_rng_state, flush=True)
            #     print("#--#"*10, flush=True)     
            output_tensor_grad = p2p_communication.recv_backward(timers=timers)

            input_tensor_grad = \
                backward_step(optimizer, input_tensor, output_tensor,
                            output_tensor_grad)

            p2p_communication.send_backward(input_tensor_grad, timers=timers)

    return losses_reduced

class PipeSchedule:
    def __init__(self, micro_batches, stages, stage_id) -> None:
        self.micro_batches = micro_batches
        self.stages = stages
        self.stage_id = stage_id
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1
        self.pipe_buffers = {
            'inputs' : [],
            'outputs' : [],
            'grad':[]
        }
        for key in self.pipe_buffers:
            self.pipe_buffers[key].extend([None] * micro_batches)
    def _is_even(self, x):
        return x % 2 == 0
        
    def _is_odd(self, x):
        return x % 2 != 0
        
    def _step_to_micro_batch(self, step_id):
        # print("^&&&&^"*8)
        # print(step_id, self.stage_id)
        # print("^&&&&^"*8)
        if self._is_even(step_id) and self._is_even(self.stage_id):
            micro_batch_id = self._even_step_forward_id(step_id)
            is_forward = True

        elif self._is_odd(step_id) and self._is_odd(self.stage_id):
            micro_batch_id = self._odd_step_forward_id(step_id)
            is_forward = True

        elif self._is_even(step_id) and self._is_odd(self.stage_id):
            micro_batch_id = self._even_step_backward_id(step_id)
            is_forward = False

        elif self._is_odd(step_id) and self._is_even(self.stage_id):
            micro_batch_id = self._odd_step_backward_id(step_id)
            is_forward = False
        else:
            assert False

        return micro_batch_id, is_forward 

    def _even_step_forward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id(self, step_id):
        base = ((step_id - 1) // 2) - self.stages + 1
        micro_batch_id = int(base + self.stage_id // 2)
        return micro_batch_id

    def _valid_micro_batch(self, micro_batch_id):
        return 0 <= micro_batch_id < self.micro_batches

    def _valid_stage(self, stage_id):
        return 0 <= stage_id < self.stages

    def num_pipe_buffers(self):
        """As many buffers as the distance from this stage to the last stage.
        """
        buffers = min(self.stages - self.stage_id + 1, self.micro_batches)
        return max(2, buffers)


    def _buffer_idx(self, micro_batch_id):
        """Map a micro-batch index to a pipeline buffer index.

        This method uses a cyclic allocation strategy.

        Args:
            micro_batch_id (int): The micro-batch index relative to the beginning of the schedule.

        Returns:
            int: The index of the buffer that should store data.
        """
        assert self._valid_micro_batch(micro_batch_id)
        return micro_batch_id % self.num_pipe_buffers()
    
    def steps(self,forward_step_func, data_iterator,
                model, optimizer, timers, forward_only):
        losses_reduced = []
        args = get_args()
        timers = get_timers()

        assert len(model) == 1
        model = model[0]
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        grad = None
        for step_id in range(total_steps):
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)
            
            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)
            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.prev_stage):
                        if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                            self.prev_stage):
                            self.pipe_buffers['inputs'][curr_buffer] = p2p_communication.send_backward_recv_forward(input_grad)
        
                        else:
                            self.pipe_buffers['inputs'][curr_buffer] = p2p_communication.recv_forward(timers=timers)
                            
                elif self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                            self.prev_stage):
                        p2p_communication.send_backward(input_grad)
            else:
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                        self.next_stage):
                    if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.next_stage):
                        output_grad = p2p_communication.send_forward_recv_backward(self.pipe_buffers['outputs'][prev_buffer])
                    else:
                        p2p_communication.send_forward(self.pipe_buffers['outputs'][prev_buffer])
                elif self._valid_micro_batch(micro_batch_id) and self._valid_stage(self.next_stage):
                    self.pipe_buffers['outputs'][curr_buffer] = forward_step(forward_step_func, data_iterator, model,
                                        self.pipe_buffers['inputs'][curr_buffer], losses_reduced)
                    # print("^&&&&^"*8)
                    # print("curr_buffers:{},outputs.shape:{}".format(curr_buffer, self.pipe_buffers['outputs'][curr_buffer].shape)) 
                    # print("^&&&&^"*8)
                    output_grad = p2p_communication.recv_backward()
                    


            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    if self.micro_batches - micro_batch_id + self.stage_id < self.stages:
                        self.pipe_buffers['outputs'][curr_buffer] = forward_step(forward_step_func, data_iterator, model,
                                        self.pipe_buffers['inputs'][curr_buffer], losses_reduced, recompute=args.sprecompute)

                    else:
        
                        self.pipe_buffers['outputs'][curr_buffer] = forward_step(forward_step_func, data_iterator, model,
                                        self.pipe_buffers['inputs'][curr_buffer], losses_reduced)
                            # print("##"*8)
                else:
                    if mpu.is_pipeline_last_stage():
                        output_grad = None
                    input_grad = \
                    backward_step(optimizer, self.pipe_buffers['inputs'][curr_buffer], self.pipe_buffers['outputs'][curr_buffer],
                            output_grad)
            prev_micro_batch_id = micro_batch_id
        return losses_reduced   



def forward_backward_pipelining_with_PreRecompute_batch8(forward_step_func, data_iterator,
                                                     model, optimizer, timers,
                                                     forward_only):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    args = get_args()
    timers = get_timers()

    assert len(model) == 1
    model = model[0]

    # Compute number of warmup microbatches.
    num_microbatches = get_num_microbatches()
    num_warmup_microbatches = \
        (mpu.get_pipeline_model_parallel_world_size() -
         mpu.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    losses_reduced = []
   
    forward_index = 0
    backward_index = 0
    send_forward_index = 0
    input_tensors_buffer = [None]*10
    output_tensors_buffer = [None]*10
    s = [8,6,3,0]
    send_forward_guide = [6,3,1]
    stage_id = mpu.get_pipeline_model_parallel_rank()
    stages = mpu.get_pipeline_model_parallel_world_size()
    for i in range(s[stage_id]):
        # print("stage_id: {},warmup,i={}".format(stage_id,i))
        input_tensor = p2p_communication.recv_forward(timers=timers)
        input_tensors_buffer[forward_index] = input_tensor
        if 3*(stages-stage_id)-2 > 8 or i < 2*(stages-stage_id-1):
            output_tensors_buffer[forward_index] = forward_step(forward_step_func, data_iterator, model,
                                    input_tensor, losses_reduced, recompute=args.sprecompute)
        else:
            output_tensors_buffer[forward_index] = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, losses_reduced)
        forward_index += 1
        if i < send_forward_guide[stage_id]:
            p2p_communication.send_forward(output_tensors_buffer[send_forward_index], timers=timers)
            send_forward_index += 1       
    if mpu.is_pipeline_last_stage():
        input_tensors_buffer[forward_index] = p2p_communication.recv_forward(timers=timers)

    precompution = [5,4,2,0]
    cooldown = [3,2,1,0]
    forward_remain = [0,2,5,8]
    for i in range(precompution[stage_id]):
        # print("stage_id: {},precompute,i={}".format(stage_id,i))
        input_tensor = input_tensors_buffer[backward_index]
        output_tensor = forward_step(forward_step_func, data_iterator, model,
                                    input_tensor, losses_reduced)
        if i < precompution[stage_id + 1] - 1:
            output_tensor_grad = p2p_communication.recv_backward(timers=timers)
        else:
            output_tensor_grad = \
                p2p_communication.send_forward_recv_backward(output_tensors_buffer[send_forward_index],
                                                            timers=timers)
            send_forward_index += 1
        input_tensor_grad = \
            backward_step(optimizer, input_tensor, output_tensor,
                        output_tensor_grad)
        backward_index += 1
        if i < precompution[stage_id] - 1:
            p2p_communication.send_backward(input_tensor_grad)
        else:
            input_tensors_buffer[forward_index] = p2p_communication.send_backward_recv_forward(input_tensor_grad)

    for i in range(forward_remain[stage_id]):
        # print("stage_id: {},forward_remain,i={}".format(stage_id,i))
        input_tensor = input_tensors_buffer[forward_index]
        if i >= forward_remain[stage_id] - (stages - stage_id -1):
            output_tensors_buffer[forward_index] = forward_step(forward_step_func, data_iterator, model,
                                    input_tensor, losses_reduced, recompute=args.sprecompute)
        else:
            output_tensors_buffer[forward_index] = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, losses_reduced)
        forward_index += 1

        output_tensor_grad = p2p_communication.send_forward_recv_backward(output_tensors_buffer[send_forward_index])
        # p2p_communication.send_forward(output_tensors_buffer[send_forward_index])
        # output_tensor_grad = p2p_communication.recv_backward()
        
        
        send_forward_index += 1

        input_tensor_grad = \
            backward_step(optimizer, input_tensors_buffer[backward_index], output_tensors_buffer[backward_index],
                        output_tensor_grad)
        backward_index += 1
        # print("stage_id: {},forward_remain after backward,i={}".format(stage_id,i))
        if i == (forward_remain[stage_id] - 1):
            p2p_communication.send_backward(input_tensor_grad)
            # print("stage_id: {},communication.send_backward_recv_forward after backword,i={}".format(stage_id,i))
        else:
            # p2p_communication.send_backward(input_tensor_grad)
            # print("stage_id: {},send_backward after backward,i={}".format(stage_id,i))

            input_tensors_buffer[forward_index] = p2p_communication.send_backward_recv_forward(input_tensor_grad)

            # print("stage_id: {},recv_forward after backward,i={}".format(stage_id,i))
            

    for i in range(cooldown[stage_id]):
        # print("stage_id: {},cooldown,i={}".format(stage_id,i))
        input_tensor = input_tensors_buffer[backward_index]
        output_tensor = forward_step(forward_step_func, data_iterator, model,
                                    input_tensor, losses_reduced)
        output_tensor_grad = p2p_communication.recv_backward(timers=timers)
        input_tensor_grad = \
            backward_step(optimizer, input_tensor, output_tensor,
                        output_tensor_grad)
        backward_index += 1
        p2p_communication.send_backward(input_tensor_grad)      
    return losses_reduced


def forward_backward_pipelining_with_PreRecompute_batch8_WFBP(forward_step_func, data_iterator,
                                                     model, optimizer, timers,
                                                     forward_only):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    args = get_args()
    timers = get_timers()

    assert len(model) == 1
    model = model[0]

    context_handler = dummy_handler
    if isinstance(model, torchDDP):
        context_handler = model.no_sync

    # Compute number of warmup microbatches.
    num_microbatches = get_num_microbatches()
    num_warmup_microbatches = \
        (mpu.get_pipeline_model_parallel_world_size() -
         mpu.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    losses_reduced = []
   
    forward_index = 0
    backward_index = 0
    send_forward_index = 0
    input_tensors_buffer = [None]*10
    output_tensors_buffer = [None]*10
    s = [8,6,3,0]
    send_forward_guide = [6,3,1]
    stage_id = mpu.get_pipeline_model_parallel_rank()
    stages = mpu.get_pipeline_model_parallel_world_size()
    
    with context_handler():
        for i in range(s[stage_id]):
            # print("stage_id: {},warmup,i={}".format(stage_id,i))
            input_tensor = p2p_communication.recv_forward(timers=timers)
            input_tensors_buffer[forward_index] = input_tensor
            if 3*(stages-stage_id)-2 > 8 or i < 2*(stages-stage_id-1):
                output_tensors_buffer[forward_index] = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, losses_reduced, recompute=args.sprecompute)
            else:
                output_tensors_buffer[forward_index] = forward_step(forward_step_func, data_iterator, model,
                                            input_tensor, losses_reduced)
            forward_index += 1
            if i < send_forward_guide[stage_id]:
                p2p_communication.send_forward(output_tensors_buffer[send_forward_index], timers=timers)
                send_forward_index += 1       
        if mpu.is_pipeline_last_stage():
            input_tensors_buffer[forward_index] = p2p_communication.recv_forward(timers=timers)

        precompution = [5,4,2,0]
        cooldown = [3,2,1,0]
        forward_remain = [0,2,5,8]
        for i in range(precompution[stage_id]):
            # print("stage_id: {},precompute,i={}".format(stage_id,i))
            input_tensor = input_tensors_buffer[backward_index]
            output_tensor = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, losses_reduced)
            if i < precompution[stage_id + 1] - 1:
                output_tensor_grad = p2p_communication.recv_backward(timers=timers)
            else:
                output_tensor_grad = \
                    p2p_communication.send_forward_recv_backward(output_tensors_buffer[send_forward_index],
                                                                timers=timers)
                send_forward_index += 1
            input_tensor_grad = \
                backward_step(optimizer, input_tensor, output_tensor,
                            output_tensor_grad)
            backward_index += 1
            if i < precompution[stage_id] - 1:
                p2p_communication.send_backward(input_tensor_grad)
            else:
                input_tensors_buffer[forward_index] = p2p_communication.send_backward_recv_forward(input_tensor_grad)

        for i in range(forward_remain[stage_id]):
            # print("stage_id: {},forward_remain,i={}".format(stage_id,i))
            input_tensor = input_tensors_buffer[forward_index]
            if i >= forward_remain[stage_id] - (stages - stage_id -1):
                output_tensors_buffer[forward_index] = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, losses_reduced, recompute=args.sprecompute)
            else:
                output_tensors_buffer[forward_index] = forward_step(forward_step_func, data_iterator, model,
                                            input_tensor, losses_reduced)
            forward_index += 1

            output_tensor_grad = p2p_communication.send_forward_recv_backward(output_tensors_buffer[send_forward_index])
            # p2p_communication.send_forward(output_tensors_buffer[send_forward_index])
            # output_tensor_grad = p2p_communication.recv_backward()
            
            
            send_forward_index += 1

            input_tensor_grad = \
                backward_step(optimizer, input_tensors_buffer[backward_index], output_tensors_buffer[backward_index],
                            output_tensor_grad)
            backward_index += 1
            # print("stage_id: {},forward_remain after backward,i={}".format(stage_id,i))
            if i == (forward_remain[stage_id] - 1):
                p2p_communication.send_backward(input_tensor_grad)
                # print("stage_id: {},communication.send_backward_recv_forward after backword,i={}".format(stage_id,i))
            else:
                # p2p_communication.send_backward(input_tensor_grad)
                # print("stage_id: {},send_backward after backward,i={}".format(stage_id,i))

                input_tensors_buffer[forward_index] = p2p_communication.send_backward_recv_forward(input_tensor_grad)

                # print("stage_id: {},recv_forward after backward,i={}".format(stage_id,i))
                

        for i in range(cooldown[stage_id]-1):
            # print("stage_id: {},cooldown,i={}".format(stage_id,i))
            input_tensor = input_tensors_buffer[backward_index]
            output_tensor = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, losses_reduced)
            output_tensor_grad = p2p_communication.recv_backward(timers=timers)
            input_tensor_grad = \
                backward_step(optimizer, input_tensor, output_tensor,
                            output_tensor_grad)
            backward_index += 1
            p2p_communication.send_backward(input_tensor_grad)
    if not mpu.is_pipeline_last_stage():
        input_tensor = input_tensors_buffer[backward_index]
        output_tensor = forward_step(forward_step_func, data_iterator, model,
                                    input_tensor, losses_reduced)
        output_tensor_grad = p2p_communication.recv_backward(timers=timers)
        input_tensor_grad = \
            backward_step(optimizer, input_tensor, output_tensor,
                        output_tensor_grad)
        backward_index += 1
        p2p_communication.send_backward(input_tensor_grad)
    return losses_reduced


def forward_backward_pipelining_with_PreRecompute_batch8_WFBP_stage8(forward_step_func, data_iterator,
                                                     model, optimizer, timers,
                                                     forward_only):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    args = get_args()
    timers = get_timers()

    assert len(model) == 1
    model = model[0]

    context_handler = dummy_handler
    if isinstance(model, torchDDP):
        context_handler = model.no_sync

    # Compute number of warmup microbatches.
    num_microbatches = get_num_microbatches()
    num_warmup_microbatches = \
        (mpu.get_pipeline_model_parallel_world_size() -
         mpu.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    losses_reduced = []
   
    forward_index = 0
    backward_index = 0
    send_forward_index = 0
    input_tensors_buffer = [None]*18
    output_tensors_buffer = [None]*18
    s = [16,16,15,12,9,6,3,0]
    send_forward_guide = [16,15,12,9,6,3,1]
    stage_id = mpu.get_pipeline_model_parallel_rank()
    stages = mpu.get_pipeline_model_parallel_world_size()
    
    with context_handler():
        for i in range(s[stage_id]):
            # print("stage:{}, warmup{}".format(stage_id, s[stage_id]))
            # print("stage_id: {},warmup,i={}".format(stage_id,i))
            input_tensor = p2p_communication.recv_forward(timers=timers)
            input_tensors_buffer[forward_index] = input_tensor
            if 3*(stages-stage_id -1) > num_microbatches or i < 2*(stages-stage_id-1):
                output_tensors_buffer[forward_index] = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, losses_reduced, recompute=args.sprecompute)
            else:
                output_tensors_buffer[forward_index] = forward_step(forward_step_func, data_iterator, model,
                                            input_tensor, losses_reduced)
            forward_index += 1
            if i < send_forward_guide[stage_id]:
                p2p_communication.send_forward(output_tensors_buffer[send_forward_index], timers=timers)
                send_forward_index += 1       
        if mpu.is_pipeline_last_stage():
            input_tensors_buffer[forward_index] = p2p_communication.recv_forward(timers=timers)

        precompution = [9,10,10,8,6,4,2,0]
        cooldown = [7,6,5,4,3,2,1,0]
        forward_remain = [0,0,1,4,7,10,13,16]
        for i in range(precompution[stage_id]):
            # print("stage_id: {},precompute,i={}".format(stage_id,i))
            input_tensor = input_tensors_buffer[backward_index]
            output_tensor = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, losses_reduced)
            if i < precompution[stage_id + 1] - 1:
                output_tensor_grad = p2p_communication.recv_backward(timers=timers)
            else:
                output_tensor_grad = \
                    p2p_communication.send_forward_recv_backward(output_tensors_buffer[send_forward_index],
                                                                timers=timers)
                send_forward_index += 1
            input_tensor_grad = \
                backward_step(optimizer, input_tensor, output_tensor,
                            output_tensor_grad)
            backward_index += 1
            if i < (precompution[stage_id] - 1) or forward_remain[stage_id]==0:
                p2p_communication.send_backward(input_tensor_grad)
            else:
                input_tensors_buffer[forward_index] = p2p_communication.send_backward_recv_forward(input_tensor_grad)

        for i in range(forward_remain[stage_id]):
            # print("stage_id: {},forward_remain,i={}".format(stage_id,i))
            input_tensor = input_tensors_buffer[forward_index]
            if i >= forward_remain[stage_id] - (stages - stage_id -1):
                output_tensors_buffer[forward_index] = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, losses_reduced, recompute=args.sprecompute)
            else:
                output_tensors_buffer[forward_index] = forward_step(forward_step_func, data_iterator, model,
                                            input_tensor, losses_reduced)
            forward_index += 1

            output_tensor_grad = p2p_communication.send_forward_recv_backward(output_tensors_buffer[send_forward_index])
            # p2p_communication.send_forward(output_tensors_buffer[send_forward_index])
            # output_tensor_grad = p2p_communication.recv_backward()
            
            
            send_forward_index += 1

            input_tensor_grad = \
                backward_step(optimizer, input_tensors_buffer[backward_index], output_tensors_buffer[backward_index],
                            output_tensor_grad)
            backward_index += 1
            # print("stage_id: {},forward_remain after backward,i={}".format(stage_id,i))
            if i == (forward_remain[stage_id] - 1):
                # print("stage_id: {},****before*** communication.send_backward after backword,i={}".format(stage_id,i))
                p2p_communication.send_backward(input_tensor_grad)
                # print("stage_id: {},communication.send_backward_recv_forward after backword,i={}".format(stage_id,i))
            else:
                # p2p_communication.send_backward(input_tensor_grad)
                # print("stage_id: {},send_backward after backward,i={}".format(stage_id,i))

                input_tensors_buffer[forward_index] = p2p_communication.send_backward_recv_forward(input_tensor_grad)

                # print("stage_id: {},recv_forward after backward,i={}".format(stage_id,i))
                

        for i in range(cooldown[stage_id]-1):
            # print("stage_id: {},cooldown,i={}".format(stage_id,i))
            input_tensor = input_tensors_buffer[backward_index]
            output_tensor = forward_step(forward_step_func, data_iterator, model,
                                        input_tensor, losses_reduced)
            output_tensor_grad = p2p_communication.recv_backward(timers=timers)
            input_tensor_grad = \
                backward_step(optimizer, input_tensor, output_tensor,
                            output_tensor_grad)
            backward_index += 1
            p2p_communication.send_backward(input_tensor_grad)
    if not mpu.is_pipeline_last_stage():
        input_tensor = input_tensors_buffer[backward_index]
        output_tensor = forward_step(forward_step_func, data_iterator, model,
                                    input_tensor, losses_reduced)
        output_tensor_grad = p2p_communication.recv_backward(timers=timers)
        input_tensor_grad = \
            backward_step(optimizer, input_tensor, output_tensor,
                        output_tensor_grad)
        backward_index += 1
        p2p_communication.send_backward(input_tensor_grad)
    return losses_reduced