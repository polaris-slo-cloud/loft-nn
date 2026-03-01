# The following code is based on the official PyTorch code related to the 1F1B schedule.
# Source: https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/pipelining/schedules.py
# Original Copyright (c) Meta Platforms, Inc. and affiliates
# See PYTORCH_LICENSE in the project root for the full license text of the official PyTorch code.
# The code has been extended to support DDP stages, enabling hybrid pipeline parallelism.

import logging
import time
import types

from typing import Any, Callable, Optional, Union

import torch

import torch.distributed as dist

from torch import Tensor

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed import ProcessGroup
from torch.distributed.pipelining.schedules import PipelineScheduleSingle
from torch.distributed.pipelining.stage import _PipelineStageBase
from torch.distributed.pipelining.microbatch import TensorChunkSpec

from loftnn.logging import log_execution_time
from loftnn.types import Device
from loftnn.worker_utils import free_memory

logger = logging.getLogger(__name__)


def _batch_p2p(
    p2p_ops: list[dist.P2POp], desc: Optional[str] = None
) -> list[dist.Work]:
    """
    Simple wrapper over batch_isend_irecv from torch.distributed, which just adds a descriptive logger on top.
    """
    if len(p2p_ops) == 0:
        return []

    desc_str = f"{desc}, " if desc else ""
    logger.debug(f"rank {dist.get_rank()}: batch_p2p {desc_str}p2p_ops={p2p_ops}")
    logger.debug(
        f"rank {dist.get_rank()}: batch_p2p {desc_str}tensors={[p.tensor for p in p2p_ops]}"
    )
    return dist.batch_isend_irecv(p2p_ops)


def _wait_batch_p2p(work: list[dist.Work]):
    """
    Waits for a list of dist.Work (typically from _batch_p2p / _sorted_batch_p2p).
    """
    for w in work:
        w.wait()


def _new_forward_maybe_with_nosync(self, *args, **kwargs):
    device = str(next(self.submod.parameters()).device)
    if "cuda" in device:
        args = [a.to(Device.cuda) for a in args]
        kwargs = {k: v.to(Device.cuda) for k, v in kwargs.items()}

    if isinstance(self.submod, DDP):
        with self.submod.no_sync():
            out_val = self.submod(*args, **kwargs)
    else:
        out_val = self.submod(*args, **kwargs)

    if "cuda" in device:
        if isinstance(out_val, Tensor):
            out_val = out_val.to(Device.cpu)
        else:
            out_val = tuple([o.to(Device.cpu) for o in out_val])

    return out_val


class ScheduleHybridPipelineParallel1F1B(PipelineScheduleSingle):
    """
    The 1F1B schedule with support for DDP stages, enabling hybrid pipeline parallelism.
    Will perform one forward and one backward on the microbatches in steady state.
    """

    def __init__(
        self,
        stage: _PipelineStageBase,
        n_microbatches: int,
        microbatch_split_at: list[int],
        ddp_world_ranks: list[int],
        ddp_group: Optional[ProcessGroup],
        prev_stage_is_ddp: bool,
        next_stage_is_ddp: bool,
        prev_non_ddp_stage_world_rank: int,
        next_non_ddp_stage_world_rank: int,
        loss_fn: Optional[Callable] = None,
        args_chunk_spec: Optional[tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[dict[str, Any], tuple[Any]]] = None,
        scale_grads: bool = True,
    ):
        super().__init__(
            stage=stage,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
        )

        self.microbatch_split_at = microbatch_split_at
        self.ddp_world_ranks = ddp_world_ranks
        self.ddp_group = ddp_group
        self.prev_stage_is_ddp = prev_stage_is_ddp
        self.next_stage_is_ddp = next_stage_is_ddp
        self.prev_non_ddp_stage_world_rank = prev_non_ddp_stage_world_rank
        self.next_non_ddp_stage_world_rank = next_non_ddp_stage_world_rank

        self._stage.forward_maybe_with_nosync = types.MethodType(
            _new_forward_maybe_with_nosync, self._stage
        )

    def _split_ops_for_ddp_stage(self, ops):
        ddp_ops = []
        for op in ops:
            tensor_chunks = op.tensor.chunk(len(self.ddp_world_ranks))
            peer_infos = [(torch.distributed.group.WORLD, op.peer)]
            peer_infos.extend(
                [
                    (torch.distributed.group.WORLD, rank)
                    for rank in self.ddp_world_ranks
                    if rank != op.peer
                ]
            )
            new_ops = [
                dist.P2POp(op=op.op, tensor=op.tensor, peer=op.peer, group=op.group)
                for _ in range(len(self.ddp_world_ranks))
            ]
            for chunked_tensor, peer_info, op in zip(
                tensor_chunks, peer_infos, new_ops
            ):
                op.tensor = chunked_tensor
                group, peer = peer_info
                op.group = group
                op.peer = peer
                op.group_peer = peer
            ddp_ops.extend(new_ops)
        return ddp_ops

    def _update_ops_for_ddp_stage(self, ops, peer):
        for op in ops:
            op.group = torch.distributed.group.WORLD
            op.peer = peer
            op.group_peer = op.peer

    def _step_microbatches(
        self,
        arg_mbs: Optional[list] = None,
        kwarg_mbs: Optional[list] = None,
        target_mbs: Optional[list] = None,
        losses: Optional[list] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the 1F1B schedule.

        Args:
            microbatches: list of microbatch args.
        """

        logger.debug(
            f"free memory for cpu={free_memory(Device.cpu)}{(', gpu=' + str(free_memory(Device.cuda))) if torch.cuda.is_available() else ''}"
        )

        logger.debug("starting iteration")
        start = time.time()

        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        training = self._has_backward

        if isinstance(self._stage.submod, DDP):
            arg_mbs = [
                (
                    (
                        torch.tensor_split(arg_mb[0], self.microbatch_split_at)[
                            self.ddp_group.rank()
                        ],
                    )
                    if arg_mb
                    else arg_mb
                )
                for arg_mb in arg_mbs
            ]
            kwarg_mbs = [
                (
                    (
                        torch.tensor_split(kwarg_mb[0], self.microbatch_split_at)[
                            self.ddp_group.rank()
                        ],
                    )
                    if kwarg_mb
                    else kwarg_mb
                )
                for kwarg_mb in kwarg_mbs
            ]
            if target_mbs:
                target_mbs = [
                    torch.tensor_split(target_mb, self.microbatch_split_at)[
                        self.ddp_group.rank()
                    ]
                    for target_mb in target_mbs
                ]

        if not self._stage_initialized:
            self._initialize_stage(arg_mbs[0], kwarg_mbs[0])

        # Last stage has 1 warmup, second-to-last 2 warmups, ...
        # first stage `num_stages` warmups
        warmup_chunks = min(
            self._n_microbatches,
            self._num_stages - self._stage.stage_index,
        )

        # Chunk counters
        fwd_mb_index = 0
        bwd_mb_index = 0

        logger.debug("warmup phase")

        send_work: list[dist.Work] = []
        fwd_sends = []
        for _ in range(warmup_chunks):
            # Receive activations
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)
            if isinstance(self._stage.submod, DDP):
                self._update_ops_for_ddp_stage(
                    fwd_recvs, self.prev_non_ddp_stage_world_rank
                )
            if self.prev_stage_is_ddp:
                fwd_recvs = self._split_ops_for_ddp_stage(fwd_recvs)

            _wait_batch_p2p(_batch_p2p(fwd_recvs, desc="fwd_recv"))

            with log_execution_time(
                "forward_one_chunk_warmup", logger.debug, self._stage.submod
            ):
                # Compute
                output = self._stage.forward_one_chunk(
                    fwd_mb_index, arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index]
                )  # type: ignore[index]

            # Clear previous chunk's forward sends (hopefully they have well
            # finished, otherwise, we are heavily communication bound, in which
            # case it doesn't create a lot of benefit to compute next chunk
            # eagerly either)
            _wait_batch_p2p(send_work)

            # Send activations
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            if isinstance(self._stage.submod, DDP):
                self._update_ops_for_ddp_stage(
                    fwd_sends, self.next_non_ddp_stage_world_rank
                )
            if self.next_stage_is_ddp:
                fwd_sends = self._split_ops_for_ddp_stage(fwd_sends)

            if fwd_mb_index != warmup_chunks - 1:
                # Safe to fire
                send_work = _batch_p2p(fwd_sends, desc="fwd_send")
            # otherwise:
            #   The last forward send is left for fuse with first 1B in 1B1F below

            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)

            fwd_mb_index += 1

        # Now we should have send ops left over, to be fused with first 1B of 1B1F phase below.

        logger.debug("1B1F phase")

        while True:  # Don't worry, we have a break inside
            # We actually do 1B first as the `1B1F` name indicates, so prepare its recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index) if training else []
            if self.next_stage_is_ddp:
                bwd_recvs = self._split_ops_for_ddp_stage(bwd_recvs)
                _wait_batch_p2p(_batch_p2p(fwd_sends, desc="fwd_send_bwd_recv"))
                _wait_batch_p2p(_batch_p2p(bwd_recvs, desc="fwd_send_bwd_recv"))
            elif isinstance(self._stage.submod, DDP):
                self._update_ops_for_ddp_stage(
                    bwd_recvs, self.next_non_ddp_stage_world_rank
                )
                _wait_batch_p2p(_batch_p2p(fwd_sends, desc="fwd_send_bwd_recv"))
                _wait_batch_p2p(_batch_p2p(bwd_recvs, desc="fwd_send_bwd_recv"))
            else:
                # Now, we need to fire the fwd_sends and bwd_recvs together
                _wait_batch_p2p(
                    _batch_p2p(fwd_sends + bwd_recvs, desc="fwd_send_bwd_recv")
                )

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            if training:
                with log_execution_time(
                    "backward_one_chunk_1b1f", logger.debug, self._stage.submod
                ):
                    last_backward = bwd_mb_index == self._n_microbatches - 1
                    if last_backward:
                        logger.debug("last backward")

                    self._stage.backward_one_chunk(
                        bwd_mb_index,
                        loss=loss,
                        last_backward=last_backward,
                    )

            # Get the bwd send ops, but don't fire, to be fused with the 1F below
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index) if training else []
            if isinstance(self._stage.submod, DDP):
                self._update_ops_for_ddp_stage(
                    bwd_sends, self.prev_non_ddp_stage_world_rank
                )
            if self.prev_stage_is_ddp:
                bwd_sends = self._split_ops_for_ddp_stage(bwd_sends)

            bwd_mb_index += 1

            if fwd_mb_index == self._n_microbatches:
                # We are done with 1B1F, so break with some left-over bwd_sends
                break

            # We prepare 1F of the `1B1F`
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)
            if self.prev_stage_is_ddp:
                fwd_recvs = self._split_ops_for_ddp_stage(fwd_recvs)
                _wait_batch_p2p(_batch_p2p(fwd_recvs, desc="bwd_send_fwd_recv"))
                _wait_batch_p2p(_batch_p2p(bwd_sends, desc="bwd_send_fwd_recv"))
            elif isinstance(self._stage.submod, DDP):
                self._update_ops_for_ddp_stage(
                    fwd_recvs, self.prev_non_ddp_stage_world_rank
                )
                _wait_batch_p2p(_batch_p2p(fwd_recvs, desc="bwd_send_fwd_recv"))
                _wait_batch_p2p(_batch_p2p(bwd_sends, desc="bwd_send_fwd_recv"))
            else:
                # Fuse it with bwd_sends above
                _wait_batch_p2p(
                    _batch_p2p(bwd_sends + fwd_recvs, desc="bwd_send_fwd_recv")
                )

            with log_execution_time(
                "forward_one_chunk_1b1f", logger.debug, self._stage.submod
            ):
                # Now do the fwd
                output = self._stage.forward_one_chunk(
                    fwd_mb_index, arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index]
                )  # type: ignore[index]

            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)

            # Get the fwd send ops, but don't fire, leave it for the next iter (wrap-around)
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            if isinstance(self._stage.submod, DDP):
                self._update_ops_for_ddp_stage(
                    fwd_sends, self.next_non_ddp_stage_world_rank
                )
            if self.next_stage_is_ddp:
                fwd_sends = self._split_ops_for_ddp_stage(fwd_sends)

            fwd_mb_index += 1

        # Remember we still have some bwd_sends left over after the break? Now it is time to fire it
        send_work = _batch_p2p(bwd_sends, desc="bwd_send")

        logger.debug("cooldown phase")

        while bwd_mb_index < self._n_microbatches:
            # prepare bwd recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index) if training else []
            if isinstance(self._stage.submod, DDP):
                self._update_ops_for_ddp_stage(
                    bwd_recvs, self.next_non_ddp_stage_world_rank
                )
            if self.next_stage_is_ddp:
                bwd_recvs = self._split_ops_for_ddp_stage(bwd_recvs)

            _wait_batch_p2p(_batch_p2p(bwd_recvs, desc="bwd_recv"))

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            if training:
                with log_execution_time(
                    "backward_one_chunk_cooldown", logger.debug, self._stage.submod
                ):
                    last_backward = bwd_mb_index == self._n_microbatches - 1
                    if last_backward:
                        logger.debug("last backward")

                    self._stage.backward_one_chunk(
                        bwd_mb_index,
                        loss=loss,
                        last_backward=last_backward,
                    )

            # Clear previous chunk's backward sends (hopefully they have well finished)
            _wait_batch_p2p(send_work)

            # Get the bwd send ops, fire it
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index) if training else []
            if isinstance(self._stage.submod, DDP):
                self._update_ops_for_ddp_stage(
                    bwd_sends, self.prev_non_ddp_stage_world_rank
                )
            if self.prev_stage_is_ddp:
                bwd_sends = self._split_ops_for_ddp_stage(bwd_sends)

            send_work = _batch_p2p(bwd_sends, desc="bwd_send")
            bwd_mb_index += 1

        self._stage.scale_grads(
            grad_scale_factor=self._n_microbatches if self.scale_grads else 1
        )

        # Wait for the last backward send to finish
        _wait_batch_p2p(send_work)

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)

        end = time.time()
        logger.debug(f"finished iteration, took {(end-start):.4f} seconds")
