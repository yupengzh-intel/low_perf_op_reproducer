import torch
import random
import math
from itertools import combinations


def smooth_per_token_dynamic_quant(
    hidden_states, 
    smooth_scale, 
    dst_torch_dtype=torch.int8
):
    smoothed_input = torch.mul(hidden_states, smooth_scale).type(torch.float32)
    per_token_scale = torch.div(torch.max(smoothed_input.abs(), -1, keepdim=False)[0], 127.0)
    quant_tokens = torch.div(smoothed_input, per_token_scale.unsqueeze(-1)).round().type(dst_torch_dtype)
    return quant_tokens, per_token_scale


class MoeScatterDynamicQuantOp:

    def __init__(self):
        self.world_size = 8
        self.rank = 0
        self.ep_size = 8
        self.dp_size = 1
        self.sp_size = 1

        self.num_shared_experts = 0
        self.num_experts = 64
        self.topk = 4
        self.num_tokens = 8192
        self.hidden_size = 4096
        

        """
        select shared experts based on dp_size/dp_rank
        """
        self.dp_rank = self.rank // self.sp_size
        self.shared_experts_per_rank = self.num_shared_experts // self.dp_size
        
        """
        select tokens based on sp_size/sp_rank
        """
        self.sp_rank = self.rank % self.sp_size
        self.shared_tokens_per_sp = self.num_tokens // self.sp_size
        self.shared_token_sp_start = self.sp_rank * self.shared_tokens_per_sp
        self.shared_token_sp_end = self.shared_token_sp_start + self.shared_tokens_per_sp


        """
        select experts based on ep_rank
        """
        self.experts_per_rank = self.num_experts // self.ep_size
        self.ep_rank = self.rank
        self.expert_idx_start = self.ep_rank * self.experts_per_rank
        self.expert_idx_end = self.expert_idx_start + self.experts_per_rank
        self.other_experts_set = \
            set(range(self.num_experts)) - \
            set(range(self.expert_idx_start, self.expert_idx_end))

        """
        for convinience, we also split num_tokens to ep_size parts to generate selected_experts
        and selected tokens are also distributed to corresponding experts
        if rank == 0, experts_per_rank == 4, and topk == 5, and num_tokens == 32, so num_tokens_per_ep == 8
        token 0: 0, 1, 2, 3, 0
        token 1: 1, 2, 3, 0, 1
        token 2: 2, 3, 0, 1, 2
        token 3: 3, 0, 1, 2, 3
        ...

        other tokens will select other tokens randomly
        """
        self.tokens_per_ep = self.num_tokens // self.ep_size
        self.tokens_ep_start = self.ep_rank * self.tokens_per_ep
        self.tokens_ep_end = self.tokens_ep_start + self.tokens_per_ep
        
        # [tokens_per_ep, topk]
        self.actual_output_tokens = self.tokens_per_ep * self.topk
        self.experts_repeat_time = 1
        if self.actual_output_tokens > self.experts_per_rank:
            self.experts_repeat_time = (self.actual_output_tokens + self.experts_per_rank - 1) // self.experts_per_rank
        self.refer_expert_seq = torch.arange(
            start=self.expert_idx_start, 
            end=self.expert_idx_end, 
            dtype=torch.int32
        ).repeat(self.experts_repeat_time)[:self.actual_output_tokens].view(
            self.tokens_per_ep, self.topk)

        # all tokens topk
        dummy_experts = list(next(combinations(self.other_experts_set, self.topk)))
        self.refer_selected_experts = torch.tensor(dummy_experts, dtype=torch.int32).unsqueeze(0).repeat(self.num_tokens, 1)
        self.refer_selected_experts[self.tokens_ep_start:self.tokens_ep_end] = self.refer_expert_seq


        self.total_experts_num = self.shared_experts_per_rank + self.experts_per_rank

        # reserve tokens memory for shared_tokens_per_sp/allocated_tokens
        self.total_shared_tokens = self.shared_tokens_per_sp * self.shared_experts_per_rank

        # for extreme case
        self.max_allocated_tokens = self.num_tokens * self.topk
        self.max_scatter_tokens = self.total_shared_tokens + self.max_allocated_tokens

        # for designed real case
        self.real_allocated_tokens = self.actual_output_tokens
        self.real_scatter_tokens = self.total_shared_tokens + self.real_allocated_tokens
        

        # SRAM 96MB
        cache_size = 96 * (1024 ** 2)
        tensor_size = self.tensor_size()
        max_data_cnt = math.ceil(cache_size / tensor_size)
        self.tensor_list = self.create_tensors(max_data_cnt)
        random.shuffle(self.tensor_list)
    
    def create_tensors(self, max_data_cnt):
        all_tensor_list = []
        
        for _ in range(max_data_cnt):
            hidden_states = torch.randn(
                size=[self.num_tokens, self.hidden_size], 
                dtype=torch.bfloat16, 
                device="cuda"
            )

            selected_experts = self.refer_selected_experts.to("cuda")

            moe_weights = torch.ones(
                size=[self.num_tokens, self.topk], 
                dtype=torch.float32, 
                device="cuda"
            )

            smooth_scale = torch.ones(
                size=[self.total_experts_num, self.hidden_size], 
                dtype=torch.float32, 
                device="cuda"
            )

            scatter_tokens = torch.zeros(
                size=[self.max_scatter_tokens, self.hidden_size],
                dtype=torch.int8,
                device="cuda"
            )

            scatter_per_token_scale = torch.ones(
                size=[self.max_scatter_tokens], 
                dtype=torch.float32, 
                device="cuda"
            )

            scatter_tokens_offset = torch.ones(
                size=[self.max_scatter_tokens], 
                dtype=torch.int32, 
                device="cuda"
            ) * -1

            experts_token_count = torch.zeros(
                size=[self.total_experts_num], 
                dtype=torch.int32, 
                device="cuda"
            )

            experts_token_start = torch.zeros(
                size=[self.total_experts_num], 
                dtype=torch.int32, 
                device="cuda"
            )

            tensors = {
                'hidden_states': hidden_states,
                'selected_experts': selected_experts,
                'moe_weights': moe_weights,
                'smooth_scale': smooth_scale,
                'scatter_tokens': scatter_tokens,
                'scatter_per_token_scale': scatter_per_token_scale,
                'scatter_tokens_offset': scatter_tokens_offset,
                'experts_token_count': experts_token_count,
                'experts_token_start': experts_token_start,
            }
            all_tensor_list.append(tensors)
            
        return all_tensor_list


    def tensor_size(self):
        size = 0
        
        size += self.num_tokens * self.hidden_size * torch.bfloat16.itemsize  # hidden_states
        size += self.num_tokens * self.topk * torch.int32.itemsize  # selected_experts
        size += self.num_tokens * self.topk * torch.float32.itemsize  # moe_weights
        size += self.total_experts_num * self.hidden_size * torch.float32.itemsize  # smooth_scale
        
        size += self.max_scatter_tokens * self.hidden_size * torch.int8.itemsize  # scatter_tokens
        size += self.max_scatter_tokens * torch.float32.itemsize  # scatter_per_token_scale
        size += self.max_scatter_tokens * torch.int32.itemsize  # scatter_tokens_offset
        size += self.total_experts_num * torch.int32.itemsize  # experts_token_count
        size += self.total_experts_num * torch.int32.itemsize  # experts_token_start
        
        return size
    
    def op(self, tensors):
        # get pre-allocated input tensors
        hidden_states = tensors["hidden_states"]
        selected_experts = tensors["selected_experts"]
        moe_weights = tensors["moe_weights"]
        smooth_scale = tensors["smooth_scale"]

        # get pre-allocated output tensors
        scatter_tokens = tensors["scatter_tokens"]
        scatter_per_token_scale = tensors["scatter_per_token_scale"]
        scatter_tokens_offset = tensors["scatter_tokens_offset"]
        experts_token_count = tensors["experts_token_count"]
        experts_token_start = tensors["experts_token_start"]
        
        # shared experts
        for idx in range(self.shared_experts_per_rank):
            expert_idx = idx

            # dynamic quant
            input_token_start = self.shared_token_sp_start
            input_token_end = self.shared_token_sp_end

            quant_tokens, tokens_scale = smooth_per_token_dynamic_quant(
                hidden_states[input_token_start:input_token_end], 
                smooth_scale[expert_idx], 
                dst_torch_dtype=torch.int8
            )

            # assign output
            output_token_start = idx * self.shared_tokens_per_sp
            output_token_end = output_token_start + self.shared_tokens_per_sp

            scatter_tokens[output_token_start:output_token_end] = quant_tokens
            scatter_per_token_scale[output_token_start:output_token_end] = tokens_scale
            experts_token_count[expert_idx] = self.num_tokens
            experts_token_start[expert_idx] = output_token_start
            scatter_tokens_offset[output_token_start:output_token_end] = torch.arange(
                start=input_token_start, 
                end=input_token_end, 
                dtype=torch.int32, 
                device="cuda"
            )
                
        # experts
        cur_output_token_offset = self.total_shared_tokens
        for idx in range(self.experts_per_rank):
            expert_idx = self.shared_experts_per_rank + idx

            # get token indices
            token_indices, topk_indices = torch.where(selected_experts == self.expert_idx_start + idx)
            cur_token_count = token_indices.numel()

            experts_token_count[expert_idx] = cur_token_count    
            experts_token_start[expert_idx] = cur_output_token_offset
            
            output_token_start = cur_output_token_offset
            cur_output_token_offset += cur_token_count
            output_token_end = cur_output_token_offset

            if cur_token_count > 0:
                cur_tokens = hidden_states[token_indices]
                cur_tokens_weight = moe_weights[token_indices, topk_indices]

                # dynamic quant
                quant_tokens, tokens_scale = smooth_per_token_dynamic_quant(
                    torch.mul(cur_tokens, cur_tokens_weight.view(cur_token_count, 1)), 
                    smooth_scale[expert_idx], 
                    dst_torch_dtype=torch.int8
                )

                # assign output
                scatter_tokens[output_token_start:output_token_end] = quant_tokens
                scatter_per_token_scale[output_token_start:output_token_end] = tokens_scale
                scatter_tokens_offset[output_token_start:output_token_end] = token_indices.type(scatter_tokens_offset.dtype)
    
    def perf(self, iterations, profiling=False):
        if profiling:
            schedule = torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=1,
            )
            prof=torch.profiler.profile(
                schedule=schedule,
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.HPU,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(".", use_gzip=True),
                record_shapes=True,
                with_modules=False,
                profile_memory=False,
                with_stack=True,
            )

            prof.start()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for i in range(iterations):
            _ = self.op(self.tensor_list[i % len(self.tensor_list)])
        end_event.record()
        end_event.synchronize()

        if profiling:
            prof.stop()

        latency_us = start_event.elapsed_time(end_event) * 1e3 / iterations
        return latency_us
    
    def run(self, profiling=False):
        # warmup
        self.perf(2)
        latency_us = self.perf(32, profiling)
        print(f"{latency_us=:.2f}")
    
if __name__ == "__main__":
    op = MoeScatterDynamicQuantOp()
    op.run()
