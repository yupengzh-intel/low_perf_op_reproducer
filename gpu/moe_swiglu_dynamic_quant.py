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


class MoeSwigluDynamicQuantOp:

    def __init__(self):
        # pre-defined attrs
        self.world_size = 8
        self.rank = 0
        self.ep_size = 8
        self.dp_size = 1
        self.sp_size = 1

        self.num_shared_experts = 0
        self.num_experts = 64
        self.topk = 4
        self.num_tokens = 8192
        self.hidden_size = 1024

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
        no remainder on **num_experts**
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
        """
        self.tokens_per_ep = self.num_tokens // self.ep_size
        self.tokens_ep_start = self.ep_rank * self.tokens_per_ep
        self.tokens_ep_end = self.tokens_ep_start + self.tokens_per_ep

        # [tokens_per_ep, topk]
        self.allocated_tokens = self.tokens_per_ep * self.topk
        self.allocated_tokens_per_expert = self.allocated_tokens // self.experts_per_rank
        self.allocated_tokens_per_expert_remainder = self.allocated_tokens % self.experts_per_rank

        self.token_list = []
        self.token_start_list = []
        temp_token_start = 0
        for i in range(self.shared_experts_per_rank):
            self.token_start_list.append(temp_token_start)
            self.token_list.append(self.shared_tokens_per_sp)
            temp_token_start += self.token_list[-1]
        for i in range(self.experts_per_rank):
            self.token_start_list.append(temp_token_start)
            if i < self.allocated_tokens_per_expert_remainder:
                self.token_list.append(self.allocated_tokens_per_expert + 1)
            else:
                self.token_list.append(self.allocated_tokens_per_expert)
            temp_token_start += self.token_list[-1]


        self.total_experts_num = self.shared_experts_per_rank + self.experts_per_rank

        self.total_shared_tokens = self.shared_tokens_per_sp * self.shared_experts_per_rank

        self.real_allocated_tokens = self.allocated_tokens
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
            scatter_tokens = torch.randn(
                size=[self.real_scatter_tokens, self.hidden_size * 2], 
                dtype=torch.bfloat16, 
                device="cuda"
            )

            smooth_scale = torch.ones(
                size=[self.total_experts_num, self.hidden_size], 
                dtype=torch.float32, 
                device="cuda"
            )

            experts_token_count = torch.tensor(
                self.token_list, 
                dtype=torch.int32, device="cuda"
            )

            experts_token_start = torch.tensor(
                self.token_start_list, 
                dtype=torch.int32, device="cuda"
            )


            quant_tokens = torch.randn(
                size=[self.real_scatter_tokens, self.hidden_size], 
                dtype=torch.bfloat16, 
                device="cuda"
            )

            per_token_scale = torch.randn(
                size=[self.real_scatter_tokens], 
                dtype=torch.float32, 
                device="cuda"
            )

            tensors = {
                'scatter_tokens': scatter_tokens,
                'smooth_scale': smooth_scale,
                'experts_token_count': experts_token_count,
                'experts_token_start': experts_token_start,
                'quant_tokens': quant_tokens,
                'per_token_scale': per_token_scale,
            }
            all_tensor_list.append(tensors)
            
        return all_tensor_list


    def tensor_size(self):
        size = 0
        
        # scatter_tokens: [real_scatter_tokens, hidden_size * 2], dtype=torch.bfloat16
        size += self.real_scatter_tokens * self.hidden_size * 2 * torch.bfloat16.itemsize
        
        # smooth_scale: [total_experts_num, hidden_size], dtype=torch.float32
        size += self.total_experts_num * self.hidden_size * torch.float32.itemsize
        
        # experts_token_count: [total_experts_num], dtype=torch.int32
        size += self.total_experts_num * torch.int32.itemsize
        
        # experts_token_start: [total_experts_num], dtype=torch.int32
        size += self.total_experts_num * torch.int32.itemsize
        
        # quant_tokens: [real_scatter_tokens, hidden_size], dtype=torch.bfloat16
        size += self.real_scatter_tokens * self.hidden_size * torch.bfloat16.itemsize
        
        # per_token_scale: [real_scatter_tokens], dtype=torch.float32
        size += self.real_scatter_tokens * torch.float32.itemsize
        
        return size
    
    def op(self, tensors):
        # get pre-allocated input tensors
        scatter_tokens = tensors["scatter_tokens"]
        smooth_scale = tensors["smooth_scale"]
        experts_token_count = tensors["experts_token_count"]
        experts_token_start = tensors["experts_token_start"]

        # get per-allocated output tensors
        quant_tokens = tensors["quant_tokens"]
        per_token_scale = tensors["per_token_scale"]


        # swiglu, x1 used as gating, x2 used as up
        x1, x2 = torch.chunk(scatter_tokens, 2, dim=-1)
        swiglu_tokens = torch.mul(torch.nn.functional.silu(x1), x2)

        # per expert dynamic quant
        for i in range(self.total_experts_num):
            cur_token_start = experts_token_start[i]
            cur_token_end = cur_token_start + experts_token_count[i]

            cur_quant_tokens, cur_per_token_scale = smooth_per_token_dynamic_quant(
                swiglu_tokens[cur_token_start:cur_token_end], 
                smooth_scale[i]
            )
            quant_tokens[cur_token_start:cur_token_end] = cur_quant_tokens
            per_token_scale[cur_token_start:cur_token_end] = cur_per_token_scale

        return quant_tokens, per_token_scale
    
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
                    torch.profiler.ProfilerActivity.CUDA,
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
    op = MoeSwigluDynamicQuantOp()
    op.run()
