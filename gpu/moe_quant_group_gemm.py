import torch
import random
import math


def fake_quant_gemm(
    tokens, per_token_scale, 
    weights, weight_scale, 
    dst_torch_dtype=torch.bfloat16
):
    fake_gemm_output = torch.matmul(
        tokens.type(torch.bfloat16), 
        weights.type(torch.bfloat16)
    )
    dequant_scale = torch.matmul(
        per_token_scale.unsqueeze(-1), 
        weight_scale.unsqueeze(0)
    )
    y = torch.mul(
        fake_gemm_output, 
        dequant_scale
    ).type(dst_torch_dtype)
    return y

class MoeQuantGroupGemmOp:

    def __init__(self):
        self.num_tokens = 8192
        self.hidden_size = 4096
        self.new_hidden_size = 2048
        self.num_experts = 8
        self.topk = 4
        self.world_size = 8
        self.rank = 0
        self.ep_size = 8
        self.dp_size = 1
        self.sp_size = 1
        self.num_shared_experts = 0
        
        self.dp_rank = self.rank // self.sp_size
        self.shared_experts_per_rank = self.num_shared_experts // self.dp_size

        self.sp_rank = self.rank % self.sp_size
        self.shared_tokens_per_sp = self.num_tokens // self.sp_size
        self.shared_token_sp_start = self.sp_rank * self.shared_tokens_per_sp
        self.shared_token_sp_end = self.shared_token_sp_start + self.shared_tokens_per_sp

        self.experts_per_rank = self.num_experts // self.ep_size
        self.ep_rank = self.rank
        self.expert_idx_start = self.ep_rank * self.experts_per_rank
        self.expert_idx_end = self.expert_idx_start + self.experts_per_rank
        
        self.tokens_per_ep = self.num_tokens // self.ep_size
        self.tokens_ep_start = self.ep_rank * self.tokens_per_ep
        self.tokens_ep_end = self.tokens_ep_start + self.tokens_per_ep
        
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
            scatter_tokens = torch.randint(
                low=-16, 
                high=17, 
                size=[self.real_scatter_tokens, self.hidden_size], 
                dtype=torch.int8, 
                device="cuda"
            )

            per_token_scale = torch.ones(
                size=[self.real_scatter_tokens], 
                dtype=torch.float32, 
                device="cuda"
            )

            experts_weight = torch.randint(
                low=-16, 
                high=17, 
                size=[self.total_experts_num, self.hidden_size, self.new_hidden_size], 
                dtype=torch.int8, 
                device="cuda"
            )

            experts_scale = torch.ones(
                size=[self.total_experts_num, self.new_hidden_size], 
                dtype=torch.float32, 
                device="cuda"
            )

            experts_token_count = torch.tensor(
                self.token_list, 
                dtype=torch.int32, 
                device="cuda"
            )

            experts_token_start = torch.tensor(
                self.token_start_list, 
                dtype=torch.int32, 
                device="cuda"
            )

            y = torch.zeros(
                size=[self.real_scatter_tokens, self.new_hidden_size], 
                dtype=torch.bfloat16, 
                device="cuda"
            )

            tensors = {
                'scatter_tokens': scatter_tokens,
                'per_token_scale': per_token_scale,
                'experts_weight': experts_weight,
                'experts_scale': experts_scale,
                'experts_token_count': experts_token_count,
                'experts_token_start': experts_token_start,
                'y': y,
            }
            all_tensor_list.append(tensors)
            
        return all_tensor_list


    def tensor_size(self):
        size = 0

        size += self.real_scatter_tokens * self.hidden_size * torch.int8.itemsize  # scatter_tokens
        size += self.real_scatter_tokens * torch.float32.itemsize  # per_token_scale
        size += self.total_experts_num * self.hidden_size * self.new_hidden_size * torch.int8.itemsize  # experts_weight
        size += self.total_experts_num * self.new_hidden_size * torch.float32.itemsize  # experts_scale
        size += self.total_experts_num * torch.int32.itemsize  # experts_token_count
        size += self.total_experts_num * torch.int32.itemsize  # experts_token_start
        
        size += self.real_scatter_tokens * self.new_hidden_size * torch.bfloat16.itemsize  # y
        return size
    
    def op(self, tensors):
        # get pre-allocated input tensors
        scatter_tokens = tensors["scatter_tokens"]
        per_token_scale = tensors["per_token_scale"]
        experts_weight = tensors["experts_weight"]
        experts_scale = tensors["experts_scale"]
        experts_token_count = tensors["experts_token_count"]
        experts_token_start = tensors["experts_token_start"]

        # get pre-allocated output tensor
        y = tensors["y"]


        # use loop gemm and fp32 to simulate int8 group_gemm
        for i in range(self.total_experts_num):
            cur_token_start = experts_token_start[i]
            cur_token_end = cur_token_start + experts_token_count[i]

            cur_tokens = scatter_tokens[cur_token_start:cur_token_end]
            cur_tokens_scale = per_token_scale[cur_token_start:cur_token_end]

            cur_weight = experts_weight[i]
            cur_weight_scale = experts_scale[i]

            y[cur_token_start:cur_token_end] = fake_quant_gemm(
                cur_tokens, cur_tokens_scale, 
                cur_weight, cur_weight_scale, 
                torch.bfloat16
            )
        return y
    
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
    op = MoeQuantGroupGemmOp()
    op.run()
