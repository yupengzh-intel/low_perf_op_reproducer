import torch
import random
import math

class HeadRMSNormOp:

    def __init__(self):
        self.num_tokens = 32768
        self.total_head_num = 8
        self.head_dim = 128

        self.norm_head_start = 0
        self.norm_head_num = 8
        self.norm_head_end = self.norm_head_start + self.norm_head_num

        self.eps = 1e-5

        # SRAM 96MB
        cache_size = 96 * (1024 ** 2)
        tensor_size = self.tensor_size()
        max_data_cnt = math.ceil(cache_size / tensor_size)
        self.tensor_list = self.create_tensors(max_data_cnt)
        random.shuffle(self.tensor_list)
    
    def create_tensors(self, max_data_cnt):
        all_tensor_list = []
        for _ in range(max_data_cnt):
            token_data = torch.randn(
                size=[self.num_tokens, self.total_head_num, self.head_dim], 
                dtype=torch.bfloat16, 
                device="cuda"
            )
            
            weight = torch.randn(
                size=[self.norm_head_num, self.head_dim], 
                dtype=torch.bfloat16, 
                device="cuda"
            )

            y = torch.randn(
                size=[self.num_tokens, self.total_head_num, self.head_dim], 
                dtype=torch.bfloat16, 
                device="cuda"
            )

            tensors = {
                'token_data': token_data,
                'weight': weight,
                'y': y
            }
            all_tensor_list.append(tensors)
            
        return all_tensor_list


    def tensor_size(self):
        size=0
        size+=self.num_tokens*self.total_head_num*self.head_dim*torch.bfloat16.itemsize
        size+=self.norm_head_num*self.head_dim*torch.bfloat16.itemsize
        size+=self.num_tokens*self.total_head_num*self.head_dim*torch.bfloat16.itemsize
        return size
    
    def op(self, tensors):
        # get pre-allocated input tensors
        token_data = tensors["token_data"]
        weight = tensors["weight"]

        # get pre-allocated output tensors
        y = tensors["y"]

        # per head rms_norm
        for head_idx in range(self.norm_head_num):
            head_data = token_data[:, head_idx, :]
            head_weight = weight[head_idx, :]
            y[:, head_idx, :] = torch.nn.functional.rms_norm(
                head_data, 
                normalized_shape=head_weight.shape,
                weight=head_weight,
                eps=self.eps
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
    op = HeadRMSNormOp()
    op.run()