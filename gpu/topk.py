import torch
import random
import math

class TopkOp:

    def __init__(self):
        self.batch_size = 1024
        self.dim_size = 16384
        self.k = 16

        # SRAM 48MB
        cache_size = 48 * (1024 ** 2)
        tensor_size = self.tensor_size()
        max_data_cnt = math.ceil(cache_size / tensor_size)
        self.tensor_list = self.create_tensors(max_data_cnt)
        random.shuffle(self.tensor_list)
    
    def create_tensors(self, max_data_cnt):
        all_tensor_list = []
        for _ in range(max_data_cnt):
            src = torch.randn(
                size=(self.batch_size, self.dim_size), 
                dtype=torch.bfloat16, 
                device="cuda"
            )

            tensors = {
                'src': src,
            }
            all_tensor_list.append(tensors)
            
        return all_tensor_list


    def tensor_size(self):
        size=0
        # src
        size += self.batch_size * self.dim_size * torch.bfloat16.itemsize
        # dst
        size += self.batch_size * self.k * torch.bfloat16.itemsize * 2
        return size
    
    def op(self, tensors):
        src = tensors['src']
        value, indice = torch.topk(
            src, self.k, dim=-1, 
            largest=True, sorted=False
        )
        return value, indice
    
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
    op = TopkOp()
    op.run()