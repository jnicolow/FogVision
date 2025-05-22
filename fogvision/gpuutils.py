import torch


def print_gpu_info():
    # prints out gpu memorry info and number of gpus
    num_gpus = torch.cuda.device_count()
    print(f'Number of GPUs: {num_gpus}')

    for i in range(num_gpus):
        device = torch.device(f'cuda:{i}')
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        
        free_memory = total_memory - allocated_memory
        
        print(f"GPU {i}:")
        print(f"  Total Memory: {total_memory / (1024 ** 2):.2f} MB")
        print(f"  Allocated Memory: {allocated_memory / (1024 ** 2):.2f} MB")
        print(f"  Cached Memory: {cached_memory / (1024 ** 2):.2f} MB")
        print(f"  Free Memory: {free_memory / (1024 ** 2):.2f} MB")


def get_gpu_most_memory():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0: return 'cpu' # this means there are no gpus
    free_memories = []
    for i in range(num_gpus):
        device = torch.device(f'cuda:{i}')
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device) # this should be memory cach exchange
        
        free_memory = total_memory - (allocated_memory + cached_memory)
        
        free_memories.append(free_memory)
    best_gpu = max(enumerate(free_memories),key=lambda x: x[1])[0]
        
    return f'cuda:{best_gpu}'

    