import torch

cuda_available = torch.cuda.is_available()
mps_available = False #torch.mps.is_available()
device_count = torch.cuda.device_count() if cuda_available else 0
all_ids = list(range(device_count))

def set_device(*args, prefer_gpu=True):
 
    prefer_cuda_flag = prefer_gpu
    
    if args:

        if len(args) == 1 and isinstance(args[0], bool):
            prefer_cuda_flag = args[0]
            ids = None

        else:

            ids = []
            for arg in args:
                if isinstance(arg, list):
                    ids.extend(arg)
                elif isinstance(arg, int):
                    ids.append(arg)
                elif isinstance(arg, tuple):
                    ids.extend(list(arg))
            
            ids = sorted(set(ids))
            prefer_cuda_flag = prefer_gpu if ids else prefer_cuda_flag
    else:
        ids = None
    
    if ids is not None:

        if cuda_available and prefer_cuda_flag:

            valid_ids = [i for i in ids if i < device_count]
            if valid_ids:
                if len(valid_ids) == 1:
                    return f"cuda:{valid_ids[0]}"
                else:
                    return f"cuda:{','.join(map(str, valid_ids))}"
            else:
                return "cuda:0"
        elif mps_available and prefer_cuda_flag:
            return "mps"
        else:
            return "cpu"
    else:

        if cuda_available and prefer_cuda_flag:

            if device_count == 1:
                return "cuda:0"
            elif device_count > 1:
                return f"cuda:{','.join(map(str, all_ids))}"
            else:
                return "cpu"
        elif mps_available and prefer_cuda_flag:
            return "mps"
        else:
            return "cpu"