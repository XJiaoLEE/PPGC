import DPmechanisms
import torch

class TernGradCompressor:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        return
    
    def compress(self, tensor):
        shape = tensor.size()
        tensor = tensor.flatten()

        if self.epsilon > 0 :
            sensitivity = tensor.max() - tensor.min()
            tensor = DPmechanisms.add_laplace(tensor, sensitivity, self.epsilon)

        std = (tensor - torch.mean(tensor)) ** 2
        std = torch.sqrt(torch.mean(std))
        c = 2.5 * std.item()
        gradient = torch.clamp(tensor, -c, c)
        abs_gradient = gradient.abs()
        scalar = abs_gradient.max()
        
        sign_gradient = gradient.sign() * scalar
        rnd_sample = torch.empty_like(tensor).uniform_(0, scalar.item())
        sign_gradient[rnd_sample >= abs_gradient] = 0
        new_sign = sign_gradient.sign()  # -1, 0, 1

        tensor_compressed = new_sign.type(torch.int8), scalar.flatten()

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, scalar = tensor_compressed
        sign = tensor_compressed.type(torch.float32)
        tensor_decompressed = sign * scalar
        return tensor_decompressed.view(shape)
    

class PruningAwareTernGradCompressor():
    def __init__(self):
        self.mask_map = {}
        self.update_count_map = {}
        self.stable_map = {}
        
    def compress(self, bucket):
        tensor = bucket.buffer()
        shape = tensor.size()
        tensor = tensor.flatten()
        index = bucket.index()

        std = (tensor - torch.mean(tensor)) ** 2
        std = torch.sqrt(torch.mean(std))
        c = 2.5 * std.item()
        gradient = torch.clamp(tensor, -c, c)
        abs_gradient = gradient.abs()
        scalar = abs_gradient.max()
        
        sign_gradient = gradient.sign() * scalar
        new_sign = sign_gradient.sign()  # -1, 0, 1

        if index in self.stable_map and self.stable_map[index]:
            new_sign = new_sign[~self.mask_map[index]]

        tensor_compressed = new_sign.type(torch.int8), scalar.flatten()
        
        return tensor_compressed, shape
    
    def zero_mask_track(self, bucket, decompressed_tensor):
        index = bucket.index()
        current_mask = (decompressed_tensor == 0).type(torch.bool)
        if index not in self.mask_map or self.mask_map[index].size() != current_mask.size():
            self.mask_map[index] = current_mask
            return 
        previous_mask = self.mask_map[index]
        self.mask_map[index] = previous_mask & current_mask
        
        num_updates = (self.mask_map[index] != previous_mask).sum().item()
        self.stable_track(index=index, num_updates=num_updates)
    
    def stable_track(self, index, num_updates):
        if index not in self.stable_map:
            self.stable_map[index] = False
        if self.stable_map[index]:
            return 
        if num_updates == 0:
            if index not in self.update_count_map:
                self.update_count_map[index] = 1
            else:
                self.update_count_map[index] += 1
        else:
            self.update_count_map[index] = 0

        if self.update_count_map[index] >= 10:
            self.stable_map[index] = True
            print(f'index {index} is stable')

    def decompress(self, compressed_data, shape, bucket):
        index = bucket.index()
        tensor_compressed, scalar = compressed_data
        if tensor_compressed.size() != shape:
            sign = torch.zeros(shape, dtype=torch.float32, device=tensor_compressed.device)
            sign[~self.mask_map[index]] = tensor_compressed.type(torch.float32)
        else:
            sign = tensor_compressed.type(torch.float32)
        tensor_decompressed = sign * scalar
        return tensor_decompressed