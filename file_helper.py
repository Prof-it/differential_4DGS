#
# Copyright (C) 2025, Felix Hirt
# All rights reserved.
#

import torch
import numpy as np
import struct

def save_tensor_to_file(tensor, filename):
    """
    save reference counts to file.
    """ 
    arr = tensor.detach().cpu().numpy().astype(np.uint8)
    N = arr.size  
    
    with open(filename, "wb") as f:
        #write the number of elements as a 4-byte little-endian unsigned int.
        f.write(struct.pack("<I", N))
        
        #if there is an odd number of elements, pad with a zero so we can pack in pairs.
        if N % 2 != 0:
            arr = np.concatenate([arr, np.array([0], dtype=np.uint8)])
        
        #pack two numbers per byte:
        #for every two numbers, the first becomes the high nibble and the second the low nibble.
        packed = bytearray()
        for i in range(0, len(arr), 2):
            byte_val = (arr[i] << 4) | (arr[i + 1] & 0x0F)
            packed.append(byte_val)
        
        #write the packed bytes to the file.
        f.write(packed)

def load_tensor_from_file(filename):
    """
    load a binary file back into a tensor.
    """
    with open(filename, "rb") as f:
        #read the header: first 4 bytes hold the number of elements.
        header = f.read(4)
        if len(header) < 4:
            raise ValueError("File too short: missing header.")
        N = struct.unpack("<I", header)[0]
        
        #read the packed data bytes.
        packed = f.read()
    
    #unpack each byte into two numbers (nibbles).
    result = []
    for byte_val in packed:
        #get the high nibble.
        result.append(byte_val >> 4)
        #append the low nibble only if we haven't already collected N numbers.
        if len(result) < N:
            result.append(byte_val & 0x0F)
    
    #convert the list back to a PyTorch tensor.
    return torch.tensor(result, dtype=torch.uint8)