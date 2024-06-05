import torch
import time
# /data/tzq/.local/lib/python3.10/site-packages/deepspeed/inference/v2/kernels/ragged_ops/blocked_flash
from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.inference_utils import DtypeEnum
from deepspeed.inference.v2.kernels.ragged_ops import (
    AtomBuilder,
    BlockedFlashAttn,
    get_q_block_size,
    get_kv_block_size,
    LinearBlockedKVCopy,
)
from deepspeed.inference.v2.ragged import split_kv
from deepspeed.ops.op_builder import RaggedUtilsBuilder


import sys
sys.path.append("/data/tzq/DeepSpeed/tests/unit/inference/v2/kernels/ragged_ops")
from ragged_testing_utils import build_batch_and_manager



def test_block_flash_attn(seq_params=[(768,0)]):
        
    head_size  = 128
    n_heads_q  = 32
    n_heads_kv = 32
    

    atom_flash = BlockedFlashAttn(head_size, DtypeEnum.fp16)

    q_block_size = get_q_block_size(head_size)
    kv_block_size = get_kv_block_size(head_size)

    kvs = []
    for qlen, history_len in seq_params:
        
        #print("qlen: ",qlen,"history_vklen: ", history_len)
        
        if history_len > 0:
            kvs.append(
                torch.randn((history_len, 2 * n_heads_kv * head_size),
                            device=get_accelerator().current_device(),
                            dtype=torch.float16))
        else:
            kvs.append(None)

    batch, state_manager, _ = build_batch_and_manager(seq_params, head_size, n_heads_kv, kv_block_size, kv_fill=kvs)

    atom_builder = AtomBuilder()
    kv_copy = LinearBlockedKVCopy(head_size, n_heads_q, n_heads_kv, DtypeEnum.fp16)
    atom_flash = BlockedFlashAttn(head_size, DtypeEnum.fp16)

    total_atoms = sum((seq[0] + q_block_size - 1) // q_block_size for seq in seq_params)
    atoms = torch.empty((total_atoms, 8), dtype=torch.int32, device=get_accelerator().current_device())
    alloc_func = RaggedUtilsBuilder().load().allocate_fast_host_buffer
    atoms_host = alloc_func(atoms)

    qkv = torch.randn((batch.current_tokens, (n_heads_q + 2 * n_heads_kv) * head_size),
                        device=get_accelerator().current_device(),
                        dtype=torch.float16)

    atoms_host, n_atoms = atom_builder(atoms_host, batch, q_block_size, kv_block_size)
    
    # for i in range(n_atoms):
    #     atoms_host[i][6]=766
        
    #print("atoms_host: ",atoms_host)
    #print("n_atoms: ",n_atoms)
    atoms.copy_(atoms_host[:n_atoms])
    #print("atoms: ",atoms)

    kv_cache = state_manager.get_cache(0)
    kv_copy(kv_cache, qkv, batch)

    out = torch.empty((batch.current_tokens, head_size * n_heads_q),
                        device=get_accelerator().current_device(),
                        dtype=torch.float16)
    k_cache, v_cache = split_kv(kv_cache)
    q = qkv[:, :head_size * n_heads_q]

    # warm up
    atom_flash(out, q, k_cache, v_cache, atoms, 1.0)
    atom_flash(out, q, k_cache, v_cache, atoms, 1.0)
    atom_flash(out, q, k_cache, v_cache, atoms, 1.0)
    atom_flash(out, q, k_cache, v_cache, atoms, 1.0)

    # real compute
    print("start compute ")

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    start_time = time.perf_counter()
    atom_flash(out, q, k_cache, v_cache, atoms, 1.0)

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print("time1: ",(end_time-start_time)* 1000000,"us" )
    torch.cuda.cudart().cudaProfilerStart()
    print("done compute ")
    
    return (end_time-start_time)* 1000000

def test_prefill(batch_sz = 1):
    qseq = 768
    histseq = 0
    params = [(qseq,histseq)]*batch_sz
    return test_block_flash_attn(seq_params=params)

def test_decode(batch_sz = 1):
    qseq = 1
    histseq = 768
    params = [(qseq,histseq)]*batch_sz
    return test_block_flash_attn(seq_params=params)

def test_prefill_decode(batch_sz = 1):
    qseq = 769
    histseq = 0
    params = [(qseq,histseq)]*batch_sz
    return test_block_flash_attn(seq_params=params)
    
if __name__ == "__main__":
    batch_sz = 64
    t_prefill=test_prefill(batch_sz)
    t_decode=test_decode(batch_sz)
    t_prefill_decode=test_prefill_decode(batch_sz)
    
    
    print("t_prefill: ",t_prefill,"t_decode: ",t_decode,"t_prefill_decode: ",t_prefill_decode)