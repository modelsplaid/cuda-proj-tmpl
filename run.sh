
./build/src/atten_cuda  --head_number=32 --batch_size=64 --head_size=128 --head_size_v=64 --seq_length=1 --seq_length_kv=768  --iterations=1 --fixed_seq_length=true --causal=false
./build/src/atten_cuda  --head_number=32 --batch_size=64 --head_size=128 --head_size_v=64 --seq_length=1 --seq_length_kv=1  --iterations=1 --fixed_seq_length=true --causal=false
./build/src/atten_cuda  --head_number=32 --batch_size=64 --head_size=128 --head_size_v=64 --seq_length=1 --seq_length_kv=769  --iterations=1 --fixed_seq_length=true --causal=false

