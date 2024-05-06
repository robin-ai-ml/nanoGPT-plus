"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

'''
These lines set the random seed for CPU and CUDA (GPU), respectively. 
Setting the seed ensures reproducibility in your experiments. 
This means that every time you run your script with the same seed and under the same conditions, 
you will get the exact same results, which is crucial for debugging and comparing models.
'''
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

'''
TensorFloat-32 (TF32): This is a data format introduced by NVIDIA for their Ampere architecture GPUs. 
TF32 allows faster computations by reducing the precision requirements for matrix multiplications 
and convolution operations without significant loss in model accuracy.
MatMul and CuDNN Settings: These settings enable TF32 for matrix multiplication operations (matmul) and CuDNN operations. 
CuDNN is NVIDIA's GPU-accelerated library for deep neural networks. Enabling TF32 can lead to performance improvements 
in training and inference on compatible NVIDIA GPUs.
'''
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

'''
float32 is the standard floating point precision,
float16 (half precision) can speed up computation and reduce memory usage,
bfloat16 provides a balance between the range of float32 and the precision near zero of float16.
'''
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

print(f"using device: {device}")

# model
if init_from == 'resume':
    print('init from a model saved in a specific directory')
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    print('init from a given GPT-2 model')
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

'''
The model is switched to evaluation mode using model.eval(), 
which is essential for testing as it disables dropout and batch normalization effects 
that are typically used during training.
'''
model.eval()
model.to(device)
if compile:
    #Optimizes given model/function using TorchDynamo and specified backend.
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    #'data/shakespeare_char/meta.pkl'
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 eyncodings...")
    #tiktoken is a fast BPE tokeniser for use with OpenAI's models.
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
'''
The list of integers (start_ids) is then converted into a PyTorch tensor 
with data type torch.long and moved to the appropriate device (CPU or GPU 
as specified by the device variable). Adding [None, ...] reshapes the tensor 
by adding a batch dimension, which is necessary for the model to process it correctly.
'''
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
'''
torch.no_grad(): Disables gradient computation, reducing memory consumption and speeding up computations. 
This is used here because gradients are not needed when merely generating text (inference mode).
'''
with torch.no_grad():
    '''
    ctx: Activates the context for automatic mixed precision if necessary, 
    or provides a no-operation context otherwise, 
    as set up previously depending on whether the model is running on CPU or GPU.
    '''
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
