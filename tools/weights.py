"""Shard full denoiser weights and fuse exported adapters without loading a full model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors.torch import save, load_file
from smd.inference.metadata import denoiser_config, sampling_metadata

from smd.inference.tensors import TensorReader


class IndexedTensorReader:
    """Read a native sharded denoiser with the same interface as a single file."""
    def __init__(self, directory):
        directory = Path(directory)
        index = json.loads((directory / 'diffusion_pytorch_model.safetensors.index.json').read_text())
        self.readers = {}
        self.locations = {}
        self.header = {}
        offset = 0
        for filename in dict.fromkeys(index['weight_map'].values()):
            if Path(filename).name != filename:
                raise ValueError('Invalid shard filename')
            reader = TensorReader(directory / filename)
            self.readers[filename] = reader
            expected = {name for name, shard in index['weight_map'].items() if shard == filename}
            if set(reader.keys()) != expected:
                raise ValueError(f'Shard/index mismatch: {filename}')
            for name in reader.keys():
                spec = reader.header[name]
                a, b = spec['data_offsets']
                self.header[name] = {**spec, 'data_offsets': [offset, offset + b - a]}
                self.locations[name] = filename
                offset += b - a

    def keys(self):
        return list(self.locations)

    def tensors(self, keys):
        grouped = {}
        for name in keys:
            grouped.setdefault(self.locations[name], []).append(name)
        return {name: value for filename, names in grouped.items()
                for name, value in self.readers[filename].tensors(names).items()}


class TorchTensorReader:
    """Read tensor-only PyTorch bases, including Shape and causal initialization."""
    def __init__(self, source, *, state_key=None, causal_base=False):
        state=torch.load(source,map_location='cpu',weights_only=True)
        if state_key is not None:state=state[state_key]
        if causal_base:
            from smd.networks.Wan.network_causal import convert_self_forcing_state_dict
            state={name.removeprefix('transformer.'):value
                   for name,value in convert_self_forcing_state_dict(state).items()}
        if not isinstance(state,dict) or not state or not all(torch.is_tensor(value) for value in state.values()):
            raise ValueError('Select a tensor state dict with --state-key, or use --causal-base')
        self.state=state;self.header={};offset=0
        for name,value in state.items():
            size=value.numel()*value.element_size()
            self.header[name]={'shape':list(value.shape),'data_offsets':[offset,offset+size]}
            offset+=size

    def keys(self):return list(self.state)

    def tensors(self,keys):return {name:self.state[name] for name in keys}


def open_weights(source, *, state_key=None, causal_base=False):
    if Path(source).suffix in {'.pt','.pth','.ckpt'}:
        return TorchTensorReader(source,state_key=state_key,causal_base=causal_base)
    if state_key is not None or causal_base:raise ValueError('PyTorch base options require a .pt/.pth/.ckpt input')
    return IndexedTensorReader(source) if Path(source).is_dir() else TensorReader(source)


def shard_weights(reader, write, *, max_bytes=1_000_000_000, transform=None):
    """Write each tensor once and return the native safetensors shard index."""
    groups=[];group=[];size=0
    for key in reader.keys():
        a,b=reader.header[key]['data_offsets'];n=b-a
        if size and size+n>max_bytes:groups.append(group);group=[];size=0
        group.append(key);size+=n
    if group:groups.append(group)
    weight_map={};hashes={};total=0
    for i,keys in enumerate(groups,1):
        state=reader.tensors(keys)
        if transform is not None:state=transform(state)
        for key in state:
            if key in weight_map:raise ValueError(f'Duplicate output tensor: {key}')
        total+=sum(t.numel()*t.element_size() for t in state.values())
        filename=f'diffusion_pytorch_model-{i:05d}-of-{len(groups):05d}.safetensors'
        payload=save(state,metadata={'format':'pt'})
        write(filename,payload);hashes[filename]=hashlib.sha256(payload).hexdigest()
        weight_map.update({key:filename for key in state})
    index={'metadata':{'total_size':total},'weight_map':weight_map}
    write('diffusion_pytorch_model.safetensors.index.json',json.dumps(index,indent=2).encode())
    write('sha256.json',json.dumps(hashes,indent=2).encode())
    return index


class AdapterFusion:
    """Strict float32 fusion of A/B tensors and explicitly saved trainable replacements."""
    def __init__(self, tensors, config):
        self.pairs={};self.replacements={};self.used=set();self.scales={}
        precision = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
        self.merge_dtype = precision[config['merge_precision']] if 'merge_precision' in config else None
        self.delta_dtype = precision[config['delta_precision']] if 'delta_precision' in config else None
        consumed=set()
        for key,a in tensors.items():
            if '.lora_A' not in key:continue
            prefix,suffix=key.split('.lora_A',1);bkey=prefix+'.lora_B'+suffix
            if bkey not in tensors:raise ValueError(f'Missing B tensor for {key}')
            name=prefix+'.weight';b=tensors[bkey]
            if a.ndim!=2 or b.ndim!=2 or b.shape[1]!=a.shape[0]:raise ValueError(f'Invalid adapter shape: {prefix}')
            self.pairs[name]=(a,b)
            scaling=config.get('scaling',{})
            self.scales[name]=float(scaling.get(prefix,config.get('lora_alpha',a.shape[0])/a.shape[0]))
            consumed.update((key,bkey))
        for key,value in tensors.items():
            if key not in consumed:
                if 'lora_' in key:raise ValueError(f'Unsupported adapter tensor: {key}')
                self.replacements[key]=value

    def __call__(self, state):
        result={}
        for key,base in state.items():
            if self.merge_dtype is not None:base=base.to(self.merge_dtype)
            if key in self.pairs:
                a,b=self.pairs[key];delta=b.float()@a.float()
                if base.shape!=delta.shape:raise ValueError(f'Base/adapter shape mismatch: {key}')
                delta=self.scales[key]*delta
                if self.delta_dtype is not None:delta=delta.to(self.delta_dtype).float()
                base=(base.float()+delta).to(base.dtype);self.used.add(key)
            if key in self.replacements:
                replacement=self.replacements[key]
                if replacement.shape!=base.shape:raise ValueError(f'Replacement shape mismatch: {key}')
                base=replacement.to(base.dtype);self.used.add(key)
            result[key]=base.contiguous()
        return result

    def finish(self):
        missing=(self.pairs.keys()|self.replacements.keys())-self.used
        if missing:raise ValueError(f'Unconsumed adapter tensors: {sorted(missing)}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True,help='Native denoiser config JSON')
    p.add_argument('--adapter',type=Path);p.add_argument('--adapter-config',type=Path)
    p.add_argument('--shard-bytes',type=int,default=1_000_000_000)
    p.add_argument('--state-key',help='Tensor dictionary key in a PyTorch base checkpoint')
    p.add_argument('--causal-base',action='store_true',help='Convert official causal initialization keys to the native Wan namespace')
    p.add_argument('--sampling',type=Path,help='Bundle or sampling JSON; export only loading and inference metadata')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):raise ValueError('Output directory must be empty')
    config=denoiser_config(json.loads(args.config.read_text()))
    fusion=None
    if args.adapter:
        if args.adapter_config is None:raise ValueError('--adapter-config is required with --adapter')
        fusion=AdapterFusion(load_file(args.adapter),json.loads(args.adapter_config.read_text()))
    def write(name,payload):(args.output/name).write_bytes(payload)
    try:
        shard_weights(open_weights(args.input,state_key=args.state_key,causal_base=args.causal_base),write,max_bytes=args.shard_bytes,transform=fusion)
        if fusion:fusion.finish()
        write('config.json',json.dumps(config,indent=2).encode())
        if args.sampling:
            metadata=sampling_metadata(json.loads(args.sampling.read_text()))
            metadata['weights']='diffusion_pytorch_model.safetensors.index.json'
            write('sampling.json',json.dumps(metadata,indent=2).encode())
    except Exception:
        # Failed exports are not valid packages.
        for file in args.output.iterdir():file.unlink()
        raise

if __name__=='__main__':main()
