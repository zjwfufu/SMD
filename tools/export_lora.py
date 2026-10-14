"""Export trainable student adapters and saved native replacements from a DCP checkpoint."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from safetensors.torch import save


def export(checkpoint, output, *, rank, alpha, base_model, strip_prefix, writer=None,
           merge_precision=None, delta_precision=None, sampling=None):
    public_sampling = None
    if sampling is not None:
        from smd.inference.metadata import sampling_metadata
        source = json.loads(Path(sampling).read_text())
        public_sampling = sampling_metadata(source)
        if source.get('network_family') == 'minimax_h3':
            from smd.inference.h3 import validate_sampling
            validate_sampling(public_sampling)
            if base_model != public_sampling['sampler']['model_id']:
                raise ValueError('Adapter base must match the H3 training bundle')
            prefix = source.get('checkpoint_path')
            if prefix and Path(checkpoint).name != Path(prefix).name + '.lora_net_model':
                raise ValueError('H3 sampling bundle must accompany this student checkpoint')
    precisions = ('float32', 'float16', 'bfloat16')
    for value in (merge_precision, delta_precision):
        if value is not None and value not in precisions:
            raise ValueError(f'Unsupported merge precision: {value}')
    reader=FileSystemReader(str(checkpoint));metadata=reader.read_metadata()
    state={key:torch.empty(value.size,dtype=value.properties.dtype) for key,value in metadata.state_dict_metadata.items() if isinstance(value,TensorStorageMetadata)}
    if len(state)!=len(metadata.state_dict_metadata):raise ValueError('Student checkpoint must contain only tensor parameters')
    dcp.load(state,storage_reader=reader)
    result={}
    for name,value in state.items():
        name=name.replace('_checkpoint_wrapped_module.','')
        if not name.startswith(strip_prefix):raise ValueError(f'Unexpected student namespace: {name}')
        name=name[len(strip_prefix):].replace('.lora_A.default.weight','.lora_A.weight').replace('.lora_B.default.weight','.lora_B.weight')
        if name.endswith(('.lora_A','.lora_B')):name+='.weight'
        if name in result:raise ValueError(f'Duplicate exported parameter: {name}')
        result[name]=value.contiguous()
    if not any('.lora_A.' in name for name in result):raise ValueError('No original LoRA tensors in checkpoint')
    if public_sampling is not None and public_sampling.get('network_family') == 'minimax_h3':
        from smd.inference.h3 import validate_adapter
        validate_adapter(result, rank=rank, alpha=alpha)
    output=Path(output)
    if writer is None:
        if output.exists() and any(output.iterdir()):raise ValueError('Output directory must be empty')
        output.mkdir(parents=True,exist_ok=True)
        writer=lambda name,payload:(output/name).write_bytes(payload)
    payload = save(result,metadata={'format':'pt'})
    writer('lora.safetensors',payload)
    config={'format':'smd_lora','r':rank,'lora_alpha':alpha,'base_model':base_model,'saved_replacements':[n for n in result if '.lora_' not in n]}
    if merge_precision is not None: config['merge_precision'] = merge_precision
    if delta_precision is not None: config['delta_precision'] = delta_precision
    writer('adapter_config.json',(json.dumps(config,indent=2)+'\n').encode())
    if public_sampling is not None:
        import hashlib
        public_sampling['weights'] = 'lora.safetensors'
        public_sampling['weights_sha256'] = hashlib.sha256(payload).hexdigest()
        writer('sampling.json',(json.dumps(public_sampling,indent=2)+'\n').encode())
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--alpha',type=float,required=True)
    p.add_argument('--base-model',required=True);p.add_argument('--strip-prefix',default='transformer.')
    p.add_argument('--sampling',type=Path,help='Matching training bundle.json; required for H3 inference')
    p.add_argument('--merge-precision', choices=('float32','float16','bfloat16'),
                   help='Base/output dtype for fusion; default preserves the base dtype')
    p.add_argument('--delta-precision', choices=('float32','float16','bfloat16'),
                   help='Round the scaled LoRA product before addition; default keeps float32')
    args=p.parse_args();export(args.checkpoint,args.output,rank=args.rank,alpha=args.alpha,base_model=args.base_model,strip_prefix=args.strip_prefix,
                              merge_precision=args.merge_precision,delta_precision=args.delta_precision,sampling=args.sampling)

if __name__=='__main__':main()
