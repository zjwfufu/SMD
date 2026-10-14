"""Sample with an exported student denoiser, or an H3 student adapter."""
import argparse,json,sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from smd.configs.reference import load_experiment
from smd.methods.simplified_distill.policy import SegmentPolicy
from smd.methods.simplified_distill.causal import CausalSegmentPolicy
from smd.methods.simplified_distill.basis import create_basis
from smd.networks.io_contract import NetworkIOAdapter
from smd.utils import instantiate
from smd.trainer import Trainer


def apply_sampling_metadata(config, directory):
    """Use the exported student's schedule and causal protocol for inference."""
    path=directory/'sampling.json'
    if not path.exists():path=directory/'bundle.json'
    metadata=json.loads(path.read_text())
    sampler=metadata['sampler']
    for name in ('student_sample_steps','final_step_size_scale','embedded_guidance_scale','input_shape'):
        if name in sampler:setattr(config,name,sampler[name])
    schedule=metadata.get('noise_scheduler',{})
    dynamic=bool(schedule.get('dynamic_shift',config.dynamic_shift))
    config.dynamic_shift=dynamic
    if dynamic:
        for name in ('base_shift','max_shift','base_seq_len','max_seq_len'):
            if name in schedule:setattr(config,'dynamic_shift_'+name,schedule[name])
        if 'time_shift_type' in schedule:config.time_shift_type=schedule['time_shift_type']
        if 'empirical_steps' in schedule:config.dynamic_shift_empirical_steps=int(schedule['empirical_steps'])
    elif 'shift' in schedule:
        config.net.shift=float(schedule['shift'])
        config.sample_t_cfg.shift=float(schedule['shift'])
    if sampler['type']=='causal_wan_chunked_student':
        if sampler.get('protocol_version')!=1:raise ValueError('Unsupported causal sampling protocol')
        for name in ('transition_type','renoise_mode','cache_noise'):
            setattr(config,'causal_'+name,sampler[name])
        for name in ('chunk_size','total_num_frames','use_wan_official_sinusoidal'):
            config.net[name]=sampler[name]
    return metadata


def load_denoiser(module, directory, skip_prefixes=()):
    index_path=directory/'diffusion_pytorch_model.safetensors.index.json'
    if index_path.exists():index=json.loads(index_path.read_text())
    else:
        from smd.inference.tensors import TensorReader
        filename='weights.safetensors'
        if not (directory/filename).exists():raise ValueError('No full student weights; export and fuse the student adapter first')
        reader=TensorReader(directory/filename)
        index={'weight_map':{name:filename for name in reader.keys()}}
    expected={key for key in module.state_dict() if not any(key.startswith(prefix) for prefix in skip_prefixes)};supplied=set(index['weight_map'])
    if expected!=supplied:
        raise ValueError(f'Weight namespace mismatch: missing={sorted(expected-supplied)[:12]}, unexpected={sorted(supplied-expected)[:12]}')
    for filename in dict.fromkeys(index['weight_map'].values()):
        if Path(filename).name!=filename:raise ValueError('Invalid shard filename')
        state=load_file(directory/filename,device='cpu')
        correct={k for k,v in index['weight_map'].items() if v==filename}
        if set(state)!=correct:raise ValueError(f'Shard/index mismatch: {filename}')
        result=module.load_state_dict(state,strict=False)
        if result.unexpected_keys:raise ValueError(result.unexpected_keys)


from smd.inference.media import decode_3d, save_media


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment');p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prompt')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--reduced-geometry',action='store_true')
    p.add_argument('--decode-resolution',type=int);p.add_argument('--render-size',type=int,default=256)
    p.add_argument('--dataset-case-index',type=int,help='Fixed, unshuffled case from this experiment training dataset')
    p.add_argument('--dataset-route',help='Optional route for a routed prompt dataset')
    p.add_argument('--condition-cache',type=Path,help='H3 production condition dataset; defaults to the bundled examples')
    p.add_argument('--base',type=Path,help='H3 local snapshot containing the pinned base components')
    p.add_argument('--transformer-base',type=Path,help='H3 local snapshot containing transformer or transformer_ref')
    p.add_argument('--ff-chunk-size',type=int,default=1024,help='H3 inference FF chunk size')
    p.add_argument('--qk-norm-chunk-size',type=int,default=2048,help='H3 inference Q/K norm chunk size')
    p.add_argument('--n-gpus','--n_gpus',type=int,help='H3 GPU count: default 1 with python, or the existing torchrun world size')
    args=p.parse_args()
    if args.experiment.startswith('h3_'):
        from smd.inference.h3 import launch_if_needed, run
        try:
            if launch_if_needed(args):return
        except ValueError as error:
            p.error(str(error))
        return run(args)
    if args.n_gpus is not None:p.error('--n-gpus is an H3-only option')
    if any(value is not None for value in (args.condition_cache,args.base,args.transformer_base)):
        p.error('--condition-cache, --base and --transformer-base are H3-only options')
    if args.prompt is None:args.prompt='A red kite flying over a green field.'
    experiment_config=load_experiment(args.experiment);cfg=experiment_config.model
    if args.dataset_case_index is None and args.experiment in {'qwen_image_edit','hunyuan3d_shape','trellis2_shape','trellis2_texture'}:
        args.dataset_case_index=0
    dataset_data=None;input_manifest=None
    if args.dataset_case_index is not None:
        from smd.inference.dataset_cases import select_case, save_input
        dataset_data,input_manifest=select_case(experiment_config,args.experiment,args.dataset_case_index,args.dataset_route)
        args.output.mkdir(parents=True,exist_ok=True)
        save_input(dataset_data,input_manifest,args.output)
        (args.output/'input_case.json').write_text(json.dumps(input_manifest,ensure_ascii=False,indent=2))
    elif args.dataset_route is not None:
        p.error('--dataset-route requires --dataset-case-index')
    metadata=apply_sampling_metadata(cfg,args.weights)
    cfg.net.use_lora=False;cfg.device='cuda';device=torch.device('cuda');dtype=getattr(torch,cfg.precision)
    torch.manual_seed(args.seed)
    family=cfg.net._target_.split('.')[2]
    if family not in {'Hunyuan3DShape','Trellis2'}:cfg.net.load_pretrained=False
    net=instantiate(cfg.net).to(dtype=dtype)
    if net.network_family!=metadata['network_family']:raise ValueError('Model and weight package families differ')
    denoiser_directory=args.weights/'transformer' if (args.weights/'transformer').is_dir() else args.weights
    load_denoiser(net.get_inference_module(),denoiser_directory, net.bundle_skip_prefixes)
    net.eval().requires_grad_(False)
    if hasattr(net,'init_preprocessors'):net.init_preprocessors()
    net.to(device=device,dtype=dtype)
    for name in ['text_encoder','vae','image_encoder']:
        value=getattr(net,name,None)
        if value is not None and hasattr(value,'to'):value.to(device=device,dtype=dtype)
    if args.reduced_geometry:
        if net.network_family=='wan':cfg.input_shape=[16,3,16,16]
        elif net.network_family in {'flux','qwen_image','qwen_image_edit','sd35medium'}:cfg.input_shape=[16,16,16]
        elif net.network_family in {'flux2','lens'}:cfg.input_shape=[128,16,16]
    causal=args.experiment.endswith('causal')
    if causal:net.total_num_frames=cfg.input_shape[1]
    adapter=NetworkIOAdapter.from_model(net)
    if dataset_data is not None:
        data=dataset_data
    else:data={'condition':[args.prompt],'neg_condition':[cfg.neg_prompt or '']}
    # Reuse the actual training preprocessing contract for text, edit images and 3D conditions.
    runtime=type('InferenceRuntime',(),{'net':net,'config':cfg,'device':device,'precision':dtype,
                  'precision_amp_enc':getattr(torch,cfg.precision_amp_enc) if cfg.precision_amp_enc else None,'input_shape':cfg.input_shape})()
    data=Trainer.preprocess_data(object.__new__(Trainer),runtime,data)
    condition=adapter.pack_condition(data['condition'],data)
    input_shape=adapter.input_shape_from_data(data,default_input_shape=list(cfg.input_shape))
    edit_conditioning = None
    if net.network_family == 'qwen_image_edit':
        edit_conditioning = {'source_latent_shape': list(data['source_latents'].shape),
                             'condition_tokens': int(data['condition'][1].sum().item()),
                             'noise_shape': [1, *input_shape]}
    policy_type=CausalSegmentPolicy if causal else SegmentPolicy
    generator=torch.Generator(device=device).manual_seed(args.seed)
    if net.network_family=='trellis2':
        from smd.methods.simplified_distill.sparse import SparseSegmentPolicy
        policy_type=SparseSegmentPolicy
        noise=data['real'].replace(torch.randn(data['real'].feats.shape,device=device,dtype=dtype,generator=generator))
    else:noise=torch.randn(1,*input_shape,device=device,dtype=dtype,generator=generator)
    policy=policy_type(net,None,None,create_basis(cfg),cfg,device,dtype)
    with torch.no_grad():
        latent=policy.sample_student(noise,condition,net,generator=generator) if causal else policy.sample_student(noise,condition,net)
        values=latent.feats if hasattr(latent,'feats') else latent
        if not torch.isfinite(values).all():raise RuntimeError('Non-finite inference output')
        net.get_inference_module().to(device='cpu')
        torch.cuda.empty_cache()
        decoded=decode_3d(net,latent,data,resolution=args.decode_resolution,render_size=args.render_size) if net.network_family in {'trellis2','hunyuan3d_shape'} else net.vae.decode(latent)
        panels=decoded.values() if isinstance(decoded,dict) else [decoded]
        if not all(torch.isfinite(value).all() for value in panels):raise RuntimeError('Non-finite decoded output')
    args.output.mkdir(parents=True,exist_ok=True)
    save_media(decoded,args.output)
    torch.save({'latent':values.cpu(),'decoded':{k:v.cpu() for k,v in decoded.items()} if isinstance(decoded,dict) else decoded.cpu()},args.output/'sample.pt')
    (args.output/'inference.json').write_text(json.dumps({'experiment':args.experiment,'seed':args.seed,'shape':list(values.shape),'decoded':decoded is not None,'finite':True,'sampler':metadata['sampler'],'reduced_geometry':args.reduced_geometry,'input':input_manifest,'edit_conditioning':edit_conditioning},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
