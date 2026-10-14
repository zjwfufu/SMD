"""Public weight metadata: loading and sampling, without training provenance."""
from copy import deepcopy

SAMPLING_FIELDS = {
    'schema_version', 'network_family', 'model_id', 'precision', 'weights', 'adapter_base',
}
SCHEDULE_FIELDS = {
    'type', 'schedule_type', 'network_family', 'shift', 'min_t', 'max_t',
    'sigma_min', 'dynamic_shift', 'base_shift', 'max_shift', 'base_seq_len',
    'max_seq_len', 'time_shift_type', 'empirical_steps',
    'video_shift', 'audio_shift', 'supports_joint_av',
}
SAMPLER_FIELDS = {
    'type', 'student_sample_steps', 'final_step_size_scale', 'embedded_guidance_scale',
    'input_shape', 'protocol_version', 'chunk_size', 'total_num_frames',
    'transition_type', 'renoise_mode', 'cache_noise', 'use_wan_official_sinusoidal',
    'model_id', 'model_revision', 'condition_mode', 'transformer_subfolder',
    'video_shift', 'audio_shift', 'condition_video_timestep', 'guidance_distilled',
    'fps', 'num_frames', 'height', 'width',
}


def sampling_metadata(source):
    """Select the public inference protocol from a training bundle or package."""
    result = {key: deepcopy(value) for key, value in source.items() if key in SAMPLING_FIELDS}
    result['sampler'] = {key: deepcopy(value) for key, value in source['sampler'].items()
                         if key in SAMPLER_FIELDS}
    if 'noise_scheduler' in source:
        result['noise_scheduler'] = {
            key: deepcopy(value) for key, value in source['noise_scheduler'].items()
            if key in SCHEDULE_FIELDS
        }
    return result


def denoiser_config(source):
    """Keep native architecture settings, stripping exporter provenance fields."""
    # Class names are needed by native loaders; other private Diffusers fields
    # describe the machine, library or checkpoint that produced the config.
    provenance = {'iteration', 'global_step', 'training_step', 'selected_iteration',
                  'basis', 'residual_basis', 'num_basis_terms', 'checkpoint_path',
                  'source_checkpoint', 'training_config', 'experiment', 'run_id',
                  'created_at', 'wandb'}
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()
                    if key not in provenance and (not key.startswith('_') or key == '_class_name')}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return deepcopy(value)
    return clean(source)
