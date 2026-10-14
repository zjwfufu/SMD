"""Safetensors access shared by inference and weight conversion."""
import json
import struct
import torch

DTYPES = {'F64':torch.float64,'F32':torch.float32,'F16':torch.float16,'BF16':torch.bfloat16,
          'I64':torch.int64,'I32':torch.int32,'I16':torch.int16,'I8':torch.int8,'U8':torch.uint8,'BOOL':torch.bool}


class TensorReader:
    """Safetensors reader with local or HTTP range access."""
    def __init__(self, source):
        self.source = str(source)
        self.remote = self.source.startswith(('https://', 'http://'))
        length = struct.unpack('<Q', self.read_at(0, 8))[0]
        if length > 100_000_000:
            raise ValueError('Invalid safetensors header length')
        self.header = json.loads(self.read_at(8, length))
        self.offset = 8 + length

    def read_at(self, offset, length):
        if not length:return b''
        if self.remote:
            import requests
            try:
                response = requests.get(self.source, headers={'Range':f'bytes={offset}-{offset+length-1}'},timeout=120)
            except requests.RequestException:
                raise RuntimeError('Weight range request failed') from None
            if response.status_code != 206:
                raise RuntimeError(f'Weight range request returned HTTP {response.status_code}')
            content=response.content
        else:
            with open(self.source, 'rb') as stream:
                stream.seek(offset);content=stream.read(length)
        if len(content)!=length:raise ValueError('Truncated weight range')
        return content

    def keys(self):return sorted((key for key in self.header if key!='__metadata__'), key=lambda key:self.header[key]['data_offsets'][0])

    def tensor(self, key):
        spec=self.header[key];start,end=spec['data_offsets']
        if end==start:return torch.empty(spec['shape'],dtype=DTYPES[spec['dtype']])
        data=bytearray(self.read_at(self.offset+start,end-start))
        return torch.frombuffer(data,dtype=DTYPES[spec['dtype']]).reshape(spec['shape'])


    def tensors(self, keys):
        start=min(self.header[k]['data_offsets'][0] for k in keys)
        end=max(self.header[k]['data_offsets'][1] for k in keys)
        payload=bytearray(self.read_at(self.offset+start,end-start))
        result={}
        for key in keys:
            spec=self.header[key];a,b=spec['data_offsets']
            result[key]=torch.frombuffer(memoryview(payload)[a-start:b-start],dtype=DTYPES[spec['dtype']]).reshape(spec['shape']) if b>a else torch.empty(spec['shape'],dtype=DTYPES[spec['dtype']])
        return result
