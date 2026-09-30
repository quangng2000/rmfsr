"""libgsm full-rate GSM round trip for training only (8 kHz, 20 ms packets)."""
import ctypes as C
import ctypes.util
import os
import numpy as np
from scipy.signal import resample_poly

def library_path():
    path=os.environ.get('RMFSR_LIBGSM') or C.util.find_library('gsm')
    if not path and os.path.exists('/opt/homebrew/lib/libgsm.dylib'):path='/opt/homebrew/lib/libgsm.dylib'
    if not path:raise RuntimeError('libgsm required: brew install libgsm (macOS), libgsm1 (Linux)')
    return path

def roundtrip(x,sr):
    import math
    lib=C.CDLL(library_path());lib.gsm_create.restype=C.c_void_p
    lib.gsm_destroy.argtypes=[C.c_void_p]
    lib.gsm_encode.argtypes=[C.c_void_p,C.POINTER(C.c_short),C.POINTER(C.c_ubyte)]
    lib.gsm_decode.argtypes=[C.c_void_p,C.POINTER(C.c_ubyte),C.POINTER(C.c_short)]
    gcd=math.gcd(8000,sr);wave=resample_poly(x,8000//gcd,sr//gcd)
    count=len(wave);wave=np.pad(wave,(0,(-count)%160))
    pcm=np.round(wave.clip(-1,.999969)*32768).astype(np.int16)
    decoded=np.empty_like(pcm);enc=lib.gsm_create();dec=lib.gsm_create()
    if not enc or not dec:raise MemoryError('gsm_create')
    packet=(C.c_ubyte*33)()
    try:
        for start in range(0,len(pcm),160):
            lib.gsm_encode(enc,pcm[start:start+160].ctypes.data_as(C.POINTER(C.c_short)),packet)
            code=lib.gsm_decode(dec,packet,decoded[start:start+160].ctypes.data_as(C.POINTER(C.c_short)))
            if code!=0:raise RuntimeError(f'GSM decode failed: {code}')
    finally:lib.gsm_destroy(enc);lib.gsm_destroy(dec)
    return resample_poly(decoded[:count].astype(np.float32)/32768,sr//gcd,8000//gcd)[:len(x)]
