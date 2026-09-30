"""Compute DAPS produced-speech LTAS only from the explicitly supplied folder."""
import argparse,hashlib,json,pathlib
import numpy as np
from scipy import signal
from .data import read_audio
from .degradations import level

def make_ltas(directory,output,sr=16000):
    paths=sorted(p for p in pathlib.Path(directory).rglob('*') if p.suffix.lower() in ('.wav','.flac'))
    if not paths:raise ValueError('No produced DAPS WAVs found')
    power=np.zeros(161,np.float64);frames=0;provenance=[]
    for path in paths:
        wave=level(read_audio(path,sr),-25)
        _,_,z=signal.stft(wave,sr,nperseg=320,noverlap=160)
        power+=(abs(z)**2).sum(axis=1);frames+=z.shape[1]
        provenance.append({'path':str(path.relative_to(directory)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    np.save(output,power/frames+1e-10)
    pathlib.Path(str(output)+'.json').write_text(json.dumps({'source':'DAPS produced (user-specified folder)',
        'sample_rate':sr,'files':provenance,'normalization_dbfs':-25},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--produced-dir',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();make_ltas(a.produced_dir,a.output)
