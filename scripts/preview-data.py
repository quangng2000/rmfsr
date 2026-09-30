"""Export deterministic input/target examples for listening before a training run."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from rmfsr.data import SpeechPairs
from rmfsr.train import get_ffmpeg

parser=argparse.ArgumentParser()
parser.add_argument('--config',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--count',type=int,default=12)
a=parser.parse_args()
cfg=json.loads(Path(a.config).read_text())
output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
pairs=SpeechPairs(cfg['train_manifest'],seconds=cfg['seconds'],sr=cfg['sample_rate'],seed=20260930,
                  noise_dir=cfg.get('noise_dir'),ltas_path=cfg.get('ltas_path'),ffmpeg=get_ffmpeg(cfg.get('ffmpeg')),
                  pilot=cfg['pilot'],augmentation_profile=cfg.get('augmentation_profile','legacy'))
rows=[]
for i in range(a.count):
    target,damaged,mask,kinds=pairs.one()
    sf.write(output/f'{i:03d}-target.wav',target,cfg['sample_rate'],subtype='FLOAT')
    sf.write(output/f'{i:03d}-input.wav',damaged,cfg['sample_rate'],subtype='FLOAT')
    np.save(output/f'{i:03d}-missing-mask.npy',mask)
    rows.append(dict(index=i,seconds=len(target)/cfg['sample_rate'],degradations=kinds,
                     missing_samples=int(mask.sum()),finite=bool(np.isfinite(target).all() and np.isfinite(damaged).all())))
report=dict(pilot=cfg['pilot'],augmentation_profile=cfg.get('augmentation_profile','legacy'),
            uses_real_dns=bool(cfg.get('noise_dir')),uses_daps_eq=bool(cfg.get('ltas_path')),
            seed=20260930,examples=rows)
(output/'manifest.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
