# python utils/concat_ebd.py data/accent_cv ivector accent mix
import os.path as P
import sys
from tqdm import tqdm
import numpy as np
out_prefix = sys.argv[1]
in_scps = [out_prefix+ '/' + basename for basename in sys.argv[2:-1]]
out_basename = out_prefix + '/' + sys.argv[-1]
print("Write to", out_basename)
from kaldiio import ReadHelper, WriteHelper

in_readers = [ReadHelper(f'scp:{in_scp}.scp') for in_scp in in_scps]
with WriteHelper(f"ark,scp:{out_basename}.ark,{out_basename}.scp") as writer:
    for pairs in tqdm(zip(*in_readers)):
        key = pairs[0][0]
        # check key
        for pair in pairs:
            assert pair[0] == key, f"Key error:{key} {pair[0]}"
        
        # merge feat
        out_ebd = [pair[1] for pair in pairs]
        out_ebd = np.concatenate(out_ebd)
        writer(key, out_ebd)
















