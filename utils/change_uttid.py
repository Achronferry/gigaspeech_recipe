import sys,os
utt2utt = dict()
with open(sys.argv[1], 'r') as f:
    for line in f.readlines():
        new, old = line.strip('\n ').split(' ')
        utt2utt[old] = new

new_scp = dict()
with open(sys.argv[2], 'r') as f:
    for line in f.readlines():
        old, v = line.strip('\n ').split(' ')
        new = utt2utt[old]
        new_scp[new] = v
print(len(new_scp))
with open(sys.argv[2], 'w') as f:
    for k,v in new_scp.items():
        f.write(f"{k} {v}\n")
