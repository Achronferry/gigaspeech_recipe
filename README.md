# NOTES

- transducer_loss will print a lot logs, which is disgusting(verbose=False may fix). JIT is used here(not necessary???)
- colorlog is used here `colorlog==4.7.2`(old version), may be changed using --color False
- batch_per_gpu to suppose `batch_size in espnet` = `batch_size in config` * (`ngpu` * `num_nodes`)
- num_threads in asr.sh to support `num_workers`
- resume is set to true by default
- try/catch is used in batch cuda OOM
- rnn_decoder parameters is renamed
- encoder/decoder/joint is located in espnet_model
- warmup_epoch/warmup_ratio/warmup_steps are both used to customize
- logging.error to debug `show all info in all gpus`
- update average model
- remove wandb
- reference_file reference in inference
- default dump dir to exp/dump
- transducer_loss_type: str = "warp-transducer",
- dry_run type
- _use_new_zipfile_serialization=False

# HINTS/TODO

- remove pyc cache in repo!!!
- cmd.sh; path.sh; db.sh; conf/slurm.conf should be located in templated dir, and designed by user-self.
- lm rescore needed?
- CUDA memory is unstable:
  - normally 6200MB; sometimes 9800MB
- fps=1.5*1e6(max in 11G cuda mem)
- VALID time 10min; train time unknown
- streaming
- [W NNPACK.cpp:79] Could not initialize NNPACK! Reason: Unsupported hardware.

# Environment

- cuda10.2 pytorch1.4
- cuda11.1 pytorch1.10

# DEBUG

bash run.sh --stage 11 --dry_run
bash run.sh --stage 12 --inference_asr_model dry_run.pth

- find log (train/infer) : change to conf/{tuning,decode}/debug.yaml and output_dir exp/debug

# sed

```
grep -rl "espnet.nets" . | xargs sed -i 's/espnet\.nets/espnet2\.nets/g'
grep -rl "espnet.nets.pytorch_backend" . | xargs sed -i 's/espnet\.nets\.pytorch_backend/espnet2\.nets/g'
grep -rl "espnet2.nets.pytorch_backend" . | xargs sed -i 's/espnet2\.nets\.pytorch_backend/espnet2\.nets/g'
grep -rl "espnet2.asr.nets_utils" . | xargs sed -i 's/espnet2\.asr\.nets_utils/espnet2\.nets\.nets_utils/g'
```