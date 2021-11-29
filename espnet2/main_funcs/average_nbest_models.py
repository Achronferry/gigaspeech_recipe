import logging
from pathlib import Path
from typing import Sequence
from typing import Union
import warnings

import torch
from typeguard import check_argument_types
from typing import Collection

from espnet2.train.reporter import Reporter


@torch.no_grad()
def average_nbest_models(
    output_dir: Path,
    reporter: Reporter,
    best_model_criterion: Sequence[Sequence[str]],
    nbest: Union[Collection[int], int],
) -> None:
    """Generate averaged model from n-best models

    Args:
        output_dir: The directory contains the model file for each epoch
        reporter: Reporter instance
        best_model_criterion: Give criterions to decide the best model.
            e.g. [("valid", "loss", "min"), ("train", "acc", "max")]
        nbest:
    """
    assert check_argument_types()
    if isinstance(nbest, int):
        nbests = [nbest]
    else:
        nbests = list(nbest)
    if len(nbests) == 0:
        warnings.warn("At least 1 nbest values are required")
        nbests = [1]
    # 1. Get nbests: List[Tuple[str, str, List[Tuple[epoch, value]]]]
    nbest_epochs = [
        (ph, k, reporter.sort_epochs_and_values(ph, k, m)[: max(nbests)])
        for ph, k, m in best_model_criterion
        if reporter.has(ph, k)
    ]

    _loaded = {}
    for ph, cr, epoch_and_values in nbest_epochs:
        _nbests = [i for i in nbests if i <= len(epoch_and_values)]
        if len(_nbests) == 0:
            _nbests = [1]

        for n in _nbests:
            if n == 0:
                continue
            elif n == 1:
                # The averaged model is same as the best model
                e, _ = epoch_and_values[0]
                op = output_dir / f"{e}epoch.pth"
                sym_op = output_dir / f"{ph}.{cr}.ave_1best.pth"
                if sym_op.is_symlink() or sym_op.exists():
                    sym_op.unlink()
                sym_op.symlink_to(op.name)
            else:
                op = output_dir / f"{ph}.{cr}.ave_{n}best.pth"
                logging.info(
                    f"Averaging {n}best models: " f'criterion="{ph}.{cr}": {op}'
                )

                avg = None
                # 2.a. Averaging model
                for e, _ in epoch_and_values[:n]:
                    if e not in _loaded:
                        _loaded[e] = torch.load(
                            output_dir / f"{e}epoch.pth",
                            map_location="cpu",
                        )
                    states = _loaded[e]

                    if avg is None:
                        avg = states
                    else:
                        # Accumulated
                        for k in avg:
                            avg[k] = avg[k] + states[k]
                for k in avg:
                    if str(avg[k].dtype).startswith("torch.int"):
                        # For int type, not averaged, but only accumulated.
                        # e.g. BatchNorm.num_batches_tracked
                        # (If there are any cases that requires averaging
                        #  or the other reducing method, e.g. max/min, for integer type,
                        #  please report.)
                        pass
                    else:
                        avg[k] = avg[k] / n

                # 2.b. Save the ave model and create a symlink
                torch.save(avg, op)

        # 3. *.*.ave.pth is a symlink to the max ave model
        op = output_dir / f"{ph}.{cr}.ave_{max(_nbests)}best.pth"
        sym_op = output_dir / f"{ph}.{cr}.ave.pth"
        if sym_op.is_symlink() or sym_op.exists():
            sym_op.unlink()
        sym_op.symlink_to(op.name)

@torch.no_grad()
def save_model(model, optimizers, schedulers, scaler, reporter, iepoch, output_dir) -> None:
    """
    checkpoint.pth : resume from
    *epoch.pth: model.dict
    latest.pth: -> link to last *epoch.pth
    """
    # 4. Save/Update the checkpoint
    torch.save(
        {
            "model": model.state_dict(),
            "reporter": reporter.state_dict(),
            "optimizers": [o.state_dict() for o in optimizers],
            "schedulers": [
                s.state_dict() if s is not None else None
                for s in schedulers
            ],
            "scaler": scaler.state_dict() if scaler is not None else None,
        },
        output_dir / "checkpoint.pth",
        _use_new_zipfile_serialization=False
    )

    # 5. Save and log the model and update the link to the best model
    torch.save(model.state_dict(), output_dir / f"{iepoch}epoch.pth", _use_new_zipfile_serialization=False)

    # Creates a sym link latest.pth -> {iepoch}epoch.pth
    p = output_dir / "latest.pth"
    if p.is_symlink() or p.exists():
        p.unlink()
    p.symlink_to(f"{iepoch}epoch.pth")

@torch.no_grad()
def update_best(trainer_options, reporter, iepoch, output_dir):
    """trainer.py:369"""
    best_epoch = None
    _improved = []
    for _phase, k, _mode in trainer_options.best_model_criterion:
        # e.g. _phase, k, _mode = "train", "loss", "min"
        if reporter.has(_phase, k):
            best_epoch = reporter.get_best_epoch(_phase, k, _mode)
            # Creates sym links if it's the best result
            if best_epoch == iepoch:
                p = output_dir / f"{_phase}.{k}.best.pth"
                if p.is_symlink() or p.exists():
                    p.unlink()
                p.symlink_to(f"{iepoch}epoch.pth")
                _improved.append(f"{_phase}.{k}")
    
    return best_epoch, _improved
    
@torch.no_grad()
def remove_nbest(trainer_options, reporter, iepoch, output_dir, keep_nbest_models):
    """trainer.py Stage 6
    Remove the model files excluding n-best epoch and latest epoch
    """
    _removed = []
    # Get the union set of the n-best among multiple criterion
    nbests = set().union(
        *[
            set(reporter.sort_epochs(ph, k, m)[:keep_nbest_models])
            for ph, k, m in trainer_options.best_model_criterion
            if reporter.has(ph, k)
        ]
    )
    for e in range(1, iepoch):
        p = output_dir / f"{e}epoch.pth"
        if p.exists() and e not in nbests:
            p.unlink()
            _removed.append(str(p))
    
    return _removed
