# Fit the two score models using the declared training and validation inputs.
# 用预先指定的训练和验证输入拟合两套评分模型。
"""Fixed-role MLP fitting; no internal resplit and no calibration-label training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from MNovo.cct.cct_scoring import MLPClassifier, _pack_models

def train_fixed(train, validation, columns, output, seed=454, epochs=50, device="auto"):
    """Adopted MLP/AdamW/early-stop recipe, without resplitting roles."""
    if epochs < 1 or epochs > 50:
        raise ValueError("CCT fitting supports 1..50 epochs")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    if set(train.group_id) & set(validation.group_id):
        raise ValueError('Training and validation groups overlap')
    train_values = train[columns].to_numpy(np.float32)
    valid_values = validation[columns].to_numpy(np.float32)
    if min(len(train), len(validation)) < 2:
        raise ValueError('Insufficient train/validation rows')
    device = torch.device(device)
    branches = {}
    for offset, name in enumerate(('strict', 'relaxed')):
        y = train.denovo_label.to_numpy(np.float32, copy=True)
        v = validation.denovo_label.to_numpy(np.float32, copy=True)
        if name == 'relaxed':
            y *= train.is_decoy.eq(0).to_numpy(np.float32)
            v *= validation.is_decoy.eq(0).to_numpy(np.float32)
        if len(set(y)) != 2 or len(set(v)) != 2:
            raise ValueError(f'{name}: both classes required in fixed train and validation')
        scaler = StandardScaler().fit(train_values)
        x = torch.tensor(scaler.transform(train_values), dtype=torch.float32, device=device)
        xv = torch.tensor(scaler.transform(valid_values), dtype=torch.float32, device=device)
        yt = torch.tensor(y, dtype=torch.float32, device=device)
        torch.manual_seed(seed + offset)
        model = MLPClassifier(len(columns)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        generator = torch.Generator().manual_seed(seed + offset)
        best_auc, best_state, best_epoch, stale = -1., None, None, 0
        history = []
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, total = 0., 0
            for idx in torch.randperm(len(train), generator=generator).split(1024):
                if len(idx) < 2:
                    continue
                optimizer.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy(model(x[idx]), yt[idx])
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite CCT training loss')
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(idx)
                total += len(idx)
            model.eval()
            with torch.no_grad():
                probs = np.concatenate([model(b).cpu().numpy() for b in xv.split(4096)])
            auc = float(roc_auc_score(v, probs))
            history.append(dict(epoch=epoch, train_loss=total_loss/total, validation_auc=auc,
                                validation_accuracy=float(accuracy_score(v, probs > .5))))
            pd.DataFrame(history).to_csv(output / f'{name}_history.tsv', sep='\t', index=False)
            if auc > best_auc:
                best_auc, best_epoch, stale = auc, epoch, 0
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 8:
                    break
        model.load_state_dict(best_state)
        model.eval()
        metrics = dict(validation_auc=best_auc, best_epoch=best_epoch, epochs_ran=len(history),
                       train_rows=len(train), validation_rows=len(validation),
                       fixed_roles=True, internal_random_resplit=False)
        branches[name] = (model, scaler, metrics)
    packed = _pack_models(branches)
    torch.save(packed, output / 'score_models.pt')
    (output / 'metrics.json').write_text(json.dumps({k: v[2] for k, v in branches.items()}, indent=2))
    # Exact deployment scorer/float64 scaler path for all subsequent calibration.
    from MNovo.cct.cct_pipeline import _unpack
    return _unpack(packed, columns)
