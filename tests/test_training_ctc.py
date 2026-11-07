# Exercise valid-target masking and training-step behavior.
# 检查有效标签掩码和训练步行为。
import copy

import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from MNovo.training_ctc import masked_ctc_loss, training_target_status, CTCLossResult
from MNovo.denovo.model import Spec2Pep


def test_loss_and_gradient_match_valid_subset():
    torch.manual_seed(1)
    logits = torch.randn(4, 3, 5, requires_grad=True)
    labels = torch.tensor([[1, 2, 0], [1, 1, 1], [2, 3, 4]])
    lengths = torch.tensor([2, 3, 3])
    result = masked_ctc_loss(logits, labels, lengths, blank=0)
    assert result.reasons == ['', 'unalignable_target', '']
    result.loss.backward()
    expected = logits.detach()[:, [0, 2]].clone().requires_grad_()
    loss = torch.nn.functional.ctc_loss(expected.log_softmax(-1), labels[[0, 2]],
                                       torch.tensor([4, 4]), lengths[[0, 2]], blank=0)
    loss.backward()
    torch.testing.assert_close(result.loss, loss)
    torch.testing.assert_close(logits.grad[:, [0, 2]], expected.grad)
    assert not logits.grad[:, 1].any()


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_sample_isolated(bad):
    logits = torch.randn(4, 2, 4)
    logits[:, 1, 1] = bad
    logits.requires_grad_()
    r = masked_ctc_loss(logits, torch.tensor([[1, 2], [2, 1]]), torch.tensor([2, 2]), 0)
    assert r.reasons == ['', 'nonfinite_logits']
    r.loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert not logits.grad[:, 1].any()


def test_empty_invalid_and_bf16():
    labels = torch.tensor([[1, 1], [0, 1], [7, 2]])
    r = masked_ctc_loss(torch.randn(2, 3, 4), labels, torch.tensor([2, 2, 2]), 0)
    assert r.loss is None
    assert not r.valid.any()
    logits = torch.randn(4, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    r = masked_ctc_loss(logits, torch.tensor([[1, 2]]), torch.tensor([2]), 0)
    assert r.loss.dtype == torch.float32
    r.loss.backward()
    assert torch.isfinite(logits.grad).all()


class Fixture(pl.LightningModule):
    optimizer_step = Spec2Pep.optimizer_step
    lr_scheduler_step = Spec2Pep.lr_scheduler_step
    on_after_backward = Spec2Pep.on_after_backward

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.))
        self._ctc_window_has_data = False
        self._ctc_last_update = False
        self.snapshots = []

    def training_step(self, batch, batch_idx):
        if not batch.item():
            return None
        self._ctc_window_has_data = True
        return self.weight.square()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.1, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.9)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.snapshots.append((self.weight.detach().clone(),
                              copy.deepcopy(self.trainer.optimizers[0].state_dict()),
                              self.lr_schedulers().last_epoch))


@pytest.mark.parametrize('values,accum', [([1, 0, 0], 1), ([1, 0, 0, 0], 2)])
def test_lightning_empty_update_preserves_adam_and_scheduler(values, accum, tmp_path):
    model = Fixture()
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, accumulate_grad_batches=accum,
                         default_root_dir=tmp_path)
    trainer.fit(model, DataLoader(values, batch_size=1))
    first = model.snapshots[accum - 1]
    last = model.snapshots[-1]
    torch.testing.assert_close(first[0], last[0], rtol=0, atol=0)
    assert first[2] == last[2] == 1
    for key, value in first[1]['state'][0].items():
        torch.testing.assert_close(value, last[1]['state'][0][key], rtol=0, atol=0)


def test_guards_and_accounting_restore():
    class Guard:
        _record_ctc_batch = Spec2Pep._record_ctc_batch
        on_save_checkpoint = Spec2Pep.on_save_checkpoint
        on_load_checkpoint = Spec2Pep.on_load_checkpoint
        log = lambda *args, **kwargs: None
        _ctc_valid_count = 0
        _ctc_global_total = 1
        _ctc_empty_batches = 0
        ctc_max_empty_batches = 2
        ctc_guard_min_samples = 100
        ctc_max_invalid_fraction = .05

    m = Guard()
    m._ctc_counts = {}
    result = CTCLossResult(None, torch.tensor([False]), ['unalignable_target'])
    batch = (None, None, ['AAAA'], ['test-spectrum'])
    m._record_ctc_batch('train', result, batch)
    state = {}
    m.on_save_checkpoint(state)
    restored = Guard()
    restored.on_load_checkpoint(state)
    with pytest.raises(RuntimeError, match='consecutive empty'):
        restored._record_ctc_batch('train', result, batch)
    # Validation only excludes from loss, not from the sample accounting.
    restored._record_ctc_batch('valid', result, batch)
    assert restored._ctc_counts['valid'] == {'seen': 1, 'excluded': 1}
    restored._ctc_empty_batches = 0
    restored.ctc_guard_min_samples = 1
    with pytest.raises(RuntimeError, match='excessive excluded'):
        restored._record_ctc_batch('train', result, batch)


def test_real_tokenizer_modified_and_repeated_targets(monkeypatch):
    import MNovo.denovo.model as module
    from MNovo.pmc_schema import PMC_RESIDUES
    monkeypatch.setattr(module, 'CTCBeamSearchDecoder', lambda *a: None)
    model = Spec2Pep(dim_model=16, n_head=2, dim_feedforward=32, n_layers=1,
                    max_length=40, residues=PMC_RESIDUES, PMC_enable=False)
    assert training_target_status('AC[+57.021]DM[+15.995]K', model.decoder,
                                  40, min_length=5) == ''
    assert training_target_status('A' * 30, model.decoder, 40) == 'unalignable_target'
    assert training_target_status('ACDK', model.decoder, 40) == 'outside_training_length'
    assert training_target_status('AC[+999]DMK', model.decoder, 40,
                                  min_length=5) == 'unsupported_training_token'


def test_nonfinite_gradient_stops_before_optimizer():
    m = Fixture()
    m.weight.grad = torch.tensor(float('nan'))
    with pytest.raises(FloatingPointError, match='gradients'):
        m.on_after_backward()


def test_warmup_constant_stops_increasing(monkeypatch):
    import MNovo.denovo.model as module
    from MNovo.pmc_schema import PMC_RESIDUES
    monkeypatch.setattr(module, 'CTCBeamSearchDecoder', lambda *a: None)
    model = Spec2Pep(dim_model=16, n_head=2, dim_feedforward=32, n_layers=1,
                    max_length=40, residues=PMC_RESIDUES, PMC_enable=False,
                    lr_schedule='warmup_constant', warmup_iters=4, lr=.001)
    optimizers, schedule = model.configure_optimizers()
    optimizer = optimizers[0]
    rates = []
    for _ in range(8):
        rates.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        schedule['scheduler'].step()
    assert rates[:4] == [.00025, .0005, .00075, .001]
    assert rates[4:] == [.001] * 4


@pytest.mark.parametrize('bracketed,plain', [
    ('[-17.027]QGVDGVYDSDPK', '-17.027QGVDGVYDSDPK'),
    ('[+42.011]AC[+57.021]DK', '+42.011AC+57.021DK'),
    ('[+43.006][-17.027]QAK', '+43.006-17.027QAK'),
])
def test_bracketed_terminal_modification_metric_equivalence(bracketed, plain):
    from MNovo.evaluation import tokens
    assert tokens(bracketed) == tokens(plain)
