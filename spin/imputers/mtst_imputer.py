from typing import Callable, List, Mapping, Optional, Type, Union

import torch
from icecream import ic
from torchmetrics import Metric
from tsl.imputers import Imputer
from tsl.predictors import Predictor


class MTSTImputer(Imputer):
    def __init__(
        self,
        model_class: Type,
        model_kwargs: Mapping,
        optim_class: Type,
        optim_kwargs: Mapping,
        loss_fn: Callable,
        scale_target: bool = True,
        whiten_prob: Union[float, List[float]] = 0.2,
        prediction_loss_weight: float = 1.0,
        metrics: Optional[Mapping[str, Metric]] = None,
        scheduler_class: Optional = None,
        scheduler_kwargs: Optional[Mapping] = None,
        node_index: int = 0,
        scaler=None,
    ):
        super().__init__(
            model_class=model_class,
            model_kwargs=model_kwargs,
            optim_class=optim_class,
            optim_kwargs=optim_kwargs,
            loss_fn=loss_fn,
            scale_target=scale_target,
            whiten_prob=whiten_prob,
            prediction_loss_weight=prediction_loss_weight,
            metrics=metrics,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
        )
        ic(model_kwargs)
        self.node_index = node_index
        self.scaler = scaler

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return super().on_after_batch_transfer(batch, dataloader_idx)

    def training_step(self, batch, batch_idx):
        batch.y = self.scaler.transform(batch.y.cpu()).cuda()
        injected_missing = batch.original_mask - batch.mask
        if "target_nodes" in batch:
            injected_missing = injected_missing[..., batch.target_nodes, :]
        # batch.input.target_mask = injected_missing
        injected_missing = injected_missing.squeeze(-1)[..., self.node_index]
        y_hat, y, loss = self.shared_step(batch, mask=injected_missing)

        # Logging
        self.train_metrics.update(y_hat, y, injected_missing)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss("train", loss, batch_size=batch.batch_size)
        if "target_nodes" in batch:
            torch.cuda.empty_cache()
        return loss

    def shared_step(self, batch, mask):
        y = y_loss = batch.y.squeeze(-1)[..., self.node_index]
        y_hat = y_hat_loss = self.predict_batch(
            batch, preprocess=False, postprocess=not self.scale_target
        )

        if self.scale_target:
            y_loss = batch.transform["y"].transform(y)
            y_hat = batch.transform["y"].inverse_transform(y_hat)

        y_hat_loss, y_loss, mask = self.trim_warm_up(y_hat_loss, y_loss, mask)

        if isinstance(y_hat_loss, (list, tuple)):
            imputation, predictions = y_hat_loss
            y_hat = y_hat[0]
        else:
            imputation, predictions = y_hat_loss, []
        loss = self.loss_fn(imputation, y_loss, mask)
        for pred in predictions:
            pred_loss = self.loss_fn(pred, y_loss, mask)
            loss += self.prediction_loss_weight * pred_loss

        return y_hat.detach(), y, loss

    def validation_step(self, batch, batch_idx):
        batch.y = self.scaler.transform(batch.y.cpu()).cuda()
        # batch.input.target_mask = batch.eval_mask
        mask = batch.eval_mask.squeeze(-1)[..., self.node_index]
        y_hat, y, val_loss = self.shared_step(batch, mask)
        self.val_metrics.update(y_hat, y, mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss("val", val_loss, batch_size=batch.batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):
        batch.y = self.scaler.transform(batch.y.cpu()).cuda()
        # batch.input.target_mask = batch.eval_mask
        # Compute outputs and rescale
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)

        if isinstance(y_hat, (list, tuple)):
            y_hat = y_hat[0]

        # y, eval_mask = batch.y, batch.eval_mask
        y = batch.y.squeeze(-1)[..., self.node_index]
        mask = batch.eval_mask.squeeze(-1)[..., self.node_index]

        test_loss = self.loss_fn(y_hat, y, mask)

        # Logging
        self.test_metrics.update(y_hat.detach(), y, mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss("test", test_loss, batch_size=batch.batch_size)
        return test_loss

    @staticmethod
    def add_argparse_args(parser, **kwargs):
        parser.add_argument("--scale-target", type=bool, default=False)
        parser.add_argument("--whiten-prob", type=float, default=0.05)
        parser.add_argument("--prediction-loss-weight", type=float, default=1.0)
        parser.add_argument("--node-index", type=int, default=0)

        return parser
