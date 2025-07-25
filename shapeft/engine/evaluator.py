import logging
import os
import time
from pathlib import Path
import math
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np


class Evaluator:
    """
    Evaluator class for evaluating the models.
    Attributes:
        val_loader (DataLoader): DataLoader for the validation dataset.
        exp_dir (str | Path): Directory for experiment outputs.
        device (torch.device): Device to run the evaluation on (e.g., CPU or GPU).
        use_wandb (bool): Flag to indicate if Weights and Biases (wandb) is used for logging.
        logger (logging.Logger): Logger for logging information.
        classes (list): List of class names in the dataset.
        split (str): Dataset split (e.g., 'train', 'val', 'test').
        ignore_index (int): Index to ignore in the dataset.
        num_classes (int): Number of classes in the dataset.
        max_name_len (int): Maximum length of class names.
        wandb (module): Weights and Biases module for logging (if use_wandb is True).
    Methods:
        __init__(val_loader: DataLoader, exp_dir: str | Path, device: torch.device, use_wandb: bool) -> None:
            Initializes the Evaluator with the given parameters.
        evaluate(model: torch.nn.Module, model_name: str, model_ckpt_path: str | Path | None = None) -> None:
            Evaluates the given model. This method should be implemented by subclasses.
        __call__(model: torch.nn.Module) -> None:
            Calls the evaluator on the given model.
        compute_metrics() -> None:
            Computes evaluation metrics. This method should be implemented by subclasses.
        log_metrics(metrics: dict) -> None:
            Logs the computed metrics. This method should be implemented by subclasses.
    """

    def __init__(
            self,
            val_loader: DataLoader,
            exp_dir: str | Path,
            device: torch.device,
            inference_mode: str = 'sliding',
            sliding_inference_batch: int = None,
            use_wandb: bool = False,
            dataset_name: str = 'sen1floods11'
    ) -> None:
        self.rank = int(os.environ["RANK"])
        self.val_loader = val_loader
        self.logger = logging.getLogger()
        self.exp_dir = exp_dir
        self.device = device
        self.inference_mode = inference_mode
        self.sliding_inference_batch = sliding_inference_batch
        self.classes = self.val_loader.dataset.classes
        self.split = self.val_loader.dataset.split
        self.ignore_index = self.val_loader.dataset.ignore_index
        self.num_classes = len(self.classes)
        self.max_name_len = max([len(name) for name in self.classes])
        self.dataset_name = dataset_name

        self.use_wandb = use_wandb

    def evaluate(
            self,
            model: torch.nn.Module,
            model_name: str,
            model_ckpt_path: str | Path | None = None,
    ) -> None:
        raise NotImplementedError

    def __call__(self, model):
        pass

    def compute_metrics(self):
        pass

    def log_metrics(self, metrics):
        pass

    @staticmethod
    def sliding_inference(model, img, input_size, output_shape=None, stride=None, max_batch=None):
        b, c, t, height, width = img[list(img.keys())[0]].shape

        if stride is None:
            h = int(math.ceil(height / input_size))
            w = int(math.ceil(width / input_size))
        else:
            h = math.ceil((height - input_size) / stride) + 1
            w = math.ceil((width - input_size) / stride) + 1

        h_grid = torch.linspace(0, height - input_size, h).round().long()
        w_grid = torch.linspace(0, width - input_size, w).round().long()
        num_crops_per_img = h * w

        for k, v in img.items():
            img_crops = []
            for i in range(h):
                for j in range(w):
                    img_crops.append(v[:, :, :, h_grid[i]:h_grid[i] + input_size, w_grid[j]:w_grid[j] + input_size])
            img[k] = torch.cat(img_crops, dim=0)

        pred = []
        max_batch = max_batch if max_batch is not None else b * num_crops_per_img
        batch_num = int(math.ceil(b * num_crops_per_img / max_batch))
        for i in range(batch_num):
            img_ = {k: v[max_batch * i: min(max_batch * i + max_batch, b * num_crops_per_img)] for k, v in img.items()}
            pred_ = model.forward(img_, output_shape=(input_size, input_size))
            pred.append(pred_)
        pred = torch.cat(pred, dim=0)
        pred = pred.view(num_crops_per_img, b, -1, input_size, input_size).transpose(0, 1)

        merged_pred = torch.zeros((b, pred.shape[2], height, width), device=pred.device)
        pred_count = torch.zeros((b, height, width), dtype=torch.long, device=pred.device)
        for i in range(h):
            for j in range(w):
                merged_pred[:, :, h_grid[i]:h_grid[i] + input_size,
                w_grid[j]:w_grid[j] + input_size] += pred[:, h * i + j]
                pred_count[:, h_grid[i]:h_grid[i] + input_size,
                w_grid[j]:w_grid[j] + input_size] += 1

        merged_pred = merged_pred / pred_count.unsqueeze(1)
        if output_shape is not None:
            merged_pred = F.interpolate(merged_pred, size=output_shape, mode="bilinear")

        return merged_pred

class SegEvaluator(Evaluator):
    """
    SegEvaluator is a class for evaluating segmentation models. It extends the Evaluator class and provides methods
    to evaluate a model, compute metrics, and log the results.
    Attributes:
        val_loader (DataLoader): DataLoader for the validation dataset.
        exp_dir (str | Path): Directory for saving experiment results.
        device (torch.device): Device to run the evaluation on.
        use_wandb (bool): Flag to indicate whether to use Weights and Biases for logging.
    Methods:
        evaluate(model, model_name='model', model_ckpt_path=None):
            Evaluates the given model on the validation dataset and computes metrics.
        __call__(model, model_name, model_ckpt_path=None):
            Calls the evaluate method. This allows the object to be used as a function.
        compute_metrics(confusion_matrix):
            Computes various metrics such as IoU, precision, recall, F1-score, mean IoU, mean F1-score, and mean accuracy
            from the given confusion matrix.
        log_metrics(metrics):
            Logs the computed metrics. If use_wandb is True, logs the metrics to Weights and Biases.
    """

    def __init__(
            self,
            val_loader: DataLoader,
            exp_dir: str | Path,
            device: torch.device,
            inference_mode: str = 'sliding',
            sliding_inference_batch: int = None,
            use_wandb: bool = False,
            dataset_name: str = 'PASTIS-HD'
    ):
        super().__init__(val_loader, exp_dir, device, inference_mode, sliding_inference_batch, use_wandb, dataset_name)

    def reshape_transform(self, tensor, height=15, width=15):
        # Reshape (batch, seq_len, embed_dim) -> (batch, embed_dim, height, width)
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    @torch.no_grad()
    def evaluate(self, model, model_name='model', model_ckpt_path=None):
        t = time.time()

        if model_ckpt_path is not None:
            model_dict = torch.load(model_ckpt_path, map_location=self.device, weights_only=False)
            model_name = os.path.basename(model_ckpt_path).split(".")[0]
            if "model" in model_dict:
                model.module.load_state_dict(model_dict["model"])
            else:
                model.module.load_state_dict(model_dict)

            self.logger.info(f"Loaded {model_name} for evaluation")
        model.eval()

        tag = f"Evaluating {model_name} on {self.split} set"
        confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), device=self.device
        )

        for batch_idx, data in enumerate(tqdm(self.val_loader, desc=tag)):
            image, target = data["image"], data["target"]
            image = {k: v.to(self.device) for k, v in image.items()}
            target = target.to(self.device)

            if self.inference_mode == "sliding":
                input_size = model.module.encoder.input_size
                if model.module.encoder.model_name != "utae_encoder":
                    logits = self.sliding_inference(model, image, input_size, output_shape=target.shape[-2:],
                                                    max_batch=self.sliding_inference_batch)
                else: 
                    logits = model(image, batch_positions=data["metadata"])
            elif self.inference_mode == "whole":
                if model.module.encoder.model_name != "utae_encoder":
                    logits = model(image, output_shape=target.shape[-2:])
                else: 
                    logits = model(image, batch_positions=data["metadata"])                    
            else:
                raise NotImplementedError((f"Inference mode {self.inference_mode} is not implemented."))
            if logits.shape[1] == 1:
                pred = (torch.sigmoid(logits) > 0.5).type(torch.int64).squeeze(dim=1)
            else:
                pred = torch.argmax(logits, dim=1)

            valid_mask = target != self.ignore_index
            pred, target = pred[valid_mask], target[valid_mask]

            count = torch.bincount(
                (pred * self.num_classes + target), minlength=self.num_classes ** 2
            )
            confusion_matrix += count.view(self.num_classes, self.num_classes)

        torch.distributed.all_reduce(
            confusion_matrix, op=torch.distributed.ReduceOp.SUM
        )
        print(confusion_matrix.cpu())
        metrics = self.compute_metrics(confusion_matrix.cpu())
        self.log_metrics(metrics)

        used_time = time.time() - t

        return metrics, used_time

    @torch.no_grad()
    def __call__(self, model, model_name, model_ckpt_path=None):
        return self.evaluate(model, model_name, model_ckpt_path)

    def compute_metrics(self, confusion_matrix):
        # Calculate IoU for each class
        intersection = torch.diag(confusion_matrix)
        union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersection
        iou = (intersection / (union + 1e-6)) * 100

        # Calculate precision and recall for each class
        precision = intersection / (confusion_matrix.sum(dim=0) + 1e-6) * 100
        recall = intersection / (confusion_matrix.sum(dim=1) + 1e-6) * 100

        # Calculate F1-score for each class
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        # Calculate mean IoU, mean F1-score, and mean Accuracy
        miou = iou.mean().item()
        mf1 = f1.mean().item()
        macc = (intersection.sum() / (confusion_matrix.sum() + 1e-6)).item() * 100

        # Convert metrics to CPU and to Python scalars
        iou = iou.cpu()
        f1 = f1.cpu()
        precision = precision.cpu()
        recall = recall.cpu()

        # Prepare the metrics dictionary
        metrics = {
            "IoU": [iou[i].item() for i in range(self.num_classes)],
            "mIoU": miou,
            "F1": [f1[i].item() for i in range(self.num_classes)],
            "mF1": mf1,
            "mAcc": macc,
            "Precision": [precision[i].item() for i in range(self.num_classes)],
            "Recall": [recall[i].item() for i in range(self.num_classes)],
        }

        return metrics

    def log_metrics(self, metrics):
        def format_metric(name, values, mean_value):
            header = f"------- {name} --------\n"
            metric_str = (
                    "\n".join(
                        c.ljust(self.max_name_len, " ") + "\t{:>7}".format("%.3f" % num)
                        for c, num in zip(self.classes, values)
                    )
                    + "\n"
            )
            mean_str = (
                    "-------------------\n"
                    + "Mean".ljust(self.max_name_len, " ")
                    + "\t{:>7}".format("%.3f" % mean_value)
            )
            return header + metric_str + mean_str

        iou_str = format_metric("IoU", metrics["IoU"], metrics["mIoU"])
        f1_str = format_metric("F1-score", metrics["F1"], metrics["mF1"])

        precision_mean = torch.tensor(metrics["Precision"]).mean().item()
        recall_mean = torch.tensor(metrics["Recall"]).mean().item()

        precision_str = format_metric("Precision", metrics["Precision"], precision_mean)
        recall_str = format_metric("Recall", metrics["Recall"], recall_mean)

        macc_str = f"Mean Accuracy: {metrics['mAcc']:.3f} \n"

        self.logger.info(iou_str)
        self.logger.info(f1_str)
        self.logger.info(precision_str)
        self.logger.info(recall_str)
        self.logger.info(macc_str)

        if self.use_wandb and self.rank == 0:
            wandb.log(
                {
                    f"{self.split}_mIoU": metrics["mIoU"],
                    f"{self.split}_mF1": metrics["mF1"],
                    f"{self.split}_mAcc": metrics["mAcc"],
                    **{
                        f"{self.split}_IoU_{c}": v
                        for c, v in zip(self.classes, metrics["IoU"])
                    },
                    **{
                        f"{self.split}_F1_{c}": v
                        for c, v in zip(self.classes, metrics["F1"])
                    },
                    **{
                        f"{self.split}_Precision_{c}": v
                        for c, v in zip(self.classes, metrics["Precision"])
                    },
                    **{
                        f"{self.split}_Recall_{c}": v
                        for c, v in zip(self.classes, metrics["Recall"])
                    },
                }
            )



class RegEvaluator(Evaluator):
    def __init__(
        self,
        val_loader,
        exp_dir,
        device,
        inference_mode="sliding",
        sliding_inference_batch=None,
        use_wandb=False,
        dataset_name="regression_dataset"
    ):
        super().__init__(val_loader, exp_dir, device, inference_mode, sliding_inference_batch, use_wandb, dataset_name)

    @torch.no_grad()
    def evaluate(self, model, model_name="model", model_ckpt_path=None):
        start_time = time.time()

        if model_ckpt_path is not None:
            model_dict = torch.load(model_ckpt_path, map_location=self.device, weights_only=False)
            model_name = os.path.basename(model_ckpt_path).split(".")[0]
            if "model" in model_dict:
                model.module.load_state_dict(model_dict["model"])
            else:
                model.module.load_state_dict(model_dict)

            self.logger.info(f"Loaded {model_name} for evaluation")
        model.eval()

        y_true_all = []
        y_pred_all = []

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            # image, target = batch["image"], batch["target"]  # target: (B, 3, 256, 256)
            image, target = batch["image"], batch["target"]

            metadata = batch["metadata"]  # metadata: (B, 1) with dates
            # quitar mask de image

            image = {k: v.to(self.device) for k, v in image.items()}
            target = target.to(self.device)
            

            if self.inference_mode == "sliding":
                input_size = model.module.encoder.input_size
                logits = model(image,  batch_positions=metadata)
            elif self.inference_mode == "whole":
                logits = model(image,  batch_positions=metadata)
            else:
                raise NotImplementedError(f"Inference mode {self.inference_mode} not implemented.")

            y_true_all.append(target)
            y_pred_all.append(logits)

        y_true_all = torch.cat(y_true_all, dim=0)  # (N, 3, H, W)
        y_pred_all = torch.cat(y_pred_all, dim=0)

        # Calcular MAE y MSE por canal
        mae_channels = []
        mse_channels = []
        for c in range(y_true_all.shape[1]):
            diff = y_pred_all[:, c] - y_true_all[:, c]
            mae = torch.mean(torch.abs(diff)).item()
            mse = torch.mean(diff ** 2).item()
            mae_channels.append(mae)
            mse_channels.append(mse)
# ANNOTATIONS_combustible_disponible", f"{name}.npy"))
#             target_poder_calorico = np.load(os.path.join(self.root_path, f"ANNOTATIONS_poder_calorico", f"{name}.npy"))
#             target_resistencia_control = np.load(os.path.join(self.root_path, f"ANNOTATIONS_resistencia_control", f"{name}.npy"))
        metrics = {
            "mae_combustible_disponible": mae_channels[0],
            "mae_poder_calorico": mae_channels[1],
            "mae_resistencia_control": mae_channels[2],
            "mae_velocidad_propagacion": mae_channels[3],
            "mse_combustible_disponible": mse_channels[0],
            "mse_poder_calorico": mse_channels[1],
            "mse_resistencia_control": mse_channels[2],
            "mse_velocidad_propagacion": mse_channels[3],
            "mae_mean": sum(mae_channels) / 4,
            "mse_mean": sum(mse_channels) / 4,
        }

        self.log_metrics(metrics)
        return metrics, time.time() - start_time
    @torch.no_grad()
    def __call__(self, model, model_name=None, model_ckpt_path=None):
        if model_name is None:
            model_name = "model"
        return self.evaluate(model, model_name, model_ckpt_path)

    def log_metrics(self, metrics):
        self.logger.info(f"MAE Combustible Disponible: {metrics['mae_combustible_disponible']:.6f}")
        self.logger.info(f"MAE Poder Calorico: {metrics['mae_poder_calorico']:.6f}")
        self.logger.info(f"MAE Resistencia Control: {metrics['mae_resistencia_control']:.6f}")
        self.logger.info(f"MAE Velocidad Propagacion: {metrics['mae_velocidad_propagacion']:.6f}")
        self.logger.info(f"MSE Combustible Disponible: {metrics['mse_combustible_disponible']:.6f}")
        self.logger.info(f"MSE Poder Calorico: {metrics['mse_poder_calorico']:.6f}")
        self.logger.info(f"MSE Resistencia Control: {metrics['mse_resistencia_control']:.6f}")
        self.logger.info(f"MSE Velocidad Propagacion: {metrics['mse_velocidad_propagacion']:.6f}")
        self.logger.info(f"MAE Medio:   {metrics['mae_mean']:.6f}")
        self.logger.info(f"MSE Medio:   {metrics['mse_mean']:.6f}")

        if self.use_wandb and self.rank == 0:
            wandb.log({
                f"{self.split}mae_combustible_disponible": metrics["mae_combustible_disponible"],
                f"{self.split}mae_poder_calorico": metrics["mae_poder_calorico"],
                f"{self.split}mae_resistencia_control": metrics["mae_resistencia_control"],
                f"{self.split}mae_velocidad_propagacion": metrics["mae_velocidad_propagacion"],
                f"{self.split}mse_combustible_disponible": metrics["mse_combustible_disponible"],
                f"{self.split}mse_poder_calorico": metrics["mse_poder_calorico"],
                f"{self.split}mse_resistencia_control": metrics["mse_resistencia_control"],
                f"{self.split}mse_velocidad_propagacion": metrics["mse_velocidad_propagacion"],
                f"{self.split}_MAE_mean": metrics["mae_mean"],
                f"{self.split}_MSE_mean": metrics["mse_mean"],
            })
