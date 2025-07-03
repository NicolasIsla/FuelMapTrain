import hashlib
import os as os
os.environ["HYDRA_FULL_ERROR"] = "1"
import pathlib
import pprint
import time

import hydra
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from shapeft.datasets.base import GeoFMDataset, GeoFMSubset, RawGeoFMDataset
from shapeft.decoders.base import Decoder
from shapeft.encoders.base import Encoder
from shapeft.engine.evaluator import Evaluator
from shapeft.engine.trainer import Trainer
from shapeft.utils.collate_fn import get_collate_fn
from shapeft.utils.logger import init_logger
from shapeft.utils.subset_sampler import get_subset_indices
from shapeft.utils.utils import (
    fix_seed,
    get_best_model_ckpt_path,
    get_final_model_ckpt_path,
    get_generator,
    seed_worker,
)

print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

def get_shared_exp_info(hydra_config: HydraConf, is_distributed=False, rank=0) -> dict[str, str]:
    choices = OmegaConf.to_container(hydra_config.runtime.choices)
    cfg_hash = hashlib.sha1(
        OmegaConf.to_yaml(hydra_config).encode(), usedforsecurity=False
    ).hexdigest()[:6]
    
    if is_distributed:
        if rank == 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        else:
            timestamp = None
        # Broadcast timestamp from rank 0
        timestamp = broadcast_string(timestamp, src=0)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    fm = choices["encoder"]
    decoder = choices["decoder"]
    ds = choices["dataset"]
    task = choices["task"]
    exp_name = f"{timestamp}_{cfg_hash}_{fm}_{decoder}_{ds}"

    return {
        "timestamp": timestamp,
        "fm": fm,
        "decoder": decoder,
        "ds": ds,
        "task": task,
        "exp_name": exp_name,
    }

def broadcast_string(value: str, src: int = 0):
    """Broadcast a string from src rank to all other ranks."""
    # Convert string to bytes and to tensor
    if value is not None:
        encoded = value.encode('utf-8')
        length = torch.tensor([len(encoded)], dtype=torch.long, device='cuda')
        data = torch.tensor(list(encoded), dtype=torch.uint8, device='cuda')
    else:
        length = torch.tensor([0], dtype=torch.long, device='cuda')
        data = torch.tensor([], dtype=torch.uint8, device='cuda')

    # Broadcast length
    torch.distributed.broadcast(length, src)
    # Allocate tensor on other ranks
    if value is None:
        data = torch.empty(length.item(), dtype=torch.uint8, device='cuda')
    # Broadcast actual data
    torch.distributed.broadcast(data, src)
    return bytes(data.tolist()).decode('utf-8')


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Geofm-bench main function.

    Args:
        cfg (DictConfig): main_config
    """
    # fix all random seeds
    fix_seed(cfg.seed)
    # distributed training variables
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        is_distributed = True
    else:
        rank = int(os.environ["RANK"])
        is_distributed = False
    # true if training else false
    train_run = cfg.train
    if train_run:
        exp_info = get_shared_exp_info(HydraConfig.get(), is_distributed, rank)
        exp_name = exp_info["exp_name"]
        task_name = exp_info["task"]
        exp_dir = pathlib.Path(cfg.work_dir) / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        logger_path = exp_dir / "train.log"
        config_log_dir = exp_dir / "configs"
        config_log_dir.mkdir(exist_ok=True)
        # init wandb
        if cfg.task.trainer.use_wandb and rank == 0:
            import wandb

            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                project=cfg.wandb_project,
                name=exp_name,
                config=wandb_cfg,
                tags=[f"mt{cfg.dataset.multi_temporal}", "ft" if cfg.finetune else "no-ft", str(int(cfg.limited_label_train*100))],
            )
            cfg["wandb_run_id"] = wandb.run.id
        OmegaConf.save(cfg, config_log_dir / "config.yaml")

    else:
        exp_dir = pathlib.Path(cfg.ckpt_dir)
        exp_name = exp_dir.name
        logger_path = exp_dir / "test.log"
        # load training config
        cfg_path = exp_dir / "configs" / "config.yaml"
        inference_mode_current = cfg.task.evaluator.inference_mode
        cfg = OmegaConf.load(cfg_path)
        cfg.task.evaluator.inference_mode = inference_mode_current
        if cfg.task.trainer.use_wandb and rank == 0:
            import wandb

            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                project=cfg.wandb_project,
                name=exp_name,
                config=wandb_cfg,
                id=cfg.get("wandb_run_id"),
                resume="allow",
            )

    logger = init_logger(logger_path, rank=rank)
    logger.info("============ Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")

    encoder: Encoder = instantiate(cfg.encoder)
    if train_run:
        if cfg.from_scratch and cfg.finetune:
            encoder.load_encoder_weights(logger, from_scratch=True)
            logger.info(f"Built {encoder.model_name} from scratch.")
        else:
            encoder.load_encoder_weights(logger)
            logger.info(f"Built {encoder.model_name} from checkpoint.")
    else:
        encoder.load_encoder_weights(logger)
        logger.info(f"Built {encoder.model_name} from checkpoint.")
    
    # prepare the decoder (segmentation/regression)
    decoder: Decoder = instantiate(
            cfg.decoder,
            encoder=encoder,
        )
    decoder.to(device)
    decoder = torch.nn.parallel.DistributedDataParallel(
            decoder,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=cfg.finetune,
        )

    logger.info(
            "Built {} for with {} encoder.".format(
                decoder.module.model_name, type(encoder).__name__
            )
        )

    def non_encoder_params(module):
        for name, param in module.named_parameters():
            if not name.startswith("encoder"):
                yield param

    modalities = list(encoder.input_bands.keys())
    collate_fn = get_collate_fn(modalities)

    # training
    if train_run:
        # get preprocessor
        train_preprocessor = instantiate(
            cfg.preprocessing.train,
            dataset_cfg=cfg.dataset,
            encoder_cfg=cfg.encoder,
            _recursive_=False,
        )
        val_preprocessor = instantiate(
            cfg.preprocessing.val,
            dataset_cfg=cfg.dataset,
            encoder_cfg=cfg.encoder,
            _recursive_=False,
        )

        # get datasets
        raw_train_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="train")
        raw_val_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="val")

        is_main_process = not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

        if 0 < cfg.limited_label_train < 1:
            indices_dir = pathlib.Path(cfg.work_dir) / str(cfg.dataset["dataset_name"])
            indices_file = indices_dir / f"train_{int(cfg.limited_label_train*100)}.pt"
            
            if is_main_process:
                if not indices_file.exists():
                    indices_dir.mkdir(parents=True, exist_ok=True)
                    indices = get_subset_indices(
                        raw_train_dataset,
                        task=task_name,
                        strategy=cfg.limited_label_strategy,
                        label_fraction=cfg.limited_label_train,
                        num_bins=cfg.stratification_bins,
                        logger=logger,
                    )
                    torch.save(indices, indices_file)

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

            indices = torch.load(indices_file, weights_only=False)

            raw_train_dataset = GeoFMSubset(raw_train_dataset, indices)

        if 0 < cfg.limited_label_val < 1:
            indices = get_subset_indices(
                raw_val_dataset,
                task=task_name,
                strategy=cfg.limited_label_strategy,
                label_fraction=cfg.limited_label_val,
                num_bins=cfg.stratification_bins,
                logger=logger,
            )
            raw_val_dataset = GeoFMSubset(raw_val_dataset, indices)

        train_dataset = GeoFMDataset(
            raw_train_dataset, train_preprocessor, cfg.data_replicate
        )
        val_dataset = GeoFMDataset(
            raw_val_dataset, val_preprocessor, cfg.data_replicate
        )

        logger.info("Built {} dataset.".format(cfg.dataset.dataset_name))

        logger.info(
            f"Total number of train patches: {len(train_dataset)}\n"
            f"Total number of validation patches: {len(val_dataset)}\n"
        )

        # get train val data loaders
        train_loader = DataLoader(
            train_dataset,
            sampler=DistributedSampler(train_dataset),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
            # persistent_workers=True causes memory leak
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=get_generator(cfg.seed),
            drop_last=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            sampler=DistributedSampler(val_dataset),
            batch_size=cfg.test_batch_size,
            num_workers=cfg.test_num_workers,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            # generator=g,
            drop_last=False,
            collate_fn=collate_fn,
        )

        criterion = instantiate(cfg.criterion)

        params = [{'params': non_encoder_params(decoder.module), 'lr': cfg.optimizer.lr}]
        if cfg.finetune:
            params.append({'params': decoder.module.encoder.parameters(), 'lr': cfg.optimizer.lr * cfg.ft_rate})

        optimizer = instantiate(cfg.optimizer, params=None)
        optimizer = optimizer(params=params)

        lr_scheduler = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            total_iters=len(train_loader) * cfg.task.trainer.n_epochs,
        )
        
        val_evaluator: Evaluator = instantiate(
                cfg.task.evaluator, val_loader=val_loader, exp_dir=exp_dir, device=device,
                inference_mode=cfg.task.evaluator.inference_mode,
                dataset_name=cfg.dataset.dataset_name
            )

        trainer: Trainer = instantiate(
                cfg.task.trainer,
                model=decoder,
                train_loader=train_loader,
                lr_scheduler=lr_scheduler,
                optimizer=optimizer,
                criterion=criterion,
                evaluator=val_evaluator,
                exp_dir=exp_dir,
                device=device,
            )
        # resume training if model_checkpoint is provided
        if cfg.ckpt_dir is not None:
            trainer.load_model(cfg.ckpt_dir)

        trainer.train()

    # Evaluation
    test_preprocessor = instantiate(
        cfg.preprocessing.test,
         dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )
    # get datasets
    raw_test_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="test")
    test_dataset = GeoFMDataset(raw_test_dataset, test_preprocessor)

    test_loader = DataLoader(
            test_dataset,
            sampler=DistributedSampler(test_dataset),
            batch_size=cfg.test_batch_size,
            num_workers=cfg.test_num_workers,
            pin_memory=True,
            persistent_workers=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
    test_evaluator: Evaluator = instantiate(
            cfg.task.evaluator, val_loader=test_loader, exp_dir=exp_dir, device=device,
            inference_mode= cfg.task.evaluator.inference_mode, #'sliding' if train_run else 'whole',
            dataset_name=cfg.dataset.dataset_name
        )

    if cfg.use_final_ckpt:
        model_ckpt_path = get_final_model_ckpt_path(exp_dir)
    else:
        model_ckpt_path = get_best_model_ckpt_path(exp_dir)
    test_evaluator.evaluate(decoder, "test_model", model_ckpt_path)

    if cfg.use_wandb and rank == 0:
        wandb.finish()

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
