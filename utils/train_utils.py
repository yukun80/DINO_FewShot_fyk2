from .losses import BootstrappedCE
from .lr_schedulers import poly_lr_scheduler,cosine_lr_scheduler,step_lr_scheduler,exp_lr_scheduler
import torch

def get_lr_function(config,total_iterations):
    # get the learning rate multiplier function for LambdaLR
    name=config["lr_scheduler"]
    warmup_iters=config["warmup_iters"]
    warmup_factor=config["warmup_factor"]
    if "poly"==name:
        p=config["poly_power"]
        return lambda x : poly_lr_scheduler(x,total_iterations,warmup_iters,warmup_factor,p)
    elif "cosine"==name:
        return lambda x : cosine_lr_scheduler(x,total_iterations,warmup_iters,warmup_factor)
    elif "step"==name:
        return lambda x : step_lr_scheduler(x,total_iterations,warmup_iters,warmup_factor)
    elif "exp"==name:
        beta=config["exp_beta"]
        return lambda x : exp_lr_scheduler(x,total_iterations,warmup_iters,warmup_factor,beta)
    else:
        raise NotImplementedError()

def get_loss_fun(config):
    train_crop_size=config["train_crop_size"]
    ignore_value=config["ignore_value"]
    if isinstance(train_crop_size,int):
        crop_h,crop_w=train_crop_size,train_crop_size
    else:
        crop_h,crop_w=train_crop_size
    loss_type="cross_entropy"
    if "loss_type" in config:
        loss_type=config["loss_type"]
    if loss_type=="cross_entropy":
        loss_fun=torch.nn.CrossEntropyLoss(ignore_index=ignore_value)
    elif loss_type=="bootstrapped":
        # 8*768*768/16
        minK=int(config["batch_size"]*crop_h*crop_w/16)
        print(f"bootstrapped minK: {minK}")
        loss_fun=BootstrappedCE(minK,0.3,ignore_index=ignore_value)
    else:
        raise NotImplementedError()
    return loss_fun

def get_optimizer(model,config):
    if not config["bn_weight_decay"]:
        p_bn = [p for n, p in model.named_parameters() if "bn" in n]
        p_non_bn = [p for n, p in model.named_parameters() if "bn" not in n]
        optim_params = [
            {"params": p_bn, "weight_decay": 0},
            {"params": p_non_bn, "weight_decay": config["weight_decay"]},
        ]
    else:
        optim_params = model.parameters()
    return torch.optim.SGD(
        optim_params,
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )

def get_dataset_loaders(config):
    name=config["dataset_name"]
    if name == "disaster":
        from datasets.disaster import DisasterDataset
        import torch.utils.data as data

        # For few-shot, the 'train_split' from the config ('support') is our training set
        train_set = DisasterDataset(
            root=".", # The paths in the split file are relative to the project root
            split_file=config["split_file"],
            mode=config["train_split"],
        )
        
        # The 'val_split' ('query') is our validation set
        val_set = DisasterDataset(
            root=".", # The paths in the split file are relative to the project root
            split_file=config["split_file"],
            mode=config["val_split"],
        )

        train_loader = data.DataLoader(
            train_set,
            batch_size=config["batch_size"],
            shuffle=True, # Shuffle the support set for training
            num_workers=config["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

        val_loader = data.DataLoader(
            val_set,
            batch_size=config["batch_size"],
            shuffle=False, # No need to shuffle the query set for evaluation
            num_workers=config["num_workers"],
            pin_memory=True,
        )
    else:
        raise NotImplementedError(f"Dataset '{name}' is not supported.")
    print("train size:", len(train_loader))
    print("val size:", len(val_loader))
    return train_loader, val_loader,train_set

