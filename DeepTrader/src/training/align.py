import os
import torch
import torch.nn as nn
from tqdm import tqdm
from .utils import TrainingLogger

def run_alignment(
    model,
    train_datasets,
    val_datasets,
    training_config,
    output_dir,
    log_dir,
):
    """
    Phase 3: DPO (Direct Preference Optimization) for Classification.
    Increases the probability of winning trades relative to losing trades.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() and training_config.get("device") != "cpu" else "cpu"
    )
    model = model.to(device)

    batch_size = training_config.get("batch_size", 32)
    lr = float(training_config.get("learning_rate", 5e-6))
    weight_decay = float(training_config.get("weight_decay", 0.01))
    epochs = training_config.get("epochs", 10)
    beta = float(training_config.get("beta", 0.1)) # DPO temperature

    logger = TrainingLogger(log_dir, "phase3_align")

    # Combine datasets
    train_ds = torch.utils.data.ConcatDataset(train_datasets)
    val_ds = torch.utils.data.ConcatDataset(val_datasets)

    # Use a small number of workers or 0 on Windows
    num_workers = 0 if os.name == 'nt' else 2

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Load reference model (frozen copy for DPO)
    import copy
    ref_model = copy.deepcopy(model)
    ref_model.eval()

    best_val_loss = float("inf")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  [Align] Train: {len(train_ds)} | Val: {len(val_ds)}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs} [Train]")
        for batch in pbar:
            m30_x = batch["m30_input"].to(device)
            h1_x = batch["h1_context"].to(device)
            h4_x = batch["h4_context"].to(device)
            targets = batch["target_probs"].to(device) # [batch, 3] -> Long, Short, Abort
            
            # For DPO, we separate "chosen" (wins) vs "rejected" (losses)
            # A trade is a win if Long=1 or Short=1.
            # A trade is a loss if Abort=1 and Both=0.
            
            wins = (targets[:, 0] == 1) | (targets[:, 1] == 1)
            losses = targets[:, 2] == 1
            
            # Skip batches without pairs
            if wins.sum() == 0 or losses.sum() == 0:
                continue
                
            optimizer.zero_grad()

            # Active Model 
            logits = model.finetune_forward(m30_x, h1_x, h4_x) 
            pi_logprobs = torch.log_softmax(logits, dim=-1)
            
            # Reference Model
            with torch.no_grad():
                ref_logits = ref_model.finetune_forward(m30_x, h1_x, h4_x)
                ref_logprobs = torch.log_softmax(ref_logits, dim=-1)

            # DPO Equation
            pi_ratio = pi_logprobs[wins].mean() - pi_logprobs[losses].mean()
            ref_ratio = ref_logprobs[wins].mean() - ref_logprobs[losses].mean()
            
            loss = -torch.nn.functional.logsigmoid(beta * (pi_ratio - ref_ratio))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"dpo_loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                m30_x = batch["m30_input"].to(device)
                h1_x = batch["h1_context"].to(device)
                h4_x = batch["h4_context"].to(device)
                targets = batch["target_probs"].to(device)
                
                wins = (targets[:, 0] == 1) | (targets[:, 1] == 1)
                losses = targets[:, 2] == 1
                
                if wins.sum() == 0 or losses.sum() == 0:
                    continue

                logits = model.finetune_forward(m30_x, h1_x, h4_x)
                pi_logprobs = torch.log_softmax(logits, dim=-1)
                
                ref_logits = ref_model.finetune_forward(m30_x, h1_x, h4_x)
                ref_logprobs = torch.log_softmax(ref_logits, dim=-1)
                
                pi_ratio = pi_logprobs[wins].mean() - pi_logprobs[losses].mean()
                ref_ratio = ref_logprobs[wins].mean() - ref_logprobs[losses].mean()
                
                loss = -torch.nn.functional.logsigmoid(beta * (pi_ratio - ref_ratio))
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))

        print(f"  -> Train DPO: {avg_train_loss:.4f} | Val DPO: {avg_val_loss:.4f}")

        logger.log_epoch(
            epoch=epoch,
            train_metrics={"loss": avg_train_loss},
            val_metrics={"loss": avg_val_loss}
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(output_dir, "align_best.pt")
            model.save_checkpoint(save_path, optimizer=None, epoch=epoch)
            print(f"  [+] Saved best model to {save_path}")

    return model
