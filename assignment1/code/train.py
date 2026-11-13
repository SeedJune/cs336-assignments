import modules
import swanlab
import argparse
from bpe_tokenizer import Tokenizer
import numpy as np
from tqdm import tqdm
import numpy.typing as npt
import time
import torch

def train(model:modules.Transformer_LM, optimizer:modules.AdamW, train_dataset: npt.NDArray, valid_dataset:npt.NDArray, iteration:int, args):
    start_time = time.perf_counter()
    pbar = tqdm(range(iteration, args.num_steps), desc="Training steps")
    last = iteration
    for t in pbar:
        model.train()
        lr = modules.Cosine_Annealing_Schedule(t, args.max_lr, args.min_lr, args.warmup_steps, args.num_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        inputs, targets = modules.Data_Loading(train_dataset, args.batch_size, args.context_length, device=args.device) 
        outputs = model(inputs)
        loss = modules.cross_entropy(outputs, targets)
        loss.backward()
        modules.Gradient_Clipping(model.parameters(), M=1.0)
        optimizer.step()
        metrics = {
            "training_loss": loss.item(),
            "learning_rate": lr
        }
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        last = t
        if (t + 1) % 100 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i in range(10):
                    inputs, targets = modules.Data_Loading(valid_dataset, args.batch_size, args.context_length, device=args.device)
                    logits = model(inputs)
                    loss = modules.cross_entropy(logits, targets)
                    val_loss += loss.item()
            metrics["val_loss"] = val_loss / 10
        end_time = time.perf_counter()
        metrics["training_time"] = end_time - start_time
        swanlab.log(metrics, step=t + 1)
    modules.save_checkpoint(model, optimizer, last + 1, args.save_path)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--max_lr", type=float, default=3e-4)
    parse.add_argument("--min_lr", type=float, default=3e-5)
    parse.add_argument("--batch_size", type=int, default=32)
    parse.add_argument("--device", type=str, default="cpu")
    parse.add_argument("--train_data", type=str)
    parse.add_argument("--valid_data", type=str)
    parse.add_argument("--warmup_steps", type=int, default=500)
    parse.add_argument("--num_steps", type=int, default=40000)
    parse.add_argument("--load_from", action="store_true")
    parse.add_argument("--ckpt_path", type=str, default=None)
    parse.add_argument("--vocab_path", type=str)
    parse.add_argument("--save_path", type=str)
    parse.add_argument("--context_length", type=int, default=256)

    args = parse.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab_path, special_tokens=["<|endoftext|>"])
    vocab_size = tokenizer.vocab_size
    model = modules.Transformer_LM(vocab_size=vocab_size,context_length=args.context_length, d_model=512, num_layers=4, num_heads=16, d_ff=None, rope_theta=10000.0)
    model.to(args.device)
    optimizer = modules.AdamW(model.parameters(), args.max_lr, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    if args.load_from:
        iteration = modules.load_checkpoint(args.ckpt_path, model, optimizer)
    else:
        iteration = 0

    train_dataset = np.memmap(args.train_data, dtype=np.uint16)
    valid_dataset = np.memmap(args.valid_data, dtype=np.uint16)

    config = {
        **vars(args),
        "vocab_size": tokenizer.vocab_size,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "rope_theta": 10000.0,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
    }
    swanlab.init(project='cs336_assgnment1', name="openweb", config=config, mode="online")
    train(model=model, optimizer=optimizer, train_dataset=train_dataset, valid_dataset=valid_dataset, iteration=iteration, args=args)
    swanlab.finish()

    print(f"Finish training. The checkpoint is saved in {args.save_path}.")
