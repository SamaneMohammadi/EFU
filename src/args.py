import os
import uuid
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Experimental Configuration")
    # Add arguments
    parser.add_argument("--ds",             type=str,               default="cifar10",              help="Number of clients")
    parser.add_argument("--num_clients",    type=int,               default=5,                      help="Number of clients")
    parser.add_argument("--model_name",     type=str,               default="vgg16",                help="Model name in HF")
    #parser.add_argument("--model_name",     type=str,              default="microsoft/resnet-18",  help="Model name in HF")
    parser.add_argument("--lora_rank",      type=lambda x: int(x) if x is not None else None, default=None, help="LoRA rank (set to None for no LoRA)")
    parser.add_argument("--lr",             type=float,             default=1e-4,                   help="Learning rate")
    parser.add_argument("--num_rounds",     type=int,               default=150,                    help="Number of training rounds")
    parser.add_argument("--local_epochs",   type=int,               default=1,                      help="Number of local epochs")
    parser.add_argument("--batch_size",     type=int,               default=128,                    help="Batch size for training")
    parser.add_argument("--store_dir",      type=str,               default="../assets/ckpts",      help="Directory to store checkpoints")
    parser.add_argument("--seed",           type=int,               default=2025,                   help="Random seed")
    parser.add_argument("--device",         type=str,               default="cuda:0",               help="Device to use for training (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--train_type",     type=str,               default="FL",                   choices=["FL", "FL+DP", "FU", "FU+DP"], help="Type of training")
    parser.add_argument("--verbose",                                action="store_true",            help="Print experiment configuration")
    # Unlearn
    parser.add_argument("--erase_cids",     type=int, nargs="+",    default=[1],                    help="List of erased client IDs")
    parser.add_argument("--erase_rnd",      type=int,               default=90,                     help="Round at which clients are erased")
    parser.add_argument("--unlearn_method", type=str, default="efu", choices=["efu", "pgd", "sga_ewc", "fedosd"], help="client-side unlearning algorithm")
    # DP
    parser.add_argument("--delta",          type=float,             default=1e-1,                   help="Delta value")
    parser.add_argument("--max_grad",       type=float,             default=10.0,                   help="Maximum gradient clipping value")
    parser.add_argument("--max_noise",      type=float,             default=0.25,                   help="Maximum noise level")
    # Generate unique experiment ID
    args = parser.parse_args()
    args.unique_id = str(uuid.uuid4())[:8]
    args.ckpt_dir = os.path.join(args.store_dir, args.unique_id)
    if args.verbose:
        print("=" * 50)
        print("Experiment Configuration")
        print("=" * 50)
        [print(f"{k.replace('_',' ').capitalize():<20}: {v}") for k,v in vars(args).items()]
        print("=" * 50)
    return args