import os 
from datasets import load_dataset

import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from torch.distributed._composable.fsdp import fully_shard
import torch.distributed as dist
from tqdm import tqdm
from transformers.data import DataCollatorForSeq2Seq
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from torchdata.stateful_dataloader import StatefulDataLoader

from torchft import (
    DistributedSampler,
    Manager,
    Optimizer,
    ProcessGroupBabyNCCL,
    ProcessGroupGloo,
)
from torchft.process_group import ft_init_device_mesh

def hsdp_device_mesh(replica_group_size, sharding_group_size, device=None, manager=None):

    if replica_group_size is None or sharding_group_size is None:
        raise ValueError("Both replica_group_size and sharding_group_size must be provided.")

    device = device or f"cuda"

    device_mesh = ft_init_device_mesh(
        device_type=device,
        mesh_shape=(replica_group_size, sharding_group_size),
        mesh_dim_names=("dp_replicate", "dp_shard"),
        replicate_dim=0,
        manager=manager,
        )
    if device_mesh is None:
        raise RuntimeError("Failed to create a valid device mesh.")

    return device_mesh

def parallelize_llama(model, mesh):
    sharding_conditions = [lambda m: isinstance(m, LlamaDecoderLayer)]

    for m in reversed(list(model.modules())):
        if any(c(m) for c in sharding_conditions):
            # fully_shard(m, mesh=mesh, reshard_after_forward=True)
            fully_shard(m, mesh=mesh)
    # fully_shard([model.model.embed_tokens, model.lm_head], mesh=mesh)
    fully_shard(model, mesh=mesh)

def main():
    REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
    NUM_REPLICA_GROUPS = int(os.environ.get("NUM_REPLICA_GROUPS", 2))
    NUM_REPLICAS = int(os.environ.get("NUM_REPLICAS", 2))

    rank = int(os.environ.get("RANK", 0))

    model_name = "Meta-Llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    assert len(tokenizer) == model.get_input_embeddings().weight.shape[0]

    train_data = load_dataset("samsum", split="train")
    
    class SAMSumDataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer
        def __getitem__(self, idx):
            text = self.data[idx]
            prompt = self.tokenizer.encode(tokenizer.bos_token + f"Summarize this dialog: {text['dialogue']}\n---\nSummary: ", add_special_tokens=False)
            summary = self.tokenizer.encode(text["summary"] + self.tokenizer.eos_token, add_special_tokens=False)
            input_ids = prompt + summary
            labels = len(prompt) * [-100] + summary
            return {"input_ids": input_ids, "labels": labels}
        def __len__(self):
            return len(self.data)
    
    
    train_dataset = SAMSumDataset(train_data, tokenizer)
    
    batch_size = 8

    sampler = DistributedSampler(
        train_dataset,
        replica_group=REPLICA_GROUP_ID,
        num_replica_groups=NUM_REPLICA_GROUPS,
        rank=rank,
        shuffle=True,
        num_replicas=NUM_REPLICAS,
        )

    train_dataloader = StatefulDataLoader(train_dataset, batch_size=batch_size, collate_fn=DataCollatorForSeq2Seq(tokenizer), sampler=sampler)

    def load_state_dict(state_dict):
        set_state_dict(
            model,
            optimizer.optim,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        

    def state_dict():
        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer.optim)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
        }
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pg = ProcessGroupBabyNCCL() if torch.cuda.is_available() else ProcessGroupGloo()

    manager = Manager(
        pg=pg,
        min_replica_size=1,
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        replica_id=f"train_fsdp_{REPLICA_GROUP_ID}",
        use_async_quorum=False,
    )

    mesh = hsdp_device_mesh(NUM_REPLICA_GROUPS, NUM_REPLICAS, "cuda" if torch.cuda.is_available() else "cpu", manager=manager)
    
    parallelize_llama(model, mesh)

    model.to(device)
    
    optimizer = Optimizer(manager, torch.optim.Adam(model.parameters(), lr=1e-5))

    while manager.current_step() < 500:
        model.train()
        for batch in tqdm(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            if manager.current_step() % 100 == 0:
                print(f"[{manager.current_step()}] loss = {loss.item()}")


if __name__ == "__main__":
    main()
