#!/usr/bin/env python3
"""Offline HPC training entry point using transferred encoder weights."""
import argparse,json
from pathlib import Path
import pandas as pd
import yaml
from dsarp.models.multimodal_model import MultimodalRefactoringModel
from dsarp.models.registry import get_ablation
from dsarp.semantic.encoders import load_auto_model
from dsarp.semantic.tokenizer import load_tokenizer
from dsarp.training.trainer import train
from dsarp.training.reproducibility import run_metadata,set_seed

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=Path("configs/training.yaml")); parser.add_argument("--transformer-config",type=Path,default=Path("configs/transformer.yaml")); parser.add_argument("--model-config",type=Path,default=Path("configs/multimodal_model.yaml")); parser.add_argument("--dataset-config",type=Path,default=Path("configs/dataset.yaml")); parser.add_argument("--dataset",type=Path,default=Path("outputs/datasets/training_examples.parquet")); parser.add_argument("--ablation",default="full"); args=parser.parse_args()
    import torch
    from torch.utils.data import DataLoader,Dataset
    training=yaml.safe_load(args.config.read_text()); transformer=yaml.safe_load(args.transformer_config.read_text()); model_config=yaml.safe_load(args.model_config.read_text()); dataset_config=yaml.safe_load(args.dataset_config.read_text()); set_seed(int(training["seed"])); metadata=run_metadata([args.config,args.transformer_config,args.model_config,args.dataset_config],[args.dataset],transformer["model_name_or_path"])
    frame=pd.read_parquet(args.dataset) if args.dataset.suffix==".parquet" else pd.read_csv(args.dataset); numeric=dataset_config["numeric_features"]; categories=dataset_config["categorical_features"]
    vocabularies={name:{value:i+1 for i,value in enumerate(sorted(frame[name].fillna("<MISSING>").astype(str).unique()))} for name in categories}
    tokenizer=load_tokenizer(transformer["tokenizer_name_or_path"],local_files_only=True); encoder=load_auto_model(transformer["model_name_or_path"],local_files_only=True) if get_ablation(args.ablation)["semantic"] else None
    class FrameDataset(Dataset):
        def __init__(self,data:pd.DataFrame)->None: self.data=data.reset_index(drop=True)
        def __len__(self)->int: return len(self.data)
        def __getitem__(self,index:int):
            row=self.data.iloc[index]; encoded=tokenizer(str(row.semantic_context_text),max_length=int(transformer["max_sequence_length"]),truncation=True,padding="max_length",return_tensors="pt")
            item={"input_ids":encoded["input_ids"].squeeze(0),"attention_mask":encoded["attention_mask"].squeeze(0),"numeric_features":torch.tensor([float(row.get(c,0)) if pd.notna(row.get(c)) else 0 for c in numeric],dtype=torch.float32),"categorical_features":{c:torch.tensor(vocabularies[c].get(str(row.get(c,"<MISSING>")),0)) for c in categories},"labels":{name:torch.tensor(float(row.get(column))) if pd.notna(row.get(column)) else torch.tensor(float("nan")) for name,column in {"is_applicable":"is_applicable","relevance_grade":"relevance_grade","expected_benefit_label":"expected_benefit_label","estimated_risk_label":"estimated_risk_label"}.items()}}
            return item
    train_frame=frame[frame.split=="train"]; validation_frame=frame[frame.split=="validation"]
    if train_frame.empty or validation_frame.empty: raise ValueError("Training and validation splits must both contain examples")
    model=MultimodalRefactoringModel(encoder,int(encoder.config.hidden_size) if encoder else 0,len(numeric),model_config["numeric_hidden_dims"],model_config["numeric_output_dim"],{c:max(len(vocabularies[c]),model_config["categorical_cardinalities"].get(c,0)) for c in categories},model_config["categorical_embedding_dim"],model_config["fusion_hidden_dims"],model_config["fusion_output_dim"],model_config["dropout"],model_config["relevance_classes"],model_config["relevance_mode"],model_config["uncertainty_head"],get_ablation(args.ablation),model_config["loss_weights"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(training["learning_rate"]),weight_decay=float(training["weight_decay"])); scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def evaluate(current,loader,device):
        current.eval(); correct=total=0
        with torch.no_grad():
            for batch in loader:
                batch={k:({x:y.to(device) for x,y in v.items()} if isinstance(v,dict) else v.to(device)) for k,v in batch.items()}; output=current(**batch); labels=batch["labels"]["is_applicable"]; mask=torch.isfinite(labels); correct+=int(((output.applicability_logits.squeeze(-1)[mask]>0).float()==labels[mask]).sum()); total+=int(mask.sum())
        return correct/max(total,1)
    best=train(model,DataLoader(FrameDataset(train_frame),batch_size=int(training["batch_size"]),shuffle=True),DataLoader(FrameDataset(validation_frame),batch_size=int(training["evaluation_batch_size"])),optimizer,scheduler,training,device,Path(training["checkpoint_dir"]),evaluate)
    metadata.update({"best":best,"ablation":args.ablation,"vocabularies":vocabularies,"numeric_features":numeric}); Path("models/metadata").mkdir(parents=True,exist_ok=True); Path("models/metadata/training_run.json").write_text(json.dumps(metadata,indent=2,default=str))
if __name__=="__main__": main()
