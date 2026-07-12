import pytest
torch=pytest.importorskip("torch")
from torch import nn
from dsarp.models.losses import multitask_loss
from dsarp.models.multimodal_model import MultimodalRefactoringModel


class TinyEncoder(nn.Module):
    def __init__(self): super().__init__(); self.embedding=nn.Embedding(20,8)
    def forward(self,input_ids,attention_mask): return type("Output",(),{"last_hidden_state":self.embedding(input_ids)})()


@pytest.mark.parametrize("mode,classes",[("multiclass",5),("ordinal",4)])
def test_multitask_shapes_and_masking(mode,classes):
    model=MultimodalRefactoringModel(TinyEncoder(),8,3,[4],4,{"smell":3},2,[8],6,relevance_mode=mode)
    labels={"is_applicable":torch.tensor([1.,float("nan")]),"relevance_grade":torch.tensor([4.,float("nan")]),"expected_benefit_label":torch.tensor([.8,float("nan")]),"estimated_risk_label":torch.tensor([.2,float("nan")])}
    output=model(torch.tensor([[1,2],[2,0]]),torch.tensor([[1,1],[1,0]]),torch.randn(2,3),{"smell":torch.tensor([1,2])},labels)
    assert output.applicability_logits.shape==(2,1); assert output.relevance_logits.shape==(2,classes); assert output.fused_embedding.shape==(2,6); assert output.loss is not None; assert set(output.task_losses)=={"applicability","relevance","benefit","risk"}


def test_numeric_only_ablation():
    model=MultimodalRefactoringModel(None,0,3,[5],4,{"smell":3},2,[6],4,branches={"semantic":False,"numeric":True,"categorical":False}); output=model(numeric_features=torch.randn(2,3)); assert output.fused_embedding.shape==(2,4)
