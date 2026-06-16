import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from transformers import AutoModel


class ElementLinear(nn.Module):
    """Element-wise affine transform: y = x * w + b"""
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.weight, 1.0)
        nn.init.constant_(self.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias
    
    def get_sparsity(self) -> float:
        with torch.no_grad():
            w = self.weight.abs()
            sparse_ratio = ((w < 0.1) | (w > 0.9)).float().mean().item()
        return sparse_ratio


class SparseModalGate(nn.Module):
    """Sparse modal gating: G = Sigmoid(ElementLinear(V))"""
    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.element_linear = ElementLinear(embedding_dim)
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_weights = torch.sigmoid(self.element_linear(x))
        gated_x = gate_weights * x
        return gated_x, gate_weights


class PhysicochemicalBranch(nn.Module):
    """Physicochemical CNN branch: 29D -> CNN -> 768D"""
    def __init__(self, input_dim: int = 29, channels: list = None,
                 kernel_size: int = 3, output_dim: int = 768):
        super().__init__()
        if channels is None:
            channels = [128, 256, 512]
        layers = []
        in_channels = input_dim
        for out_channels in channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            ])
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_map_channels = channels[-1]
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
        self.output_dim = output_dim
        self.input_dim = input_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)
        feature_map = self.cnn(x)
        pooled = self.global_pool(feature_map).squeeze(-1)
        pooled = self.fc(pooled)
        return pooled, feature_map


class StructureBranch(nn.Module):
    """Secondary structure CNN branch: 8D -> CNN -> 768D"""
    def __init__(self, input_dim: int = 8, channels: list = None,
                 kernel_size: int = 3, output_dim: int = 768):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]
        layers = []
        in_channels = input_dim
        for out_channels in channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            ])
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_map_channels = channels[-1]
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
        self.output_dim = output_dim
        self.input_dim = input_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)
        feature_map = self.cnn(x)
        pooled = self.global_pool(feature_map).squeeze(-1)
        pooled = self.fc(pooled)
        return pooled, feature_map


class ReconstructionDecoder(nn.Module):
    """Reconstruction decoder: restore from 768-dim to high-dim feature map"""
    def __init__(self, input_dim: int = 768, target_channels: list = None,
                 target_length: int = 25):
        super().__init__()
        if target_channels is None:
            target_channels = [512, 256]
        layers = []
        prev_dim = input_dim
        for ch in target_channels:
            layers.extend([
                nn.Linear(prev_dim, ch * target_length),
                nn.BatchNorm1d(ch * target_length),
                nn.ReLU(inplace=True)
            ])
            prev_dim = ch * target_length
        self.decoder = nn.Sequential(*layers)
        self.final_channels = target_channels[-1]
        self.target_length = target_length
        self.input_dim = input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.final_channels, self.target_length)
        return x


class DNABERTFeatureExtractor(nn.Module):
    """DNABERT feature extractor (frozen)"""
    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    @property
    def output_dim(self) -> int:
        return self.hidden_size
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return cls_output


class GlobalClassifier(nn.Module):
    """Global classifier head: 1536D -> 256 -> 1 (logits, no sigmoid for AMP safety)"""
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)  # No Sigmoid! Use BCEWithLogitsLoss
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)


class MultimodalFusionModel(nn.Module):
    """HMF Multimodal Fusion Model (trunk network)"""
    
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is not None:
            self._load_from_config(config)
        else:
            self._load_from_kwargs(kwargs)
    
    def _load_from_config(self, config):
        from config import Config
        self.bert_extractor = DNABERTFeatureExtractor(config.MODEL_NAME, freeze=True)
        self.phy_branch = PhysicochemicalBranch(
            input_dim=config.PHY_FEATURE_DIM,
            channels=config.CNN_CHANNELS_PHY,
            kernel_size=config.KERNEL_SIZE,
            output_dim=config.EMBEDDING_DIM
        )
        self.str_branch = StructureBranch(
            input_dim=config.STR_FEATURE_DIM,
            channels=config.CNN_CHANNELS_STR,
            kernel_size=config.KERNEL_SIZE,
            output_dim=config.EMBEDDING_DIM
        )
        self.gate_phy = SparseModalGate(config.EMBEDDING_DIM)
        self.gate_str = SparseModalGate(config.EMBEDDING_DIM)
        self.recon_decoder_phy = ReconstructionDecoder(
            input_dim=config.EMBEDDING_DIM,
            target_channels=config.RECON_TARGET_CHANNELS_PHY,
            target_length=config.RECON_TARGET_LENGTH
        )
        self.recon_decoder_str = ReconstructionDecoder(
            input_dim=config.EMBEDDING_DIM,
            target_channels=config.RECON_TARGET_CHANNELS_STR,
            target_length=config.RECON_TARGET_LENGTH
        )
        self.global_classifier = GlobalClassifier(
            input_dim=config.CLASSIFIER_INPUT_DIM,
            hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
            dropout=config.DROPOUT
        )
    
    def _load_from_kwargs(self, kwargs):
        model_name = kwargs.get('model_name', './DNABERT3')
        emb_dim = kwargs.get('embedding_dim', 768)
        self.bert_extractor = DNABERTFeatureExtractor(model_name, freeze=True)
        self.phy_branch = PhysicochemicalBranch(
            input_dim=kwargs.get('phy_input_dim', 29),
            channels=kwargs.get('phy_channels', [128, 256, 512]),
            kernel_size=kwargs.get('kernel_size', 3), output_dim=emb_dim)
        self.str_branch = StructureBranch(
            input_dim=kwargs.get('str_input_dim', 8),
            channels=kwargs.get('str_channels', [64, 128, 256]),
            kernel_size=kwargs.get('kernel_size', 3), output_dim=emb_dim)
        self.gate_phy = SparseModalGate(emb_dim)
        self.gate_str = SparseModalGate(emb_dim)
        self.recon_decoder_phy = ReconstructionDecoder(
            input_dim=emb_dim, target_channels=kwargs.get('recon_phy_channels', [256, 512]),
            target_length=kwargs.get('recon_target_length', 25))
        self.recon_decoder_str = ReconstructionDecoder(
            input_dim=emb_dim, target_channels=kwargs.get('recon_str_channels', [128, 256]),
            target_length=kwargs.get('recon_target_length', 25))
        self.global_classifier = GlobalClassifier(
            input_dim=emb_dim * 2, hidden_dim=kwargs.get('classifier_hidden', 256),
            dropout=kwargs.get('dropout', 0.3))
    
    def forward(self, batch: dict) -> dict:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        F_bert = self.bert_extractor(input_ids, attention_mask)
        V_phy, phy_feature_map = self.phy_branch(batch['phy_features'])
        V_str, str_feature_map = self.str_branch(batch['str_features'])
        
        V_phy_gated, G_phy = self.gate_phy(V_phy)
        V_str_gated, G_str = self.gate_str(V_str)
        
        F_aux = (V_phy_gated + V_str_gated) / 2.0
        F_combined = torch.cat([F_bert, F_aux], dim=-1)
        
        pred_logits = self.global_classifier(F_combined)  # (batch,) - logits, no sigmoid
        
        recon_phy = self.recon_decoder_phy(F_aux)
        recon_str = self.recon_decoder_str(F_aux)
        
        return {
            'pred_logits': pred_logits,  # Use BCEWithLogitsLoss for AMP safety
            'F_bert': F_bert, 'F_aux': F_aux,
            'F_combined': F_combined, 'V_phy': V_phy, 'V_str': V_str,
            'G_phy': G_phy, 'G_str': G_str,
            'phy_feature_map': phy_feature_map, 'str_feature_map': str_feature_map,
            'recon_phy': recon_phy, 'recon_str': recon_str
        }
    
    def get_fusion_trunk_state(self) -> dict:
        return {
            'phy_branch': self.phy_branch.state_dict(),
            'str_branch': self.str_branch.state_dict(),
            'gate_phy': self.gate_phy.state_dict(),
            'gate_str': self.gate_str.state_dict(),
            'recon_decoder_phy': self.recon_decoder_phy.state_dict(),
            'recon_decoder_str': self.recon_decoder_str.state_dict(),
            'global_classifier': self.global_classifier.state_dict()
        }
    
    def load_fusion_trunk_state(self, state_dict: dict):
        self.phy_branch.load_state_dict(state_dict['phy_branch'])
        self.str_branch.load_state_dict(state_dict['str_branch'])
        self.gate_phy.load_state_dict(state_dict['gate_phy'])
        self.gate_str.load_state_dict(state_dict['gate_str'])
        self.recon_decoder_phy.load_state_dict(state_dict['recon_decoder_phy'])
        self.recon_decoder_str.load_state_dict(state_dict['recon_decoder_str'])
        if 'global_classifier' in state_dict:
            self.global_classifier.load_state_dict(state_dict['global_classifier'])


if __name__ == "__main__":
    print("="*60)
    print("  Model architecture module test")
    print("="*60)
    device = torch.device('cpu')
    el = ElementLinear(dim=10)
    x = torch.randn(2, 10)
    y = el(x)
    assert y.shape == (2, 10)
    print(f"  [OK] ElementLinear output shape: {y.shape}, sparsity: {el.get_sparsity():.4f}")
    
    gate = SparseModalGate(embedding_dim=768)
    x = torch.randn(4, 768)
    gated_x, weights = gate(x)
    assert gated_x.shape == (4, 768)
    assert (weights >= 0).all() and (weights <= 1).all()
    print(f"  [OK] SparseModalGate passed")
    print("\n[OK] All model component tests passed!")
