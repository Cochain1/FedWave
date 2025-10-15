import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np
import re

class MoERouter(nn.Module):
    """
    Mixture-of-Experts (MoE) router — routes inputs to the most relevant value-chain stage / client.
    """
    def __init__(
        self, 
        hidden_size: int = 768,
        num_experts: int = 4, 
        top_k: int = 2,
        expert_names: List[str] = None,
        expert_descriptions: List[str] = None,
        use_keywords: bool = True
    ):
        """
        Initialize the MoE router.
        
        Args:
            hidden_size: Hidden size used for feature projection.
            num_experts: Number of experts (clients).
            top_k: Number of top experts to select.
            expert_names: List of expert names (e.g., ["Design Expert", "Production Expert"]).
            expert_descriptions: List of expert descriptions for better keyword extraction.
            use_keywords: Whether to use keyword matching as an auxiliary routing signal.
        """
        super(MoERouter, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.expert_names = expert_names or [f"Expert{i}" for i in range(num_experts)]
        self.expert_descriptions = expert_descriptions or [""] * num_experts
        self.use_keywords = use_keywords
        
        # Main routing MLP — maps input features to expert logits
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        )
        
        # Optional keyword table — list of keywords per expert
        self.expert_keywords = self._initialize_expert_keywords()
        
        # Routing statistics
        self.routing_stats = {
            "calls": 0,
            "expert_selection": [0] * num_experts,
            "avg_confidence": [0.0] * num_experts
        }
    
    def _initialize_expert_keywords(self) -> Dict[int, List[str]]:
        """Initialize expert keyword table by extracting from names/descriptions plus domain defaults."""
        keywords = {}
        
        # Predefined keyword sets for automotive domain experts (in English)
        domain_specific_keywords = {
            0: [
                "design","exterior","styling","style","aesthetics","comfort","space","interior","seat",
                "steering wheel","dashboard","ui","user experience","ux","human-machine interface","hmi",
                "industrial design"
            ],
            1: [
                "production","manufacturing","assembly","factory","line","throughput","efficiency","process",
                "materials","parts","quality control","workflow","machining","welding","painting","mold","automation"
            ],
            2: [
                "supply chain","logistics","procurement","inventory","supplier","transport","shipping",
                "warehouse","jit","just in time","cost control","vendor management","scm","components","raw materials"
            ],
            3: [
                "quality","testing","failure","inspection","defect","performance","safety","standard","compliance",
                "reliability","durability","measurement","warranty","diagnostics","experiment","validation","evaluation"
            ],
        }
        
        # Use predefined keywords and add those extracted from names/descriptions
        for i in range(self.num_experts):
            if i in domain_specific_keywords:
                keywords[i] = domain_specific_keywords[i]
            else:
                text = f"{self.expert_names[i]} {self.expert_descriptions[i]}"
                extracted = re.findall(r'\w+', text)
                keywords[i] = [word for word in extracted if len(word) > 1]
        
        return keywords
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Forward pass — determine which experts should process each input.
        
        Args:
            hidden_states: Input features of shape [batch_size, seq_len, hidden_size].
            
        Returns:
            routing_weights: Weights over experts, shape [batch_size, num_experts].
            expert_indices: Top-k expert indices per sample, shape [batch_size, top_k].
        """
        # Mean-pool across sequence to get a single vector per sample
        # [batch, seq, hidden] -> [batch, hidden]
        feature_vec = hidden_states.mean(dim=1)
        
        # Compute expert logits and softmax
        # [batch, hidden] -> [batch, num_experts]
        router_logits = self.router(feature_vec)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        scores, expert_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Update stats
        self.routing_stats["calls"] += hidden_states.size(0)
        for i in range(hidden_states.size(0)):
            for j in range(self.top_k):
                expert_idx = expert_indices[i, j].item()
                self.routing_stats["expert_selection"][expert_idx] += 1
                self.routing_stats["avg_confidence"][expert_idx] += routing_weights[i, expert_idx].item()
        
        return routing_weights, expert_indices
    
    def route_text_input(self, text_inputs: List[str], tokenizer=None) -> List[List[int]]:
        """
        Route raw text inputs to experts.
        
        Args:
            text_inputs: List of input texts.
            tokenizer: Optional tokenizer for text encoding.
            
        Returns:
            List of expert index lists (top-k per input).
        """
        batch_size = len(text_inputs)
        device = next(self.parameters()).device
        
        # Keyword-based routing (default)
        if self.use_keywords:
            expert_scores = torch.zeros((batch_size, self.num_experts), device=device)
            
            for i, text in enumerate(text_inputs):
                text_l = text.lower()
                # Compute a keyword match score per expert
                for expert_idx, kws in self.expert_keywords.items():
                    score = 0.0
                    for kw in kws:
                        kw_l = kw.lower()
                        if kw_l in text_l:
                            # Higher weight if keyword appears near the beginning
                            pos = text_l.find(kw_l)
                            pos_weight = 1.0 if pos < max(1, len(text_l) // 3) else 0.5
                            score += pos_weight
                    expert_scores[i, expert_idx] = score
            
            # Normalize and choose top-k
            expert_scores = F.softmax(expert_scores, dim=-1)
            _, expert_indices = torch.topk(expert_scores, self.top_k, dim=-1)
            expert_indices_list = expert_indices.cpu().tolist()
            
            # If an input had uniform scores (no matches), assign the first top_k experts as a fallback
            for i in range(batch_size):
                if torch.allclose(expert_scores[i], expert_scores[i][0]):
                    expert_indices_list[i] = list(range(min(self.top_k, self.num_experts)))
            
            return expert_indices_list
        
        # Feature-based routing with tokenizer (placeholder example)
        elif tokenizer is not None:
            with torch.no_grad():
                inputs = tokenizer(
                    text_inputs, padding=True, truncation=True, return_tensors="pt", max_length=512
                ).to(device)
                # NOTE: This is a placeholder; a real implementation should use
                # an encoder to obtain hidden states. Here we project token IDs as a stub.
                hidden_states = self.router[0](inputs.input_ids.float())
                _, expert_indices = self.forward(hidden_states)
                return expert_indices.cpu().tolist()
        
        # Random routing if neither keywords nor tokenizer are available
        else:
            expert_indices = []
            for _ in range(batch_size):
                indices = np.random.choice(self.num_experts, self.top_k, replace=False)
                expert_indices.append(indices.tolist())
            return expert_indices
    
    def get_routing_stats(self) -> Dict:
        """Return routing statistics with average confidence and expert names."""
        stats = self.routing_stats.copy()
        if stats["calls"] > 0:
            for i in range(self.num_experts):
                if stats["expert_selection"][i] > 0:
                    stats["avg_confidence"][i] /= stats["expert_selection"][i]
        stats["expert_names"] = self.expert_names
        return stats
    
    def reset_stats(self):
        """Reset routing statistics."""
        self.routing_stats = {
            "calls": 0,
            "expert_selection": [0] * self.num_experts,
            "avg_confidence": [0.0] * self.num_experts
        }

def create_automotive_moe_router(
    hidden_size: int = 768, 
    num_experts: int = 4, 
    top_k: int = 2
) -> MoERouter:
    """
    Create an MoE router specialized for the automotive domain.
    
    Args:
        hidden_size: Hidden size for the router MLP.
        num_experts: Number of experts.
        top_k: Number of top experts to select.
        
    Returns:
        A configured MoERouter instance.
    """
    expert_names = [
        "Automotive Design Expert",
        "Automotive Production Expert", 
        "Automotive Supply Chain Expert",
        "Automotive Quality Assurance Expert"
    ]
    
    expert_descriptions = [
        "Responsible for vehicle exterior/interior design with strong focus on aesthetics and user experience.",
        "Responsible for automotive manufacturing processes, targeting efficient and high-quality assembly.",
        "Responsible for component supply chain management to ensure continuous line material availability.",
        "Responsible for quality testing and fault analysis to ensure product quality and safety."
    ]
    
    return MoERouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        expert_names=expert_names,
        expert_descriptions=expert_descriptions,
        use_keywords=True
    )
