from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class MoEArguments:
    """
    Configuration for Mixture-of-Experts (MoE).
    """
    use_moe: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use MoE for expert selection."}
    )
    moe_top_k: Optional[int] = field(
        default=2,
        metadata={"help": "Top-K experts to select during routing."}
    )
    moe_num_experts: Optional[int] = field(
        default=4,
        metadata={"help": "Total number of MoE experts; often equals the number of FL clients."}
    )
    moe_hidden_size: Optional[int] = field(
        default=768,
        metadata={"help": "Hidden size for the MoE router."}
    )
    moe_router_type: Optional[str] = field(
        default="keyword",
        metadata={"help": "Type of MoE router: 'keyword' (keyword matching) or 'model' (model-based routing)."}
    )
    expert_names: Optional[List[str]] = field(
        default_factory=lambda: [
            "Automotive Design Expert",
            "Automotive Manufacturing Expert",
            "Automotive Supply Chain Expert",
            "Automotive Quality Assurance Expert"
        ],
        metadata={"help": "List of MoE expert names."}
    )
    expert_descriptions: Optional[List[str]] = field(
        default_factory=lambda: [
            "Handles exterior and interior design with a focus on aesthetics and UX.",
            "Responsible for manufacturing processes, aiming for efficient, high-quality assembly.",
            "Manages parts and materials supply chain to ensure steady production.",
            "Conducts quality inspection and failure analysis to ensure safety and reliability."
        ],
        metadata={"help": "List of MoE expert descriptions."}
    )
    moe_router_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pretrained MoE router checkpoint, if available."}
    )
    default_expert_indices: Optional[List[int]] = field(
        default_factory=lambda: [0, 1],
        metadata={"help": "Fallback expert indices if the MoE router is unavailable."}
    )
    log_routing_stats: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to log MoE routing statistics."}
    )
    routing_balance_strategy: Optional[str] = field(
        default="count",
        metadata={"help": "Balancing strategy for expert selection: 'none', 'count', or 'weight'."}
    )
    use_improved_routing: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use an improved routing mechanism (e.g., routing score influences loss)."}
    )

    router_loss_weight: Optional[float] = field(
        default=0.1,
        metadata={"help": "Weight for the router loss term."}
    )

    load_balancing_loss_weight: Optional[float] = field(
        default=0.01,
        metadata={"help": "Weight for the load-balancing loss term."}
    )


def update_config_with_moe_args():
    """
    Update an existing config module by adding MoE CLI args to its parser.
    """
    import sys

    # Try to import config module
    try:
        from config import get_config, save_config, parser
    except ImportError:
        print("Unable to import the config module.")
        return False

    # Check if the parser is compatible and whether MoE args already exist
    if hasattr(parser, 'parse_args_into_dataclasses'):
        if any(param.dest == 'use_moe' for param in parser._actions):
            print("MoE arguments already exist in the config.")
            return True

        # Add MoE arguments
        try:
            parser.add_argument_group("MoE Arguments")
            parser.add_argument("--use_moe", type=bool, default=False)
            parser.add_argument("--moe_top_k", type=int, default=2)
            parser.add_argument("--moe_num_experts", type=int, default=4)
            parser.add_argument("--moe_router_type", type=str, default="keyword")
            parser.add_argument("--log_routing_stats", type=bool, default=True)
            print("Successfully added MoE arguments to the config.")
            return True
        except Exception as e:
            print(f"Failed to add MoE arguments: {e}")
            return False
    else:
        print("The config parser format is incompatible.")
        return False


def fix_moe_value_chain_compatibility(moe_args, vc_args, fed_args):
    """
    Ensure compatibility between MoE arguments and value-chain arguments.

    Args:
        moe_args: MoE args.
        vc_args: Value-chain args.
        fed_args: Federated learning args.

    Returns:
        Tuple of (moe_args, vc_args, fed_args) after updates.
    """
    # Ensure the number of experts matches the number of clients
    if moe_args.use_moe:
        # Sync MoE experts with FL clients
        moe_args.moe_num_experts = fed_args.num_clients

        # Pad expert names/descriptions if needed
        while len(moe_args.expert_names) < moe_args.moe_num_experts:
            index = len(moe_args.expert_names)
            moe_args.expert_names.append(f"Expert{index}")
            moe_args.expert_descriptions.append(f"Domain{index}")

        # Constrain top_k
        moe_args.moe_top_k = min(moe_args.moe_top_k, moe_args.moe_num_experts)

        # Coordinate with value-chain if enabled
        if hasattr(vc_args, 'use_value_chain') and vc_args.use_value_chain:
            print(
                "Warning: Both MoE and Value Chain are enabled. "
                "MoE will route/choose clients, while Value Chain handles stage specialization and collaborative training."
            )

    return moe_args, vc_args, fed_args
