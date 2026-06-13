from dataclasses import dataclass
import json
import os

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.eval_b1k_wrapper import B1KPolicyWrapper, load_modality_config
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.websocket_b1k_server import WebsocketPolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 8000


@dataclass
class ServerConfig:
    """Configuration for running the Groot N1.5 inference server."""

    # Gr00t policy configs
    model_path: str
    """Path to the model checkpoint directory"""

    modality_config_path: str
    """Path to the modality configuration python file"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the model on"""

    control_mode: str = "temporal_ensemble"
    """Control mode during inference."""

    # Server configs
    host: str = "127.0.0.1"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

        
def main(config: ServerConfig):
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Modality config path: {config.modality_config_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # check if the model path exists
    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # load modality config if provided
    assert os.path.exists(config.modality_config_path) and config.modality_config_path.endswith(".py"), (
        f"Modality config path {config.modality_config_path} does not exist or is not a Python file"
    )
    load_modality_config(config.modality_config_path)
    modality_json = config.modality_config_path.replace(".py", ".json")
    assert os.path.exists(modality_json), (f"Modality config JSON file {modality_json} does not exist. ")
    with open(modality_json, "r") as f:
        modality_config = json.load(f)

    # Create and start the server
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
    )

    # Wrap with B1K policy wrapper
    policy = B1KPolicyWrapper(
        policy=policy,
        embodiment_tag=config.embodiment_tag,
        modality_config=modality_config,
        control_mode=config.control_mode,
    )

    server = WebsocketPolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
