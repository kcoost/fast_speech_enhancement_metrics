import tempfile
import subprocess
from pathlib import Path
import torch
import torch.nn as nn
from onnx2torch import convert


def graphmodule_to_sequential(gm: torch.fx.GraphModule) -> nn.Sequential:
    layers = [gm.get_submodule(n.target)for n in gm.graph.nodes if n.op == 'call_module']
    return nn.Sequential(*layers)


def create_dnsmos_weights(path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        model_path = tmpdir / "sig_bak_ovr.onnx"
        
        # Download model
        subprocess.run([
            "wget", "-c", "-O", model_path,
            "https://github.com/microsoft/DNS-Challenge/raw/refs/heads/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
        ], check=True)
        
        # Load and convert model
        primary_model = convert(model_path)
        sequential_model = graphmodule_to_sequential(primary_model)

        weights = {}
        weights['conv_real_stft.weight'] = sequential_model[7].weight
        weights['conv_imag_stft.weight'] = sequential_model[8].weight
        weights['conv_layers.0.weight'] = sequential_model[21].weight
        weights['conv_layers.0.bias'] = sequential_model[21].bias
        weights['conv_layers.2.weight'] = sequential_model[23].weight
        weights['conv_layers.2.bias'] = sequential_model[23].bias
        weights['conv_layers.4.weight'] = sequential_model[25].weight
        weights['conv_layers.4.bias'] = sequential_model[25].bias
        weights['conv_layers.6.weight'] = sequential_model[27].weight
        weights['conv_layers.6.bias'] = sequential_model[27].bias
        weights['conv_layers.9.weight'] = sequential_model[30].weight
        weights['conv_layers.9.bias'] = sequential_model[30].bias
        weights['conv_layers.12.weight'] = sequential_model[33].weight
        weights['conv_layers.12.bias'] = sequential_model[33].bias
        weights['conv_layers.15.weight'] = sequential_model[36].weight
        weights['conv_layers.15.bias'] = sequential_model[36].bias
        weights['output_layers.0.weight'] = primary_model.initializers.onnx_initializer_13.T
        weights['output_layers.0.bias'] = primary_model.initializers.onnx_initializer_14
        weights['output_layers.2.weight'] = primary_model.initializers.onnx_initializer_15.T
        weights['output_layers.2.bias'] = primary_model.initializers.onnx_initializer_16
        weights['output_layers.4.weight'] = primary_model.initializers.onnx_initializer_17.T
        weights['output_layers.4.bias'] = primary_model.initializers.onnx_initializer_18

    torch.save(weights, path)

if __name__ == "__main__":
    ...
    #create_dnsmos_weights(Path(__file__).parents[1] /"models" / "SIG_BAK_OVR.pt"))
