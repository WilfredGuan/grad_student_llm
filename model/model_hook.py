import json
import torch


class ModelOutputExporter:
    def __init__(self, index, model):
        self.model = model
        self.hooks = []
        self.outputs = [{"outputs": {}, "result": ""}]
        self.setup_hooks()
        print("ModelOutputExporter initialized.")
        # print(self.hooks)

    def setup_hooks(self):
        layers_of_interest = [f"model.layers.{i}.mlp" for i in range(32)]
        layers_of_interest += [f"model.layers.{i}.self_attn.o_proj" for i in range(32)]
        for name, module in self.model.named_modules():
            if name in layers_of_interest:
                hook = module.register_forward_hook(
                    lambda module, input, output, name=name: self.hook_function(
                        module, input, output, name
                    )
                )
                self.hooks.append(hook)

    def hook_function(self, module, input, output, layer_name):
        try:
            self.outputs[0]["outputs"][layer_name] = (
                output.detach().cpu().numpy().tolist()
            )
            # print("Hook output on", layer_name)

        except Exception as e:
            print("Error in hook function.")
            print(e)

    def run_model(self, input_data):
        self.outputs.clear()

        with torch.no_grad():
            self.model(input_data)

    def export_to_jsonl(self, save_path):
        with open(save_path, "a") as file:
            json_line = json.dumps(self.outputs[0])
            file.write(json_line + "\n")
        file.close()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
