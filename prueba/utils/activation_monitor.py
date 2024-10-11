# prueba/utils/activation_monitor.py

"""
ActivationMonitor module for tracking activations and gradients in the model.
"""

import torch
import torch.nn as nn

class ActivationMonitor:
    """
    A class to monitor activations and gradients of specified layers in a neural network model.
    """

    def __init__(self, model):
        """
        Initializes the ActivationMonitor with a given model.

        Args:
            model (nn.Module): The model whose activations and gradients are to be monitored.
        """
        self.handles = []
        self.activations = {}
        self.gradients = {}
        self.register_hooks(model)

    def register_hooks(self, model):
        """
        Registers forward and backward hooks to monitor activations and gradients.

        Args:
            model (nn.Module): The model to register hooks on.
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
                handle = module.register_forward_hook(self.save_activation(name))
                handle_grad = module.register_backward_hook(self.save_gradient(name))
                self.handles.append(handle)
                self.handles.append(handle_grad)

    def save_activation(self, name):
        """
        Creates a hook function to save activations.

        Args:
            name (str): The name of the module.

        Returns:
            function: A hook function to save activations.
        """
        def hook(module, input, output):
            self.activations[name] = output.detach()
            if not torch.isfinite(output).all():
                print(f"Non-finite activations in {name}")
        return hook

    def save_gradient(self, name):
        """
        Creates a hook function to save gradients.

        Args:
            name (str): The name of the module.

        Returns:
            function: A hook function to save gradients.
        """
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()
            if not torch.isfinite(grad_output[0]).all():
                print(f"Non-finite gradients in {name}")
        return hook

    def remove_hooks(self):
        """
        Removes all registered hooks.
        """
        for handle in self.handles:
            handle.remove()