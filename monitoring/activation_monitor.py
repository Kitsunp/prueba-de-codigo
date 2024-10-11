import torch
import torch.nn as nn

class ActivationMonitor:
    """
    Clase para monitorear activaciones y gradientes en el modelo.
    """

    def __init__(self, model):
        """
        Inicializa el monitor de activaciones y gradientes.

        Args:
            model (nn.Module): El modelo al que se le aplicarán los hooks para monitorear activaciones y gradientes.
        """
        self.handles = []
        self.activations = {}
        self.gradients = {}
        self.register_hooks(model)

    def register_hooks(self, model):
        """
        Registra hooks en las capas del modelo para capturar activaciones y gradientes.

        Args:
            model (nn.Module): El modelo al que se le aplicarán los hooks.
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
                handle = module.register_forward_hook(self.save_activation(name))
                handle_grad = module.register_backward_hook(self.save_gradient(name))
                self.handles.append(handle)
                self.handles.append(handle_grad)

    def save_activation(self, name):
        """
        Crea un hook para guardar las activaciones de una capa.

        Args:
            name (str): Nombre de la capa.

        Returns:
            function: Hook que guarda las activaciones.
        """
        def hook(module, input, output):
            self.activations[name] = output.detach()
            # Verificar rango de activaciones
            if not torch.isfinite(output).all():
                print(f"Activaciones no finitas en {name}")
        return hook

    def save_gradient(self, name):
        """
        Crea un hook para guardar los gradientes de una capa.

        Args:
            name (str): Nombre de la capa.

        Returns:
            function: Hook que guarda los gradientes.
        """
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()
            # Verificar rango de gradientes
            if not torch.isfinite(grad_output[0]).all():
                print(f"Gradientes no finitos en {name}")
        return hook

    def remove_hooks(self):
        """
        Elimina todos los hooks registrados.
        """
        for handle in self.handles:
            handle.remove()