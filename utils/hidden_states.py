"""Raw decoder-output collection, shared between calibration and inference."""
from contextlib import contextmanager


def decoder_layers(model):
    for path in (("model", "layers"), ("model", "model", "layers"),
                 ("base_model", "model", "model", "layers"), ("base_model", "model", "layers"),
                 ("transformer", "h"), ("gpt_neox", "layers")):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError("Cannot locate decoder layers on target model.")


@contextmanager
def capture_decoder_outputs(model):
    # HF hidden_states[-1] may include the final model norm, whereas interventions
    # act on decoder block outputs. Collect the same raw states in every path.
    states = [None] * len(decoder_layers(model))
    handles = []
    def hook_for(i):
        def hook(module, inputs, output):
            states[i] = output[0] if isinstance(output, tuple) else output
        return hook
    try:
        for i, layer in enumerate(decoder_layers(model)):
            handles.append(layer.register_forward_hook(hook_for(i)))
        yield states
    finally:
        for handle in handles:
            handle.remove()
