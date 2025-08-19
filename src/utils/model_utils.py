from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2DTranspose

def load_model_with_groups_fix(model_path: str):
    """
    Loads a Keras .h5 model. If Conv2DTranspose(groups=...) is present but not
    supported by your TF/Keras version, it patches the layer to drop 'groups'.
    """
    try:
        _ = Conv2DTranspose(filters=32, kernel_size=3, groups=2)  # probe support
        return load_model(model_path)  # groups supported -> normal load
    except TypeError:
        # define a wrapper that discards the 'groups' kwarg
        def Conv2DTranspose_without_groups(*args, **kwargs):
            kwargs.pop("groups", None)
            return Conv2DTranspose(*args, **kwargs)
        return load_model(model_path, custom_objects={"Conv2DTranspose": Conv2DTranspose_without_groups})
