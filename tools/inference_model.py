
from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict
from vissl.models import build_model
from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_consolidated_weights

def get_trained_model(
    train_config: str,
    model_weights: str
):
    '''
    Build a model from the training config
    and initialize the weights with the given parameter.

    Arguments:
        train_config: str
            Path to the training config.
        model_weights: str
            Path to the trained model weights.
    '''

    config = OmegaConf.load(train_config)
    default_config = OmegaConf.load("configs/defaults.yaml")
    cfg = OmegaConf.merge(default_config, config)
    cfg = AttrDict(cfg)
    cfg.config.MODEL.WEIGHTS_INIT.PARAMS_FILE = model_weights
    cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON = True
    cfg.config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY = True
    cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY = True
    cfg.config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS = True
    cfg.config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP = [
        ["res5avg", ["Identity", []]]] # Only for ResNets. Change if needed.

    model = build_model(
        cfg.config.MODEL, cfg.config.OPTIMIZER)

    weights = load_checkpoint(
        checkpoint_path=cfg.config.MODEL.WEIGHTS_INIT.PARAMS_FILE)

    init_model_from_consolidated_weights(
        config=cfg.config,
        model=model,
        state_dict=weights,
        state_dict_key_name="classy_state_dict",
        skip_layers=[])

    return model
