def build_model(config, general_transformer=False, **kwargs):
    if general_transformer:
        from .general_relationformer import build_general_relationformer
        return build_general_relationformer(config, **kwargs)
    else:
        from .relationformer import build_relationformer
        return build_relationformer(config, **kwargs)
