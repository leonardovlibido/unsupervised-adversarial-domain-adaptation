

def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def freeze_model(model):
    set_parameter_requires_grad(model, False)


def unfreeze_model(model):
    set_parameter_requires_grad(model, True)

