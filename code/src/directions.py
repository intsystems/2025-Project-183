import torch
import copy


def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]


def get_random_weights(weights):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size(), device=w.device) for w in weights]


def normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.

        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm() / (d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm() / direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    """
        The normalization scales the direction entries according to the entries of weights.
    """
    assert (len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0)  # ignore directions for weights with 1 dimension
            else:
                d.copy_(w)  # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


def inplace_sum_models(model1, model2, coef1, coef2):
    """
        return model1 := model1 * coef1 + model2 * coef2
    """
    final = model1
    for (name1, param1), (name2, param2) in zip(final.state_dict().items(), model2.state_dict().items()):
        transformed_param = param1 * coef1 + param2 * coef2
        param1.copy_(transformed_param)
    return final


def calc_sum_models(model1, model2, coef1, coef2):
    final = copy.deepcopy(model1)
    final.load_state_dict(copy.deepcopy(model1.state_dict()))
    return inplace_sum_models(final, model2, coef1, coef2)


def init_from_params(model, direction):
    """
        inplace init model from direction as from parameters()
    """
    for p_orig, p_other in zip(model.parameters(), direction):
        with torch.no_grad():
            p_orig.copy_(p_other)


def create_random_direction(net,
                            ignore='biasbn',
                            norm='filter',
                            external_norm='unit',
                            external_factor=1.0):
    """
        Set up a random (normalized) direction with the same dimension as
        the weights or states.

        Args:
          net: the given trained model
          ignore: biasbn,
          norm: direction normalization method, including
                'filter' | 'layer' | 'weight' | 'dlayer' | 'dfilter'
          external_norm: external normalization method, including
                'unit'
          external_factor: linalg norm of result direction

        Returns:
          direction: a random direction with the same dimension as weights or states.
    """
    if not isinstance(net, list):
        weights = get_weights(net)  # a list of parameters.
    else:
        weights = net
    direction = get_random_weights(weights)
    normalize_directions_for_weights(direction, weights, norm, ignore)

    squared_norms = torch.stack([d.norm() ** 2 for d in direction])
    full_direction_norm = torch.sqrt(squared_norms.sum())
    if external_norm == 'unit':
        for d in direction:
            d.div_(full_direction_norm)

    for d in direction:
        d.mul_(external_factor)

    return direction
