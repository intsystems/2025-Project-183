import torch
import copy


def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]


def get_weights_graph(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p for p in net.parameters() if p.requires_grad]


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
        new_direction = torch.stack([
            d_elem * (w_elem.norm() / (d_elem.norm() + 1e-10))
            for d_elem, w_elem in zip(direction, weights)
        ])
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        new_direction = direction * (weights.norm() / direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        new_direction = direction * weights
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        new_direction = torch.stack([
            d_elem / (d_elem.norm() + 1e-10)
            for d_elem in direction
        ])
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        new_direction = direction / direction.norm()
    else:
        new_direction = direction
    return new_direction


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    """
        The normalization scales the direction entries according to the entries of weights.
    """
    assert (len(direction) == len(weights))
    new_direction_list = []
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                new_direction_list.append(torch.zeros_like(d))  # ignore directions for weights with 1 dimension
            else:
                new_direction_list.append(w.clone())  # keep directions for weights/bias that are only 1 per node
        else:
            new_direction_list.append(normalize_direction(d, w, norm))
    return new_direction_list


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
    direction = normalize_directions_for_weights(direction, weights, norm, ignore)

    squared_norms = torch.stack([d.norm() ** 2 for d in direction])
    full_direction_norm = torch.sqrt(squared_norms.sum())
    if external_norm == 'unit':
        direction = [d / full_direction_norm for d in direction]

    direction = [d * external_factor for d in direction]
    return direction


def hessian_vector_product(loss, weights, vector):
    """
    Вычисляет произведение гессиана функции потерь на вектор с использованием двойного автоградирования.

    Шаги:
      1. Вычисляем градиенты loss по параметрам (с create_graph=True для возможности дальнейшего дифференцирования).
      2. Считаем скалярное произведение между полученными градиентами и вектором.
      3. Вычисляем градиент полученного скалярного произведения по параметрам, что и эквивалентно H*v.
    """
    grads = torch.autograd.grad(loss, weights, create_graph=True)
    grad_dot = torch.stack([(g * v).sum() for g, v in zip(grads, vector)]).sum()
    hv = torch.autograd.grad(grad_dot, weights, create_graph=True)
    return list(hv)


def subtract_projection(v, v1):
    """
    Вычитает из вектора v его проекцию на вектор v1, чтобы сделать v ортогональным v1.
    """
    dot_v_v1 = sum((a * b).sum() for a, b in zip(v, v1))
    dot_v1_v1 = sum((b * b).sum() for b in v1)
    projection = dot_v_v1 / (dot_v1_v1 + 1e-8)
    return [a - projection * b for a, b in zip(v, v1)]


def create_top_hessian_directions(net, loss,
                                  ignore='biasbn',
                                  norm='filter',
                                  external_norm='unit',
                                  external_factor=1.0,
                                  num_iter=10):
    """
    Вычисляет два направления, аппроксимирующих собственные векторы гессиана функции потерь,
    соответствующие наибольшим собственным значениям.

    Аргументы:
      model: обученная модель.
      criterion: функция потерь (например, nn.CrossEntropyLoss()).
      dataloader: даталоадер для вычисления loss.
      ignore: параметры, которые игнорируются при нормализации (например, 'biasbn').
      norm: метод нормализации ('filter', 'layer', 'weight', 'dlayer', 'dfilter').
      external_norm: внешний метод нормализации (например, 'unit').
      external_factor: коэффициент масштабирования результирующих направлений.
      num_iter: число итераций степенного метода.

    Возвращает:
      v1, v2: два направления (в виде списков тензоров), аппроксимирующие первые два собственных вектора гессиана.
    """
    # Для вычисления градиентов используем реальные параметры модели:
    if not isinstance(net, list):
        weights = get_weights_graph(net)  # a list of parameters.
    else:
        weights = net

    v1 = get_random_weights(weights)
    v2 = get_random_weights(weights)
    v1 = normalize_directions_for_weights(v1, weights, norm, ignore)
    v2 = normalize_directions_for_weights(v2, weights, norm, ignore)

    for _ in range(num_iter):
        hv = hessian_vector_product(loss, weights, v1)
        hv = normalize_directions_for_weights(hv, weights, norm, ignore)
        v1 = hv

    for _ in range(num_iter):
        hv = hessian_vector_product(loss, weights, v2)
        hv = subtract_projection(hv, v1)
        hv = normalize_directions_for_weights(hv, weights, norm, ignore)
        v2 = hv
        print(v2)

    squared_norms_v1 = torch.stack([d.norm() ** 2 for d in v1])
    full_norm_v1 = torch.sqrt(squared_norms_v1.sum())
    squared_norms_v2 = torch.stack([d.norm() ** 2 for d in v2])
    full_norm_v2 = torch.sqrt(squared_norms_v2.sum())

    if external_norm == 'unit':
        v1 = [d / full_norm_v1 for d in v1]
        v2 = [d / full_norm_v2 for d in v2]

    v1 = [d * external_factor for d in v1]
    v2 = [d * external_factor for d in v2]

    return v1, v2
