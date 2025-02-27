import torch as t

def test_get_inputs(get_inputs, model, data):
    import w5d1_solutions

    module = model.layers[1].linear2

    expected = w5d1_solutions.get_inputs(model, data, module)
    actual = get_inputs(model, data, module)

    t.testing.assert_close(actual, expected)


def test_get_outputs(get_outputs, model, data):
    import w5d1_solutions

    module = model.layers[1].linear2

    expected = w5d1_solutions.get_outputs(model, data, module)
    actual = get_outputs(model, data, module)

    t.testing.assert_close(actual, expected)


def test_get_out_by_head(get_out_by_head, model, data):
    import w5d1_solutions

    layer = 2

    expected = w5d1_solutions.get_out_by_head(model, data, layer)
    actual = get_out_by_head(model, data, layer)

    t.testing.assert_close(actual, expected)


def test_get_out_by_component(get_out_by_components, model, data):
    import w5d1_solutions

    expected = w5d1_solutions.get_out_by_components(model, data)
    actual = get_out_by_components(model, data)

    t.testing.assert_close_atol(actual, expected, 1e-4)


def test_final_ln_fit(model, data, get_ln_fit):
    import w5d1_solutions

    expected, exp_r2 = w5d1_solutions.get_ln_fit(model, data, model.norm, 0)
    actual, act_r2 = get_ln_fit(model, data, model.norm, 0)

    t.testing.assert_close(t.tensor(actual.coef_), t.tensor(expected.coef_))
    t.testing.assert_close(t.tensor(actual.intercept_), t.tensor(expected.intercept_))
    t.testing.assert_close(act_r2, exp_r2)


def test_pre_final_ln_dir(model, data, get_pre_final_ln_dir):
    import w5d1_solutions

    expected = w5d1_solutions.get_pre_final_ln_dir(model, data)
    actual = get_pre_final_ln_dir(model, data)
    similarity = t.nn.functional.cosine_similarity(actual, expected, dim=0).item()
    t.testing.assert_close(similarity, 1.0)


def test_get_WV(model, get_WV):
    import w5d1_solutions

    indices = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    for layer, head in indices:
        v = w5d1_solutions.get_WV(model, layer, head)
        their_v = get_WV(model, layer, head)
        t.testing.assert_close(their_v, v)


def test_get_WO(model, get_WO):
    import w5d1_solutions

    indices = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    for layer, head in indices:
        o = w5d1_solutions.get_WO(model, layer, head)
        their_o = get_WO(model, layer, head)
        t.testing.assert_close(their_o, o)


def test_get_pre_20_dir(model, data, get_pre_20_dir):
    import w5d1_solutions

    expected = w5d1_solutions.get_pre_20_dir(model, data)
    actual = get_pre_20_dir(model, data)

    t.testing.assert_close(actual, expected)


def embedding_test(model, tokenizer, embedding_fn):
    import w5d1_solutions

    open_encoding = w5d1_solutions.embedding(model, tokenizer, "(")
    closed_encoding = w5d1_solutions.embedding(model, tokenizer, ")")

    t.testing.assert_close(embedding_fn(model, tokenizer, "("), open_encoding)
    t.testing.assert_close(embedding_fn(model, tokenizer, ")"), closed_encoding)


def qk_test(model, their_get_q_and_k):
    import w5d1_solutions

    indices = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    for layer, head in indices:
        q, k = w5d1_solutions.get_Q_and_K(model, layer, head)
        their_q, their_k = their_get_q_and_k(model, layer, head)
        t.testing.assert_close(their_q, q)
        t.testing.assert_close(their_k, k)


def test_qk_calc_termwise(model, tokenizer, their_get_q_and_k):
    import w5d1_solutions

    embedding = model.encoder(tokenizer.tokenize(["()()()()"]).to(w5d1_solutions.DEVICE)).squeeze()
    expected = w5d1_solutions.qk_calc_termwise(model, 0, 0, embedding, embedding)
    actual = their_get_q_and_k(model, 0, 0, embedding, embedding)

    t.testing.assert_close(actual, expected)