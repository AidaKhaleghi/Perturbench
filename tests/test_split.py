from pertbench.data.preprocess import CONTROL, n_perturbed, parse_condition


def test_control_has_no_genes():
    assert parse_condition(CONTROL) == ()
    assert n_perturbed(CONTROL) == 0


def test_single_perturbation():
    assert parse_condition("AHR+ctrl") == ("AHR",)
    assert n_perturbed("AHR+ctrl") == 1


def test_double_is_order_invariant():
    assert parse_condition("KLF1+AHR") == parse_condition("AHR+KLF1")
    assert n_perturbed("AHR+KLF1") == 2
