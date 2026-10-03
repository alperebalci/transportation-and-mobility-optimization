import numpy as np

from wardrop_or import pigou_network


def test_pigou_user_equilibrium_and_vi_gap():
    network = pigou_network()
    ue = network.user_equilibrium()
    assert np.allclose(ue.path_flows, [1.0, 0.0], atol=1e-5)
    assert ue.wardrop_gap <= 1e-7
    assert np.isclose(ue.total_travel_time, 1.0, atol=1e-6)


def test_system_optimum_and_price_of_anarchy():
    network = pigou_network()
    ue = network.user_equilibrium()
    so = network.system_optimum()
    assert np.allclose(so.path_flows, [0.5, 0.5], atol=1e-5)
    assert np.isclose(so.total_travel_time, 0.75, atol=1e-6)
    assert np.isclose(ue.total_travel_time / so.total_travel_time, 4.0 / 3.0, atol=1e-5)


def test_pigouvian_toll_internalizes_external_cost():
    network = pigou_network()
    so = network.system_optimum()
    tolls = network.pigouvian_tolls(so.link_flows)
    generalized = so.link_costs + tolls
    used = so.path_flows > 1e-7
    assert np.allclose(generalized[used], generalized[used][0], atol=1e-6)
    assert np.all(tolls >= -1e-12)
