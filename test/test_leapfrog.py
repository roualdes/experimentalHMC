import pytest

import numpy as np

import experimentalhmc as ehmc

def ldg(q, g):
    np.multiply(-1, q, out = g)
    return 0.5 * np.dot(q, q)

# borrowed from Stan
# https://github.com/stan-dev/stan/blob/develop/src/test/unit/mcmc/hmc/integrators/expl_leapfrog_test.cpp
def test_leapfrog_01():

    q = np.asarray([1.99987371079118])
    p = np.asarray([-1.58612292129732])
    g = np.empty_like(q)
    ldg(q, g)

    ld = ehmc.leapfrog(q, p, 0.1, 1, g, ldg)

    assert np.isclose(q, 1.83126205010749)
    assert np.isclose(p, -1.77767970934226)
    assert np.isclose(g, -1.83126205010749)
    assert np.isclose(ld, 1.67676034808195)


def test_leapfrog_02():

    q = np.asarray([1.99987371079118])
    p = np.asarray([0.531143888645192])
    g = np.empty_like(q)
    ldg(q, g)

    ld = ehmc.leapfrog(q, p, 0.2, 1, g, ldg)

    assert np.isclose(q, 2.06610501430439)
    assert np.isclose(p, 0.124546016135635)
    assert np.isclose(g, -2.06610501430439)
    assert np.isclose(ld, 2.13439496506688)


def test_leapfrog_04():

    q = np.asarray([1.99987371079118])
    p = np.asarray([-1.01150787313287])
    g = np.empty_like(q)
    ldg(q, g)

    ld = ehmc.leapfrog(q, p, 0.4, 1, g, ldg)

    assert np.isclose(q, 1.43528066467474)
    assert np.isclose(p, -1.69853874822605)
    assert np.isclose(g, -1.43528066467474)
    assert np.isclose(ld, 1.03001529319458)


def test_leapfrog_05():

    q = np.asarray([1.99987371079118])
    p = np.asarray([-0.141638464197442])
    g = np.empty_like(q)
    ldg(q, g)

    ld = ehmc.leapfrog(q, p, 0.8, 1, g, ldg)

    assert np.isclose(q, 1.24660335198005)
    assert np.isclose(p, -1.44022928930593)
    assert np.isclose(g, -1.24660335198005)
    assert np.isclose(ld, 0.777009958583946)


def test_leapfrog_06():

    q = np.asarray([1.99987371079118])
    p = np.asarray([0.0942249427134016])
    g = np.empty_like(q)
    ldg(q, g)

    ld = ehmc.leapfrog(q, p, 1.6, 1, g, ldg)

    assert np.isclose(q, -0.409204730680088)
    assert np.isclose(p, -1.17831024137547)
    assert np.isclose(g, 0.409204730680088)
    assert np.isclose(ld, 0.0837242558054816)


def test_leapfrog_07():

    q = np.asarray([1.99987371079118])
    p = np.asarray([1.01936184962275])
    g = np.empty_like(q)
    ldg(q, g)

    ld = ehmc.leapfrog(q, p, 3.2, 1, g, ldg)

    assert np.isclose(q, -4.97752176966686)
    assert np.isclose(p, 5.78359874382383)
    assert np.isclose(g, 4.97752176966686)
    assert np.isclose(ld, 12.3878614837537)


def test_leapfrog_08():

    q = np.asarray([1.99987371079118])
    p = np.asarray([-2.73131279771964])
    g = np.empty_like(q)
    ldg(q, g)

    ld = ehmc.leapfrog(q, p, -1, 1, g, ldg)

    assert np.isclose(q, 3.73124965311523)
    assert np.isclose(p, 0.134248884233563)
    assert np.isclose(g, -3.73124965311523)
    assert np.isclose(ld, 6.96111198693627)


def test_leapfrog_09():
    # simulating diag_e_metric
    s = np.sqrt(0.733184698671436)

    q = np.asarray([1.27097196280777])
    p = np.asarray([-0.159996782671291 * s])
    g = np.empty_like(q)
    ldg(q, g)

    ld = ehmc.leapfrog(q, p, 2.40769920051673 * s, 1, g, ldg)

    assert np.isclose(q, -1.71246374711032)
    assert np.isclose(p, 0.371492925378682 * s)
    assert np.isclose(g, 1.71246374711032)
    assert np.isclose(ld, 1.46626604258356)
