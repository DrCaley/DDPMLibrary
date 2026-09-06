"""The per-draw seeds of neighbouring cases must not overlap.

Benchmarks score case ``i`` with ``seed = BASE + i``, so a sampler that seeded
draw ``k`` with ``base + k`` gave case ``i`` draw ``k`` and case ``i+1`` draw
``k-1`` the same noise. That correlates the cases and leaks noise across the
split-conformal fit/verify boundary, which is exactly what the held-out coverage
number is supposed to rule out.
"""
import pytest

from ddpm_library.inference import draw_seed


MAX_ENSEMBLE = 64      # far above any n_draws the library ships


def test_blocks_of_consecutive_base_seeds_are_disjoint():
    base = 20260830
    for n in (1, 10, 20, MAX_ENSEMBLE):
        seen = {}
        for case in range(40):                       # a full benchmark
            for k in range(n):
                s = draw_seed(base + case, k)
                assert s not in seen, (
                    f"n_draws={n}: case {case} draw {k} collides with "
                    f"case {seen[s][0]} draw {seen[s][1]}")
                seen[s] = (case, k)


def test_draw_k_does_not_depend_on_ensemble_size():
    # the n_draws sweep relies on draw k being the same field at every size
    assert draw_seed(7, 3) == draw_seed(7, 3)
    assert len({draw_seed(7, k) for k in range(20)}) == 20


def test_every_sampler_uses_the_shared_helper():
    import ddpm_library.distattn_predict as da
    import ddpm_library.repaint_predict as rp
    import ddpm_library.stream.sampler as ss
    for mod in (da, rp, ss):
        assert mod.draw_seed is draw_seed, f"{mod.__name__} has its own seed scheme"


@pytest.mark.parametrize("bad", [lambda b, k: b + k])
def test_the_naive_scheme_would_fail_this_test(bad):
    # guards the test itself: the scheme this replaced must collide
    seen = set(); collided = False
    for case in range(40):
        for k in range(20):
            s = bad(20260830 + case, k)
            collided |= s in seen
            seen.add(s)
    assert collided
