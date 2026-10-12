import pytest



@pytest.fixture(scope="session")
def small_task():
    pytest.importorskip("torch_geometric")
    from core.dataset.synthetic_events import SyntheticConfig, generate_events
    from core.dataset.temporal_graph import PRMergeTaskConfig, build_pr_merge_task

    ev = generate_events(SyntheticConfig(n_users=600, n_repos=60, days=16, seed=2))
    cfg = PRMergeTaskConfig(train_start="2025-09-05", train_end="2025-09-09", val_start="2025-09-11",
                            val_end="2025-09-12", test_start="2025-09-14", test_end="2025-09-15")
    return build_pr_merge_task(ev, cfg)
