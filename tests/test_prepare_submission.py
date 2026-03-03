from types import SimpleNamespace

from tau2.scripts.leaderboard.prepare_submission import validate_submission_traj_set


def _make_results(domain: str, agent_backend: str, user_backend: str):
    return SimpleNamespace(
        info=SimpleNamespace(
            environment_info=SimpleNamespace(domain_name=domain),
            agent_info=SimpleNamespace(
                llm="agent-model",
                llm_backend=agent_backend,
                llm_args={"temperature": 0.0},
            ),
            user_info=SimpleNamespace(
                llm="user-model",
                llm_backend=user_backend,
                llm_args={"temperature": 0.0},
            ),
        )
    )


def test_validate_submission_traj_set_detects_backend_mismatch():
    results = [
        _make_results("airline", "litellm", "litellm"),
        _make_results("retail", "transformers", "litellm"),
    ]
    valid, error = validate_submission_traj_set(results)
    assert valid is False
    assert "Agent / User Simulator should be the same" in error
