from adaptiveiv import (
    AdaptiveIV,
    fit_fully_interacted_2sls,
    fit_pooled_2sls,
    simulate_paper_dgp,
)


def main() -> None:
    data = simulate_paper_dgp(
        n_groups=20,
        n_per_group=150,
        beta=0.5,
        strong_fraction=0.3,
        weak_fraction=0.2,
        rho_uv=0.4,
        seed=321,
    )

    results = AdaptiveIV.from_formula(
        "Y ~ 1 + X + [W ~ Z]",
        data=data,
        groups="group",
    ).fit(random_state=99)
    pooled = fit_pooled_2sls(data, "Y", "W", "Z", ["X"])
    interacted = fit_fully_interacted_2sls(data, "Y", "W", "Z", ["X"], "group")

    print(f"Adaptive estimate: {results.params['W']:.4f}")
    print(f"Pooled 2SLS estimate: {pooled.beta:.4f}")
    print(f"Fully interacted estimate: {interacted.beta:.4f}")
    print(f"Selected groups: {results.selection_summary['selected_total']}")
    print(results.group_diagnostics.head().to_string(index=False))


if __name__ == "__main__":
    main()

