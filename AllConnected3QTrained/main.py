from ThreeQ import ThreeQ


def main_experience(
    n,
    n_clamped,
    Lg,
    n_tracked=0,
    seed=0,
    lr=0.01,
    num_epochs=100,
    inf_num_epochs=None,
    inf_lr=0.1,
    use_xavier=True,
    use_detach=True,
    log_every=1,
    inf_log_every=1,
):
    name = f"n_{n},n_clamped_{n_clamped},n_tracked_{n_tracked},Lg_{Lg},seed_{seed}"
    path = f"n_{n},n_clamped_{n_clamped},n_tracked_{n_tracked},seed_{seed}"
    if use_xavier:
        name = "xavier_" + name
        path = "xavier_" + path
    if not use_detach:
        name = "nodetach_" + name
        path = "nodetach_" + path

    model = ThreeQ(
        n=n,
        n_clamped=n_clamped,
        Lg=Lg,
        seed=seed,
        use_xavier=use_xavier,
        use_detach=use_detach,
        n_tracked=n_tracked,
    )
    if inf_num_epochs is None:
        inf_num_epochs = num_epochs
    if inf_lr is None:
        inf_lr = lr

    model.fix(
        num_epochs=num_epochs,
        lr=lr,
        inf_num_epochs=inf_num_epochs,
        inf_lr=inf_lr,
        log_every=log_every,
        inf_log_every=inf_log_every,
        log_path=f"data/{path}/{name}_train.csv",
        inf_log_dir=f"data/{path}/{name}_inf",
        plot_path=f"png/{path}/{name}_train.png",
        inf_plot_dir=f"png/{path}/{name}_inf",
    )


if __name__ == "__main__":
    seed = 0

    # 30节点实验：不同钳制比例 + Lg扫描
    exp_30_base = dict(
        n=30,
        n_tracked=10,
        seed=seed,
        lr=0.1,
        num_epochs=100,
        inf_num_epochs=100,
        use_xavier=False,
    )
    lg_values_30 = (1, 2, 3, 4, 5, 6, 7, 8)
    for n_clamped in (10, 20):
        for Lg in lg_values_30:
            log_every = max(1, exp_30_base["num_epochs"] // 100)
            inf_log_every = max(1, exp_30_base["inf_num_epochs"] // 10)
            main_experience(
                **exp_30_base,
                n_clamped=n_clamped,
                Lg=Lg,
                log_every=log_every,
                inf_log_every=inf_log_every,
            )

    # 500节点实验：不同钳制比例 + 高Lg扫描
    # exp_500_base = dict(
    #     n=500,
    #     n_tracked=10,
    #     seed=seed,
    #     lr=0.1,
    #     num_epochs=100,
    #     inf_num_epochs=100,
    #     use_xavier=False,
    # )
    # lg_values_500 = (30, 40, 50, 60, 70, 80)
    # for n_clamped in (100, 200, 300):
    #     for Lg in lg_values_500:
    #         log_every = max(1, exp_500_base["num_epochs"] // 100)
    #         inf_log_every = max(1, exp_500_base["inf_num_epochs"] // 10)
    #         main_experience(
    #             **exp_500_base,
    #             n_clamped=n_clamped,
    #             Lg=Lg,
    #             log_every=log_every,
    #             inf_log_every=inf_log_every,
    #         )
