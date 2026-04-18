import csv
from pathlib import Path

import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return nn.functional.relu(x)
    # return nn.functional.relu(x)
    # return torch.nn.functional.tanh(x)


def clamp01_grad(x: torch.Tensor) -> torch.Tensor:
    # Keep this derivative in sync with clamp01 above.
    return (x > 0).to(x.dtype)
    # return (x > 0).to(x.dtype)
    # y = torch.tanh(x)
    # return 1.0 - y.pow(2)


class ThreeQ(nn.Module):
    def __init__(
        self,
        n: int,
        n_clamped: int,
        Lg: float,
        seed: int,
        use_xavier: bool,
        use_detach: bool = True,
        n_tracked: int = 0,
        device=device,
    ) -> None:
        super().__init__()
        self.n = n
        self.n_clamped = n_clamped
        self.Lg = Lg
        self.seed = seed
        self.device = device
        self.use_detach = use_detach
        self.n_tracked = n_tracked

        # 设置随机种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.u = torch.randn((n,), device=self.device).requires_grad_()
        if use_xavier:
            self.w = torch.empty((n, n), device=self.device)
            nn.init.xavier_uniform_(self.w)
        else:
            self.w = torch.randn((n, n), device=self.device)
        self.w.fill_diagonal_(0.0)

        if self.n_clamped < 0 or self.n_clamped > self.n:
            raise ValueError("n_clamped must be in [0, n].")
        if self.n_tracked < 0 or self.n_tracked > (self.n - self.n_clamped):
            raise ValueError("n_tracked must be in [0, n - n_clamped].")
        if self.n_clamped > 0:
            self.register_buffer("u_clamped", self.u.detach()[: self.n_clamped].clone())
            self._apply_clamp()
        else:
            self.register_buffer("u_clamped", torch.empty(0, device=self.device))

    def _apply_clamp(self) -> None:
        if self.n_clamped <= 0:
            return
        with torch.no_grad():
            self.u[: self.n_clamped] = self.u_clamped

    def compute_rho(self) -> float:
        m = self.n - self.n_clamped
        if m <= 0:
            raise ValueError("compute_rho requires at least one non-clamped node.")

        self._apply_clamp()
        u_free = self.u[self.n_clamped :].detach()
        W_ff = self.w[self.n_clamped :, self.n_clamped :].detach()

        x = self.Lg * u_free
        g_prime = clamp01_grad(x) * self.Lg
        if g_prime.shape != (m,):
            raise RuntimeError(f"Unexpected clamp01_grad shape: {tuple(g_prime.shape)}")

        J_ff = (1.0 / (self.n - 1)) * (W_ff * g_prime)
        eigvals = torch.linalg.eigvals(J_ff)
        rho = torch.max(torch.abs(eigvals)).real
        return float(rho.cpu())

    def energy(self) -> torch.Tensor:
        source_u = self.u.detach() if self.use_detach else self.u
        pred = (1.0 / (self.n - 1)) * (self.w @ clamp01(self.Lg * source_u))
        diff = self.u - pred
        E = diff.pow(2).sum()

        return E

    def inference(
        self,
        num_epochs: int,
        lr: float,
        log_path: str | Path | None = None,
        log_every: int = 1,
    ) -> list[tuple[int, float]]:
        optimizer = torch.optim.SGD([self.u], lr=lr)
        history: list[tuple[int, float]] = []

        writer = None
        log_file = None
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", newline="")
            writer = csv.writer(log_file)
            tracked_indices = list(
                range(self.n_clamped, self.n_clamped + self.n_tracked)
            )
            tracked_headers = [f"u_{idx}" for idx in tracked_indices]
            writer.writerow(["epoch", "E", "delta", "rho", *tracked_headers])
            writer.writerow(["# n", self.n])
            writer.writerow(["# n_clamped", int(self.n_clamped)])
            writer.writerow(["# n_tracked", int(self.n_tracked)])
            writer.writerow(["# Lg", float(self.Lg)])
            writer.writerow(["# seed", int(self.seed)])
            writer.writerow(["# lr", float(lr)])
            writer.writerow(["# num_epochs", int(num_epochs)])
            writer.writerow(["# log_every", int(log_every)])
        else:
            tracked_indices = []

        for epoch in range(1, num_epochs + 1):
            self._apply_clamp()
            E = self.energy()
            e_value = float(E.detach().cpu())
            u_prev = self.u.detach().clone()
            if epoch % log_every == 0:
                history.append((epoch, e_value))
            optimizer.zero_grad()
            E.backward()
            if self.n_clamped > 0 and self.u.grad is not None:
                self.u.grad[: self.n_clamped] = 0.0
            optimizer.step()
            self._apply_clamp()
            delta = float((self.u.detach() - u_prev).abs().sum().cpu())
            if epoch % log_every == 0 and writer is not None:
                rho_value = self.compute_rho()
                tracked_values = [
                    float(self.u[idx].detach().cpu()) for idx in tracked_indices
                ]
                writer.writerow([epoch, e_value, delta, rho_value, *tracked_values])

        if log_file is not None:
            log_file.close()

        return history


def plot_energy(
    log_path: str | Path,
    out_path: str | Path | None = None,
    tracked_out_path: str | Path | None = None,
    show: bool = False,
) -> None:
    log_path = Path(log_path)
    epochs: list[int] = []
    energies: list[float] = []
    deltas: list[float] = []
    rhos: list[float] = []
    tracked_names: list[str] = []
    tracked_values: dict[str, list[float]] = {}
    meta: dict[str, str] = {}

    with log_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rho_idx = None
        if header is not None and len(header) > 3:
            columns = [item.strip() for item in header if item is not None]
            if "rho" in columns:
                rho_idx = columns.index("rho")
            tracked_start = 3
            if rho_idx is not None:
                tracked_start = 4
            tracked_names = [
                item.strip() for item in columns[tracked_start:] if item.strip()
            ]
            tracked_values = {name: [] for name in tracked_names}
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                key = row[0].lstrip("#").strip()
                value = row[1].strip() if len(row) > 1 else ""
                meta[key] = value
                continue
            epochs.append(int(row[0]))
            energies.append(float(row[1]))
            if len(row) >= 3 and row[2] != "":
                deltas.append(float(row[2]))
            if rho_idx is not None and len(row) > rho_idx and row[rho_idx] != "":
                rhos.append(float(row[rho_idx]))
            for idx, name in enumerate(
                tracked_names, start=(4 if rho_idx is not None else 3)
            ):
                if len(row) > idx and row[idx] != "":
                    tracked_values[name].append(float(row[idx]))
                else:
                    tracked_values[name].append(float("nan"))

    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def build_title(prefix: str) -> str:
        if not meta:
            return prefix
        bits = []
        for key in ("n", "n_clamped", "n_tracked", "Lg", "seed", "rho", "lr"):
            if key in meta:
                bits.append(f"{key}={meta[key]}")
        if bits:
            return f"{prefix} ({', '.join(bits)})"
        return prefix

    has_tracked = bool(tracked_names) and all(
        len(values) == len(epochs) for values in tracked_values.values()
    )

    # E plot
    fig_e, ax_e = plt.subplots(figsize=(6, 4))
    ax_e.plot(epochs, energies, linewidth=1.5, color="#1f77b4")
    ax_e.set_xlabel("epoch")
    ax_e.set_ylabel("E")
    ax_e.set_title(build_title("E"))
    ax_e.grid(True, alpha=0.3)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig_e.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig_e)

    # delta plot
    if deltas and len(deltas) == len(epochs):
        fig_d, ax_d = plt.subplots(figsize=(6, 4))
        ax_d.plot(epochs, deltas, linewidth=1.2, color="#ff7f0e")
        ax_d.set_xlabel("epoch")
        ax_d.set_ylabel("delta")
        ax_d.set_title(build_title("delta"))
        ax_d.grid(True, alpha=0.3)
        if out_path is not None:
            delta_path = Path(out_path)
            delta_path = delta_path.with_name(
                f"{delta_path.stem}_delta{delta_path.suffix}"
            )
            delta_path.parent.mkdir(parents=True, exist_ok=True)
            fig_d.savefig(delta_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig_d)

    # rho plot
    if rhos and len(rhos) == len(epochs):
        fig_r, ax_r = plt.subplots(figsize=(6, 4))
        ax_r.plot(epochs, rhos, linewidth=1.2, color="#2ca02c")
        ax_r.set_xlabel("epoch")
        ax_r.set_ylabel("rho")
        ax_r.set_title(build_title("rho"))
        ax_r.grid(True, alpha=0.3)
        if out_path is not None:
            rho_path = Path(out_path)
            rho_path = rho_path.with_name(f"{rho_path.stem}_rho{rho_path.suffix}")
            rho_path.parent.mkdir(parents=True, exist_ok=True)
            fig_r.savefig(rho_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig_r)

    if has_tracked:
        fig_u, ax_u = plt.subplots(figsize=(6, 4))
        for name in tracked_names:
            ax_u.plot(epochs, tracked_values[name], linewidth=1.0, alpha=0.8)
        ax_u.set_xlabel("epoch")
        ax_u.set_ylabel("u")
        title_u = "tracked u"
        if meta:
            bits = []
            for key in ("n", "n_clamped", "n_tracked", "Lg", "seed", "lr"):
                if key in meta:
                    bits.append(f"{key}={meta[key]}")
            if bits:
                title_u = f"{title_u} ({', '.join(bits)})"
        ax_u.set_title(title_u)
        ax_u.grid(True, alpha=0.3)

        if tracked_out_path is None and out_path is not None:
            out_path_obj = Path(out_path)
            tracked_out_path = out_path_obj.with_name(
                f"{out_path_obj.stem}_tracked{out_path_obj.suffix}"
            )
        if tracked_out_path is not None:
            tracked_out_path = Path(tracked_out_path)
            tracked_out_path.parent.mkdir(parents=True, exist_ok=True)
            fig_u.savefig(tracked_out_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig_u)
