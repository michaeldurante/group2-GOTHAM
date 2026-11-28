from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    """Maps latent vector (nz, 1, 1) -> square image (nc, 44, 44) with reduced checkerboard artifacts."""

    def __init__(self, nz: int, ngf: int, nc: int):
        super().__init__()
        self.main = nn.Sequential(
            # (nz) x 1 x 1 -> (ngf*8) x 3 x 3
            nn.ConvTranspose2d(nz, ngf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf*8) x 3 x 3 -> (ngf*4) x 5 x 5
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4) x 5 x 5 -> (ngf*2) x 11 x 11
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Upsample + Conv2d to avoid 4x4 stride-2 checkerboard noise -> 22 x 22
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Upsample + Conv2d -> 44 x 44
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)


class Discriminator(nn.Module):
    """Classifies square (nc, 44, 44) light-curve representations."""

    def __init__(self, nc: int, ndf: int):
        super().__init__()
        self.main = nn.Sequential(
            # (nc) x 44 x 44 -> (ndf) x 22 x 22
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 22 x 22 -> (ndf*2) x 11 x 11
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 11 x 11 -> (ndf*4) x 5 x 5
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 5 x 5 -> (ndf*8) x 3 x 3
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 3 x 3 -> 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, 3, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class GenerativeAdversarialNetwork:
    def __init__(self, p_dictionary: dict):
        self.dataset_path = p_dictionary.get("dataset_path")
        self.batch_size = p_dictionary.get("batch_size", 32)
        self.image_size = p_dictionary.get("image_size", 44)
        self.nc = p_dictionary.get("nc", 1)
        self.nz = p_dictionary.get("nz", 100)
        self.ngf = p_dictionary.get("ngf", 64)
        self.ndf = p_dictionary.get("ndf", 32)
        self.num_epochs = p_dictionary.get("num_epochs", 150)
        self.lr = p_dictionary.get("lr", 0.0002)
        self.beta1 = p_dictionary.get("beta1", 0.5)
        self.workers = p_dictionary.get("workers", 2)
        self.num_samples = p_dictionary.get("num_samples", 4)
        self.smooth_window = p_dictionary.get("smooth_window", 5)
        self.tv_weight = p_dictionary.get("tv_weight", 0.15)
        self.seed = p_dictionary.get("seed", 999)
        self.output_dir = p_dictionary.get("output_dir", "outputs/dream_gan")
        self.plot_window = p_dictionary.get("plot_window", 3)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.create_dataset()

        self.netG = Generator(self.nz, self.ngf, self.nc).to(self.device)
        self.netG.apply(weights_init)

        self.netD = Discriminator(self.nc, self.ndf).to(self.device)
        self.netD.apply(weights_init)

    @staticmethod
    def _moving_average_rows(arr: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return arr
        kernel = np.ones(window, dtype=np.float64) / window
        pad = window // 2
        padded = np.pad(arr, ((0, 0), (pad, pad)), mode="edge")
        return np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, padded)[:, : arr.shape[1]]

    def create_dataset(self):
        """Truncates 2000-sample light curves to 1936 points (968 flux + 968 err), denoises lightly, and reshapes to 44x44."""
        n_expected = self.image_size * self.image_size  # 1936
        half_len = n_expected // 2  # 968

        df = pd.read_csv(self.dataset_path)
        if "label" in df.columns:
            df = df[df["label"] == 1]

        meta_cols = ["toi_name", "tic", "label", "disp", "period_d", "t0_bjd", "dur_hr", "sector"]
        raw_data = df.drop(columns=meta_cols, errors="ignore").values

        flux = np.nan_to_num(
            raw_data[:, 0:half_len].astype(np.float64),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )
        err = np.nan_to_num(
            raw_data[:, 1000 : 1000 + half_len].astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # flux = raw_data[:, 0:half_len].astype(np.float64)
        # err = raw_data[:, 1000 : 1000 + half_len].astype(np.float64)

        # Anchor to baseline and apply light rolling smoothing so transit dips stand out over cadence white noise
        baseline = np.median(flux, axis=1, keepdims=True)
        baseline = np.where(np.abs(baseline) < 1e-8, 1.0, baseline)
        flux_rel = self._moving_average_rows(flux / baseline, self.smooth_window)
        err_rel = np.maximum(err / baseline, 0.0)

        err_rel = np.maximum(err / baseline, 0.0)
        # Blocca gli outlier estremi prima di calcolare i percentili
        err_cap = float(np.percentile(err_rel, 99.0))
        if err_cap <= 0.0:
            err_cap = 1e-3
        err_rel_clipped = np.clip(err_rel, 0.0, err_cap)

        self.flux_rel_low = float(np.percentile(flux_rel, 0.1))
        self.flux_rel_high = float(np.percentile(flux_rel, 99.9))
        if self.flux_rel_high <= self.flux_rel_low:
            self.flux_rel_high = self.flux_rel_low + 1e-4

        self.err_rel_low = float(np.percentile(err_rel_clipped, 1.0))
        self.err_rel_high = float(np.percentile(err_rel_clipped, 99.0))
        if self.err_rel_high <= self.err_rel_low:
            self.err_rel_high = self.err_rel_low + 1e-4

        '''DEBUG'''
        print(
            f"[DEBUG dataset] err_rel min={np.min(err_rel):.6f}, "
            f"median={np.median(err_rel):.6f}, p99={np.percentile(err_rel, 99):.6f}, "
            f"max={np.max(err_rel):.6f}"
        )
        print(
            f"[DEBUG dataset] err_rel_low={self.err_rel_low:.6f}, "
            f"err_rel_high={self.err_rel_high:.6f} | "
            f"flux_rel_low={self.flux_rel_low:.6f}, flux_rel_high={self.flux_rel_high:.6f}"
        )

        # self.flux_rel_low = float(np.percentile(flux_rel, 0.05))
        # self.flux_rel_high = float(np.percentile(flux_rel, 99.95))
        # if self.flux_rel_high <= self.flux_rel_low:
        #     self.flux_rel_high = self.flux_rel_low + 1e-4

        # self.err_rel_low = float(np.percentile(err_rel, 0.5))
        # self.err_rel_high = float(np.percentile(err_rel, 99.95))
        # if self.err_rel_high <= self.err_rel_low:
        #     self.err_rel_high = self.err_rel_low + 1e-4

        flux_norm = np.clip(
            2.0 * (flux_rel - self.flux_rel_low) / (self.flux_rel_high - self.flux_rel_low) - 1.0,
            -1.0,
            1.0,
        )
        err_norm = np.clip(
            2.0 * (err_rel - self.err_rel_low) / (self.err_rel_high - self.err_rel_low) - 1.0,
            -1.0,
            1.0,
        )

        curves_1936 = np.concatenate([flux_norm, err_norm], axis=1)[:, :n_expected].astype(np.float32)
        tensor_data = torch.from_numpy(curves_1936).reshape(
            -1, self.nc, self.image_size, self.image_size
        )

        self.dataset = TensorDataset(tensor_data)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )

    def train(self):
        criterion = nn.BCELoss()
        optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        real_label = 0.9
        fake_label = 0.0
        half_len = (self.image_size * self.image_size) // 2  # 968

        self.G_losses = []
        self.D_losses = []

        self.netG.train()
        self.netD.train()

        print(f"Starting 44x44 DCGAN training on {self.device} for {self.num_epochs} epochs...")
        for epoch in range(self.num_epochs):
            for data in self.dataloader:
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)

                # (1) Update Discriminator
                self.netD.zero_grad()
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                output = self.netD(real_cpu).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                fake = self.netG(noise)
                label.fill_(fake_label)
                output = self.netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()

                errD = errD_real + errD_fake
                optimizerD.step()

                # (2) Update Generator + 1D temporal smoothness matching on flux segment
                self.netG.zero_grad()
                label.fill_(real_label)
                output = self.netD(fake).view(-1)
                errG_adv = criterion(output, label)

                fake_flux_1d = fake.view(b_size, -1)[:, :half_len]
                real_flux_1d = real_cpu.view(b_size, -1)[:, :half_len]
                fake_diff = torch.mean(torch.abs(fake_flux_1d[:, 1:] - fake_flux_1d[:, :-1]))
                real_diff = torch.mean(torch.abs(real_flux_1d[:, 1:] - real_flux_1d[:, :-1]))
                errG_smooth = torch.abs(fake_diff - real_diff)

                errG = errG_adv + self.tv_weight * errG_smooth
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"[{epoch + 1}/{self.num_epochs}] "
                    f"Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f} | "
                    f"D(x): {D_x:.4f} | D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
                )
            
            '''print synthetic light curves at specific epochs for visual inspection'''
            if (epoch + 1) in (50, 100, 200, 300, 450, 600, 780, 800, 1050, 1540, 1940, 2890):
                self.generate_and_plot(
                    num_samples=self.num_samples,
                    window=getattr(self, "plot_window", 3),
                    output_path=str(Path(self.output_dir) / f"synthetic_epoch_{epoch + 1}.png"),
                )
                self.netG.train()
    def generate_and_plot(
        self,
        num_samples: int = None,
        window: int = 7,
        output_path: str = "synthetic_light_curves.png",
    ):
        """Reconstructs 1D flux and error arrays from 44x44 outputs and plots them."""
        n = num_samples if num_samples is not None else self.num_samples
        half_len = (self.image_size * self.image_size) // 2  # 968

        self.netG.eval()
        with torch.no_grad():
            noise = torch.randn(n, self.nz, 1, 1, device=self.device)
            fake_44 = self.netG(noise).cpu().numpy()

        f_low = getattr(self, "flux_rel_low", 0.98)
        f_high = getattr(self, "flux_rel_high", 1.02)
        e_low = getattr(self, "err_rel_low", 0.0005)
        e_high = getattr(self, "err_rel_high", 0.003)

        time_steps = np.arange(half_len)
        fig, axes = plt.subplots(n, 1, figsize=(11, 2.8 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for idx in range(n):
            flat_1936 = fake_44[idx].reshape(-1)
            flux_norm = flat_1936[:half_len]
            err_norm = flat_1936[half_len : 2 * half_len]

            flux_rel = 0.5 * (flux_norm + 1.0) * (f_high - f_low) + f_low
            raw_err_rel = 0.5 * (err_norm + 1.0) * (e_high - e_low) + e_low
            max_sensible_err = max(e_high * 1.5, 0.05)
            err_rel = np.clip(raw_err_rel, max(e_low, 0.0), max_sensible_err)
            smooth_flux = pd.Series(flux_rel).rolling(window=window, center=True, min_periods=1).mean()

            '''DEBUG'''
            transit_depth = 1.0 - float(np.min(smooth_flux))
            print(
                f"[DEBUG sample #{idx + 1}] err_rel min={np.min(err_rel):.6f}, "
                f"mean={np.mean(err_rel):.6f}, max={np.max(err_rel):.6f} | "
                f"transit_depth={transit_depth:.6f}"
            )

            # flux_rel = 0.5 * (flux_norm + 1.0) * (f_high - f_low) + f_low
            # err_rel = np.clip(0.5 * (err_norm + 1.0) * (e_high - e_low) + e_low, 0.0, None)
            # smooth_flux = pd.Series(flux_rel).rolling(window=window, center=True, min_periods=1).mean()

            axes[idx].fill_between(
                time_steps,
                flux_rel - err_rel,
                flux_rel + err_rel,
                color="tab:blue",
                alpha=0.20,
                label="±1σ uncertainty",
            )
            axes[idx].plot(
                time_steps,
                flux_rel,
                color="tab:blue",
                lw=0.8,
                alpha=0.85,
                label="Relative flux",
            )
            axes[idx].plot(
                time_steps,
                smooth_flux,
                color="crimson",
                lw=1.6,
                label=f"Rolling mean ({window} pts)",
            )
            axes[idx].axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.5)
            axes[idx].set_ylabel("Relative Flux")
            axes[idx].set_title(f"Synthetic 1D Light Curve #{idx + 1} (from 44x44)")
            axes[idx].grid(True, alpha=0.25)

        axes[0].legend(loc="upper right", ncol=3, fontsize=8.5)
        axes[-1].set_xlabel(f"Time Cadence (0–{half_len - 1})")

        plt.tight_layout()
        out_file = Path(output_path)
        plt.savefig(out_file, dpi=150)
        plt.close(fig)
        print(f"Saved reconstructed 1D light curves to: {out_file.resolve()}")
        print()