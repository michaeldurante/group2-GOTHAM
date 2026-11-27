import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import pandas as pd
from daneel.parameters import Parameters
from pathlib import Path


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.main = nn.Sequential(
            # input: (nz) x 1 x 1 -> (ngf*8) x 3 x 3
            nn.ConvTranspose2d(nz, ngf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 3 x 3 -> (ngf*4) x 5 x 5
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 5 x 5 -> (ngf*2) x 11 x 11
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 11 x 11 -> (ngf) x 22 x 22
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: (ngf) x 22 x 22 -> (nc) x 44 x 44
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.main = nn.Sequential(
            # input: (nc) x 44 x 44 -> (ndf) x 22 x 22
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf) x 22 x 22 -> (ndf*2) x 11 x 11
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 11 x 11 -> (ndf*4) x 5 x 5
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 5 x 5 -> (ndf*8) x 3 x 3
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 3 x 3 -> 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class GenerativeAdversarialNetwork:
    def __init__(self, p_dictionary):
        '''
        retrieve all parameter from dictionary and set seed for reproducibility
        '''
        # print('Start __init__')
        self.dataset_path = p_dictionary.get('dataset_path')
        self.batch_size = p_dictionary.get('batch_size', 128)  # how many lc before weight update
        self.image_size = p_dictionary.get('image_size', 44)    # size of image
        self.nc = p_dictionary.get('nc', 1)     # number of channels in the image (color:3, flux:1)
        self.nz = p_dictionary.get('nz', 100)   # size of generator input
        self.ngf = p_dictionary.get('ngf', 64)  # size of feature maps in generator
        self.ndf = p_dictionary.get('ndf', 64)  # size of feature maps in discriminator
        self.num_epochs = p_dictionary.get('num_epochs', 50)     # number of training epochs
        self.lr = p_dictionary.get('lr', 0.0002)    # learning rate for Adam optimizers
        self.beta1 = p_dictionary.get('beta1', 0.5) # hyperparameter for Adam optimizers
        self.workers = p_dictionary.get('workers', 2)
        self.num_samples = p_dictionary.get('num_samples', 4)

        # Set random seed for reproducibility
        random.seed(999)
        torch.manual_seed(999)
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        self.create_dataset()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.netG = Generator(self.nz, self.ngf, self.nc).to(self.device)
        self.netG.apply(weights_init)

        self.netD = Discriminator(self.nc, self.ndf).to(self.device)
        self.netD.apply(weights_init)

        # print('Tutto ha funzionato')


    def create_dataset(self):
        '''
        load data, select real transits, truncate light curves with errors,
        reshape and create dataset for GAN
        '''
        # print('Start create_dataset')
        # load data and select real transits
        df = pd.read_csv(self.dataset_path)
        if "label" in df.columns:
            df = df[df["label"] == 1]

        # remove metadata and select 1936 points (968 fluxes + error)
        meta_cols = ["toi_name", "tic", "label", "disp", "period_d", "t0_bjd", "dur_hr", "sector"]
        raw_data = df.drop(columns=meta_cols, errors="ignore").values  # shape: (N, 2000)

        flux = raw_data[:, 0:968]
        err = raw_data[:, 1000:1968]

        # Salva la scala fisica tipica per denormalizzare i plot
        self.flux_min = float(np.median(flux.min(axis=1)))
        self.flux_max = float(np.median(flux.max(axis=1)))
        self.err_min = float(np.median(err.min(axis=1)))
        self.err_max = float(np.median(err.max(axis=1)))

        # normalize to [-1,1] and concatenate data
        def scale_to_tanh(arr):
            a_min = arr.min(axis=1, keepdims=True)
            a_max = arr.max(axis=1, keepdims=True)
            return 2.0 * (arr - a_min) / (a_max - a_min + 1e-8) - 1.0

        flux_norm = scale_to_tanh(flux)

        ############################ *0.2 -> test per i pesi #########################
        err_norm = scale_to_tanh(err) * 0.2
        curves_norm = np.concatenate([flux_norm, err_norm], axis=1)

        # convert into torch tensor(N, 1, 44, 44) and create dataset
        tensor_data = torch.tensor(curves_norm, dtype=torch.float32).reshape(
            -1, self.nc, self.image_size, self.image_size
        )

        self.dataset = TensorDataset(tensor_data)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers
        )

    def train(self):
        '''
        Train the DCGAN on the 44x44 light-curve representations
        '''
        criterion = nn.BCELoss()
        optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        real_label = 0.9
        fake_label = 0.0

        self.G_losses = []
        self.D_losses = []

        self.netG.train()
        self.netD.train()

        print(f"Starting GAN training on {self.device} for {self.num_epochs} epochs...")
        for epoch in range(self.num_epochs):
            for i, data in enumerate(self.dataloader, 0):
                # (1) Aggiorna il Discriminatore: massimizza log(D(x)) + log(1 - D(G(z)))
                self.netD.zero_grad()
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
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

                # (2) Aggiorna il Generatore: massimizza log(D(G(z)))
                self.netG.zero_grad()
                label.fill_(real_label)

                output = self.netD(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

            # Stampa il riepilogo ogni 5 epoche (più pulito da terminale)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"[{epoch + 1}/{self.num_epochs}] "
                    f"Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f} | "
                    f"D(x): {D_x:.4f} | D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
                )

################# due funzioni per il plot per provare cose diverse #############
    def ggenerate_and_plot(self, output_path="synthetic_light_curves.png"):
        '''
        Generate synthetic 44x44 outputs, reconstruct 1D flux and error arrays,
        denormalize to physical scale, and plot scatter light curves with error bars.
        '''
        self.netG.eval()

        with torch.no_grad():
            noise = torch.randn(self.num_samples, self.nz, 1, 1, device=self.device)
            fake_2d = self.netG(noise).cpu().numpy()

        f_min = getattr(self, "flux_min", 0.98)
        f_max = getattr(self, "flux_max", 1.01)
        e_min = getattr(self, "err_min", 0.0005)
        e_max = getattr(self, "err_max", 0.003)

        time_steps = np.arange(968)
        fig, axes = plt.subplots(self.num_samples, 1, figsize=(11, 2.8 * self.num_samples), sharex=True)
        if self.num_samples == 1:
            axes = [axes]

        for idx in range(self.num_samples):
            flat_1d = fake_2d[idx].reshape(-1)
            flux_1d = flat_1d[:968]
            err_norm = flat_1d[968:]

            # Ricostruisce l'errore fisico e lo proietta sulle unità normalizzate del flusso
            err_phys = np.clip(0.5 * (err_norm + 1.0) * (e_max - e_min) + e_min, 0.0, None)
            err_1d = err_phys * (2.0 / (f_max - f_min))

            axes[idx].errorbar(
                time_steps,
                flux_1d,
                yerr=err_1d,
                fmt="o",
                markersize=2.5,
                elinewidth=0.6,
                capsize=0,
                alpha=0.75,
                color="tab:blue",
                ecolor="tab:gray",
            )
            axes[idx].set_ylabel("Normalized Flux [-1, 1]")
            axes[idx].set_title(f"Synthetic Transit Light Curve #{idx + 1}")
            axes[idx].grid(True, alpha=0.25)

        axes[-1].set_xlabel("Time Cadence (0–967)")

        plt.tight_layout()
        out_file = Path(output_path)
        plt.savefig(out_file, dpi=150)
        plt.close(fig)
        print(f"Saved reconstructed 1D light curves to: {out_file.resolve()}")
    
    def generate_and_plot2(self, output_path="synthetic_light_curves.png"):
        '''
        Generate synthetic 44x44 outputs, reconstruct 1D flux (in [-1, 1]) and
        proportionally scaled error bars, and plot scatter light curves with rolling mean.
        '''
        self.netG.eval()

        with torch.no_grad():
            noise = torch.randn(self.num_samples, self.nz, 1, 1, device=self.device)
            fake_2d = self.netG(noise).cpu().numpy()

        f_min = getattr(self, "flux_min", 0.98)
        f_max = getattr(self, "flux_max", 1.01)
        e_min = getattr(self, "err_min", 0.0005)
        e_max = getattr(self, "err_max", 0.003)

        time_steps = np.arange(968)
        fig, axes = plt.subplots(self.num_samples, 1, figsize=(11, 2.8 * self.num_samples), sharex=True)
        if self.num_samples == 1:
            axes = [axes]

        for idx in range(self.num_samples):
            flat_1d = fake_2d[idx].reshape(-1)

            # Il flusso resta direttamente in [-1, 1] (uscita della Tanh)
            flux_1d = flat_1d[:968]
            err_norm = flat_1d[968:]

            # Scala le barre d'errore affinché siano proporzionate al flusso in [-1, 1]
            err_phys = np.clip(0.5 * (err_norm + 1.0) * (e_max - e_min) + e_min, 0.0, None)
            err_1d = err_phys * (2.0 / (f_max - f_min))

            # Media mobile su 15 punti per evidenziare il profilo del transito
            smooth_flux = pd.Series(flux_1d).rolling(window=15, center=True, min_periods=1).mean()

            # axes[idx].errorbar(
            #     time_steps,
            #     flux_1d,
            #     yerr=err_1d,
            #     fmt="o",
            #     markersize=2.5,
            #     elinewidth=0.6,
            #     capsize=0,
            #     alpha=0.65,
            #     color="tab:blue",
            #     ecolor="tab:gray",
            #     label="Generated points",
            # )
            axes[idx].plot(
                time_steps,
                flux_1d,
                color="crimson",
                lw=1.8,
                label="Rolling mean (15 pts)",
            )
            axes[idx].set_ylabel("Normalized Flux [-1, 1]")
            axes[idx].set_title(f"Synthetic Transit Light Curve #{idx + 1}")
            axes[idx].grid(True, alpha=0.25)

        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("Time Cadence (0–967)")

        plt.tight_layout()
        out_file = Path(output_path)
        plt.savefig(out_file, dpi=150)
        plt.close(fig)
        print(f"Saved reconstructed 1D light curves to: {out_file.resolve()}")

    def generate_and_plot(self, num_samples=4, output_path="synthetic_light_curves.png"):
        '''
        Generate synthetic 44x44 outputs, reconstruct 1D relative flux anchored
        at a baseline of 1.0, and plot with a shaded 1-sigma uncertainty band.
        '''
        window = 5
        self.netG.eval()

        with torch.no_grad():
            noise = torch.randn(num_samples, self.nz, 1, 1, device=self.device)
            fake_2d = self.netG(noise).cpu().numpy()

        f_min = getattr(self, "flux_min", 0.98)
        f_max = getattr(self, "flux_max", 1.01)
        e_min = getattr(self, "err_min", 0.0005)
        e_max = getattr(self, "err_max", 0.003)

        time_steps = np.arange(968)
        fig, axes = plt.subplots(num_samples, 1, figsize=(11, 2.8 * num_samples), sharex=True)
        if num_samples == 1:
            axes = [axes]

        for idx in range(num_samples):
            flat_1d = fake_2d[idx].reshape(-1)
            flux_norm = flat_1d[:968]

            ######################## test per i pesi #################################
            # err_norm = flat_1d[968:]
            err_norm = np.clip(flat_1d[968:] / 0.2, -1.0, 1.0)

            # Riporta alla scala fisica e ancora la baseline (mediana) a 1.0
            flux_phys = 0.5 * (flux_norm + 1.0) * (f_max - f_min) + f_min
            err_phys = np.clip(0.5 * (err_norm + 1.0) * (e_max - e_min) + e_min, 0.0, None)

            baseline = np.median(flux_phys)
            flux_rel = flux_phys / baseline
            err_rel = err_phys / baseline

            # Media mobile per far risaltare eventuali dip di transito
            smooth_flux = pd.Series(flux_rel).rolling(window=window, center=True, min_periods=1).mean()

            # Banda d'errore +/- 1 sigma + linea continua della curva di luce
            axes[idx].fill_between(
                time_steps,
                flux_rel - err_rel,
                flux_rel + err_rel,
                color="tab:blue",
                alpha=0.22,
                label="±1σ uncertainty",
            )
            axes[idx].plot(
                time_steps,
                flux_rel,
                color="tab:blue",
                lw=0.85,
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
            axes[idx].set_title(f"Synthetic Transit Light Curve #{idx + 1}")
            axes[idx].grid(True, alpha=0.25)

        axes[0].legend(loc="upper right", ncol=3, fontsize=8.5)
        axes[-1].set_xlabel("Time Cadence (0–967)")

        plt.tight_layout()
        out_file = Path(output_path)
        plt.savefig(out_file, dpi=150)
        plt.close(fig)
        print(f"Saved reconstructed 1D light curves to: {out_file.resolve()}")


# input_file = '/home/michael/compu_astro/group2-GOTHAM/gan_params.yaml'
# dictionary = Parameters(input_file).params
# gan = GenerativeAdversarialNetwork(dictionary['dream'])
# gan.train()
# gan.generate_and_plot(output_path="synthetic_light_curves.png")