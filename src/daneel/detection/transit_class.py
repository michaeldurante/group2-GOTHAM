import batman
import numpy as np
import matplotlib.pyplot as plt

class TransitClass:
    def __init__(self, p_dictionary):
        ''' use class 'Parameters' to load YAML file inside the transit model'''
        self.p = batman.TransitParams()
        self.p.t0 = p_dictionary.get('t0', 0.0)
        self.p.per = p_dictionary.get('per')
        self.p.rp = p_dictionary.get('rp')
        self.p.a = p_dictionary.get('a')
        self.p.inc = p_dictionary.get('inc')
        self.p.ecc = p_dictionary.get('ecc')
        self.p.w = p_dictionary.get('w')
        self.p.limb_dark = "quadratic"
        self.p.u = p_dictionary.get('u')

        self.name = p_dictionary.get('name')

        ''' time array for the light curve'''
        self.t = np.linspace(-0.1, 0.1, 1000)

    def plot_transit(self, show, output_file='lc.png'):
        '''compute the light curve and plot/save it'''
        self.flux = batman.TransitModel(self.p, self.t).light_curve(self.p)
        assert np.all(np.isfinite(self.flux)), "The light curve contains non-finite values"

        plt.figure(figsize=(8, 6))
        plt.plot(self.t,self.flux, color='r', label="BATMAN transit model")
        plt.xlabel('Time from central transit [days]', fontsize=12, color='black')
        plt.ylabel('Normalized Flux', fontsize=12, color='black')
        plt.axhline(1.0, color="0.6", lw=1, ls="--", label="Out-of-transit flux")
        plt.grid(True, color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.title(f'Light curve - {self.name}')
        plt.legend(loc='lower left')
        plt.savefig(output_file, dpi=300)
        if show:
            plt.show()
        print(f"Light curve saved to {output_file}")

    def get_transit_depth(self):
        # Quick, approximate transit depth for an un-darkened stellar disc.
        estimate = (self.p.rp**2)
        print(f'Estimated flux at transit: {1-estimate:.4f}')