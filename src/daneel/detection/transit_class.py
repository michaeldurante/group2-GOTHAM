import batman
import numpy as np
import matplotlib.pyplot as plt

class TransitClass:
    def __init__(self, p_dictionary):

        self.planets_data = []
        
        for planet_name, params in p_dictionary.items():

            #Ignoring any root keys that do not contain a dictionary
            if not isinstance(params, dict):
                print(f"Skipping '{planet_name}' as it does not contain a dictionary of parameters.")
                continue

            ''' use class 'Parameters' to load YAML file inside the transit model'''
            p = batman.TransitParams()
            p.t0 = params.get('t0', 0.0)
            p.per = params.get('per')
            p.rp = params.get('rp')
            p.a = params.get('a')
            p.inc = params.get('inc')
            p.ecc = params.get('ecc')
            p.w = params.get('w')
            p.limb_dark = "quadratic"
            p.u = params.get('u')

            #time array for the light curve
            dt = 0.1
            t = np.linspace(p.t0 - dt, p.t0 + dt, 1000)

            model = batman.TransitModel(p,t)

            self.planets_data.append(
                {"name": params.get("name", planet_name),
                 "params": p, 
                 "t": t, 
                 "model": model}
            )

    def compute_lightcurve(self, planet):
        flux = planet['model'].light_curve(planet['params'])
        return flux
    
    def plot_transit(self, show, output_file='lc.png'):

        '''compute the light curve and plot/save it'''

        #DEBUG: if self.planets_data is empty we cannot plot anything
        if not self.planets_data: 
            print("ERROR: self.planets_data is empty")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axhline(1.0, color="0.6", lw=1, ls="--", label="Out-of-transit flux")

        for planet in self.planets_data:
            flux = self.compute_lightcurve(planet)
            assert np.all(np.isfinite(flux)), "The light curve contains non-finite values"

            t_centered = planet["t"] - planet["params"].t0
            ax.plot(t_centered, flux, label=planet["name"], linewidth=2)
  
        ax.set_xlabel('Time from central transit [days]', fontsize=12, color='black') 
        ax.set_ylabel('Normalized Flux', fontsize=12, color='black') 
        ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5)  
        if len(self.planets_data) > 1:
            ax.set_title('Comparison of Exoplanets Transit')  
        else:
            ax.set_title(f'Light curve of {planet["name"]}')
        ax.legend(loc='lower left', frameon = False)  
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)

        print(f"Light curve saved to {output_file}")

    def get_transit_depth(self):
        # Quick, approximate transit depth for an un-darkened stellar disc.
        for planet in self.planets_data:
            estimate = planet['params'].rp**2
            print(f"Planet '{planet['name']}': Estimate of transit depth = {1 - estimate:.4f}")