import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ClimateModel:
    def __init__(self):
        self.vars = dict(
            lam = 0.8,  # K (W m-2)-1
            d_m = 100,   # m
            d_d = 900,   # m
            K = 1e-4,    # m2 s-1
        )
        self.consts = dict(
            rho = 1000,  # kg m-3
            c_p = 4218,  # J kg-1 K-1
            dt = 365.2422 * 86400, # s
        )
        self.forcings = pd.read_csv('forcing_data.csv')
        self.obs_temp_anomaly = pd.read_csv('observed_temperature_anomaly_1961-1990.csv')
        self.obs_temp_uncert = pd.read_csv('observed_temperature_uncertainty.csv')

    def update_T(self, dTm, dTd, dF, dt, lam, d_m, d_d, K, rho, c_p):
        C_m = rho * c_p * d_m
        C_d = rho * c_p * d_d
        D = K * rho * c_p * ((dTm - dTd) / (0.5 * (d_m + d_d)))
        dTm = dTm + dt / C_m * (dF - dTm / lam - D)
        dTd = dTd + dt / C_d * D
        return dTm, dTd

    def display_controls(self):
        controls = {}
        future_scenario = widgets.Dropdown(
            options=[None, 'SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5'],
            value=None,
            description='Future scenaro:',
            disabled=False,
        )
        controls['future_scenario'] = future_scenario
        cbs = []
        sliders = []
        
        for column in self.forcings.columns[1:11]:
            checkbox = widgets.Checkbox(
                value=True,
                description=f'{column}',
                disabled=False,
                # indent=False
            )
            controls[column] = checkbox
            cbs.append(checkbox)

        for k, v in self.vars.items():
            slider = widgets.FloatSlider(
                value=v,
                min=v / 4,
                max=v * 4,
                step=1e-5,
                description=f'{k}: ',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.4f',
            )
            controls[k] = slider
            sliders.append(slider)

        # Attempt at doing this: https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html#More-control-over-the-user-interface:-interactive_output
        # Not working.
        ui = widgets.VBox([future_scenario, widgets.HBox(cbs[:4]), widgets.HBox(cbs[4:8]), widgets.HBox(cbs[8:]), widgets.HBox(sliders)])
        out = widgets.interactive_output(self.run_model, controls)
        ui.layout.height = '200px'
        display(ui, out)
        
        # Stop flickering? https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html#Flickering-and-jumping-output
        #interactive_plot = interactive(self.run_model, **controls)
        #output = interactive_plot.children[-1]
        #output.layout.height = '700px'
        #display(interactive_plot)

    def run_model(self, **control_values):
        # print(control_values)
        slider_vars = {k: control_values[k] for k in self.vars.keys()}
        forcings_enabled = [c for c in self.forcings.columns[1:11] if control_values[c]]
        total_forcing = self.forcings.loc[2:][forcings_enabled].sum(axis=1)
        years = [1750]
        dT = [(0, 0)]
        for year, F in list(zip(self.forcings.YEAR.loc[2:], total_forcing)):
            dT.append(self.update_T(dT[-1][0], dT[-1][1], F, **{**self.consts, **slider_vars}))
            years.append(int(year))
        dT = np.array(dT)
        dT = pd.DataFrame(data={'year': years, 'dTm': dT[:, 0], 'dTd': dT[:, 1]})
        anomaly_baseline = dT[(dT.year >= 1960) & (dT.year <= 1990)].mean()
        
        fig, ax = plt.subplots()
        fig.set_size_inches((15, 10))

        ax.plot(years, dT.dTm - anomaly_baseline.dTm, label='$\Delta$Tm')
        ax.plot(years, dT.dTd - anomaly_baseline.dTd, label='$\Delta$Td')
        ax.plot(self.obs_temp_anomaly.YEAR, self.obs_temp_anomaly['Obs.'], color='r', label='HadCRUT 4 obs.')
        
        ax.fill_between(
            self.obs_temp_uncert.Year, 
            self.obs_temp_uncert['Min. Uncert.'], 
            self.obs_temp_uncert['Max. Uncert.'], 
            alpha=0.4, facecolor='r', label='obs. uncert.')
        ax.legend(loc='upper left')
        # ax.set_ylim((-0.8, 0.8))
        if control_values['future_scenario']:
            ax.set_xlim((1750, 2110))
            ax.set_xticks(range(1750, 2120, 10))
        else:
            ax.set_xlim((1750, 2020))
            ax.set_xticks(range(1750, 2030, 10))
        ax.set_xlabel('year')
        ax.set_ylabel('temperature anomaly 1961-1990 ($^\circ$C)')
        
        dTm_match_obs = (dT[(dT.year >= 1850) & (dT.year <= 2015)].dTm - anomaly_baseline.dTm)
        
        rmse = np.sqrt(((self.obs_temp_anomaly['Obs.'] - dTm_match_obs)**2).mean())
        ax.set_title(f'RMSE: {rmse:.3f}')
        
