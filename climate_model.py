import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ClimateModel:
    def __init__(self):
        self.vars = dict(
            lam = 0.08,  # K (W m-2)-1
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

    def update_T(self, dTm, dTd, dF, dt, lam, d_m, d_d, K, rho, c_p):
        C_m = rho * c_p * d_m
        C_d = rho * c_p * d_d
        D = K * rho * c_p * ((dTm - dTd) / (0.5 * (d_m + d_d)))
        dTm = dTm + dt / C_m * (dF - dTm / lam - D)
        dTd = dTd + dt / C_d * D
        return dTm, dTd

    def display_controls(self):
        controls = {}
        for column in self.forcings.columns[1:11]:
            checkbox = widgets.Checkbox(
                value=True,
                description=f'{column}',
                disabled=False,
                # indent=False
            )
            controls[column] = checkbox

        for k, v in self.vars.items():
            slider = widgets.FloatSlider(
                value=v,
                min=v / 2,
                max=v * 2,
                step=1e-5,
                description=f'{k}: ',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.4f',
            )
            controls[k] = slider
        display(interactive(self.run_model, **controls))

    def run_model(self, **control_values):
        print(control_values)
        slider_vars = {k: control_values[k] for k in self.vars.keys()}
        forcings_enabled = [c for c in self.forcings.columns[1:11] if control_values[c]]
        total_forcing = self.forcings.loc[2:][forcings_enabled].sum(axis=1)
        years = [1750]
        dT = [(0, 0)]
        for year, F in list(zip(self.forcings.YEAR.loc[2:], total_forcing)):
            dT.append(self.update_T(dT[-1][0], dT[-1][1], F, **{**self.consts, **slider_vars}))
            years.append(year)
        dT = np.array(dT)

        plt.plot(years, dT[:, 0])
        plt.plot(years, dT[:, 1])
