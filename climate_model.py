import ipywidgets as widgets
import ipywidgets.widgets.interaction as interaction
from ipywidgets import GridspecLayout, Layout
from IPython.display import display

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class ClimateModel:
    def __init__(self):
        self.default_vars = dict(
            lam = 0.8,  # K (W m-2)-1
            d_m = 100,  # m
            d_d = 900,  # m
            K = 1e-4,   # m2 s-1
        )
        self.consts = dict(
            rho = 1000,             # kg m-3
            c_p = 4218,             # J kg-1 K-1
            dt = 365.2422 * 86400,  # s
        )
        # self.forcings = pd.read_csv('data/forcing_data.csv')
        self.forcings = pd.read_csv('data/ar6/data_output/AR6_ERF_1750-2019.csv')
        self.scenario_forcings = {}
        self.scenarios = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
        for scenario in self.scenarios:
            scenario_key = scenario.replace('-', '').replace('.', '').lower()
            self.scenario_forcings[scenario] = pd.read_csv(f'data/ar6/data_output/SSPs/ERF_{scenario_key}_1750-2500.csv')
        # self.obs_temp_anomaly = pd.read_csv('data/observed_temperature_anomaly_1961-1990.csv')
        self.hadcrut5 = pd.read_csv('data/gmt_HadCRUT5.csv')
        self.hadcrut5 = self.hadcrut5[self.hadcrut5.Year <= 2019]
        self.hadcrut5['HadCRUT5 1850-1899 anom (degC)'] = (self.hadcrut5['HadCRUT5 (degC)'] -
                                                           self.hadcrut5[(self.hadcrut5.Year >= 1850) &
                                                                         (self.hadcrut5.Year <= 1899)]['HadCRUT5 (degC)'].mean())

        # Forcings to use: CO2, other WMGHGs, O3, aerosol-radiation, aerosol-cloud, other anthropogenic (sum of everything else), solar, volcanic.
        # Forcings in CSV: co2, ch4, n2o, other_wmghg, o3, h2o_stratospheric, contrails, aerosol-radiation_interactions, aerosol-cloud_interactions,
        #                  bc_on_snow, land_use, volcanic, solar, nonco2_wmghg, aerosol, chapter2_other_anthro, total_anthropogenic, total_natural, total
        # Mapping
        self.control_forcings = {
            'CO2': ['co2'],
            'other WMGHGs': ['ch4', 'n2o', 'other_wmghg'],
            'O3': ['o3'],
            'aerosol-radiation': ['aerosol-radiation_interactions'],
            'aerosol-cloud': ['aerosol-cloud_interactions'],
            'other anthropogenic': ['h2o_stratospheric', 'contrails', 'bc_on_snow', 'land_use'],
            'solar': ['solar'],
            'volcanic': ['volcanic'],
        }

    def update_T(self, dTm, dTd, dF, dt, lam, d_m, d_d, K, rho, c_p):
        C_m = rho * c_p * d_m
        C_d = rho * c_p * d_d
        D = K * rho * c_p * ((dTm - dTd) / (0.5 * (d_m + d_d)))
        dTm = dTm + dt / C_m * (dF - dTm / lam - D)
        dTd = dTd + dt / C_d * D
        return dTm, dTd

    def run_model(self, total_forcing=None, **control_values):
        if scenario := control_values.get('future_scenario', None):
            max_col = 11
            forcings = self.scenario_forcings[scenario]
        else:
            max_col = 14
            forcings = self.forcings
        # print(control_values)
        mapped_control_values = {}
        for k, v in control_values.items():
            if k in self.control_forcings:
                for forcing in self.control_forcings[k]:
                    mapped_control_values[forcing] = v
            else:
                mapped_control_values[k] = v
        # print(mapped_control_values)
        self.control_values = mapped_control_values
        var_values = {k: control_values.get(k, v) for k, v in self.default_vars.items()}
        forcing_columns = [c for c in forcings.columns[1:max_col] if self.control_values.get(c, True)]
        # print(forcing_columns)
        if total_forcing is not None:
            self.total_forcing = total_forcing
        else:
            self.total_forcing = forcings.loc[1:][forcing_columns].sum(axis=1)
        years = [1750]
        dT = [(0, 0)]
        for year, F in list(zip(forcings.year.loc[1:], self.total_forcing)):
            dT.append(self.update_T(dT[-1][0], dT[-1][1], F, **{**self.consts, **var_values}))
            years.append(int(year))
        dT = np.array(dT)
        dT = pd.DataFrame(data={'year': years, 'dTm': dT[:, 0], 'dTd': dT[:, 1]})
        self.dT = dT

        self.anomaly_baseline = dT[(dT.year >= 1850) & (dT.year <= 1899)].mean()
        dTm_match_obs = (dT[(dT.year >= 1850) & (dT.year <= 2019)].dTm - self.anomaly_baseline.dTm)
        self.rmse = np.sqrt(((self.hadcrut5['HadCRUT5 1850-1899 anom (degC)'].values - dTm_match_obs.values)**2).mean())

    def optimize_lambda(self):
        def func(lam):
            self.run_model(lam=lam.item())
            return self.rmse

        res = minimize(func, 0.8)
        # print(res)
        return res.fun, res.x.item()

    def base_plot(self, ax):
        ax.plot(self.hadcrut5.Year, self.hadcrut5['HadCRUT5 1850-1899 anom (degC)'], color='r', label='HadCRUT 5 obs.')
        ax.fill_between(
            self.hadcrut5.Year,
            self.hadcrut5['HadCRUT5 1850-1899 anom (degC)'] - self.hadcrut5['HadCRUT5 uncertainty'] / 2,
            self.hadcrut5['HadCRUT5 1850-1899 anom (degC)'] + self.hadcrut5['HadCRUT5 uncertainty'] / 2,
            alpha=0.4, facecolor='r', label='obs. uncert.'
        )

        ax.set_ylim((-1.5, 2.5))
        if self.control_values.get('future_scenario', None):
            ax.set_xlim((1850, 2110))
            ax.set_xticks(range(1850, 2110, 10))
        else:
            ax.set_xlim((1850, 2020))
            ax.set_xticks(range(1850, 2030, 10))
        ax.tick_params(axis="x", rotation=90)
        ax.set_xlabel('year')
        ax.set_ylabel('temperature anomaly 1850-1899 ($^\circ$C)')

    def plot(self, ax1=None, ax2=None, show_forcing=True, show_shallow=True, show_deep=False, show_equilibrium=True):
        disp_legend = ax1 is None
        if ax1 is None:
            if show_equilibrium:
                fig, (ax1, ax2) = plt.subplots(2, 1)
            else:
                fig, ax1 = plt.subplots()
            fig.set_size_inches((18, 8))
            self.base_plot(ax=ax1)
            if self.control_values:
                lam, d_d, d_m, K = [self.control_values.get(k, v) for k, v in self.default_vars.items()]
                title = f'$\lambda = ${lam:.2f}, $d_d = ${d_d:.2f}, $d_m = ${d_m:.2f}, $\kappa = ${K:.4f}. RMSE: {self.rmse:.6f}'
            else:
                title = f'RMSE: {self.rmse:.3f}'
            ax1.set_title(title)
        if show_forcing:
            ax1t = ax1.twinx()
            ax1t.set_ylabel('forcing (W m$^{-2}$)')
        if show_shallow:
            ax1.plot(self.dT.year, self.dT.dTm - self.anomaly_baseline.dTm, label='$\Delta$Tm')
        if show_deep:
            ax1.plot(self.dT.year, self.dT.dTd - self.anomaly_baseline.dTd, label='$\Delta$Td')

        if show_equilibrium:
            ax2.plot(self.dT.year, self.dT.dTm - self.anomaly_baseline.dTm, label='transient')
            ax2.plot(self.dT.year.loc[1:], self.total_forcing * self.control_values['lam'], label='equilibrium')
            if self.control_values.get('future_scenario', None):
                ax2.set_xlim((1850, 2110))
                ax2.set_xticks(range(1850, 2110, 10))
            else:
                ax2.set_xlim((1850, 2020))
                ax2.set_xticks(range(1850, 2030, 10))
            ax2.tick_params(axis="x", rotation=90)
            ax2.set_ylabel('temperature anomaly 1850-1899 ($^\circ$C)')

        handles, labels = ax1.get_legend_handles_labels()
        if show_forcing:
            # N.B. Adding this to a separate axis, so label='forcing' here will not show up.
            ax1t.plot(self.dT.year.loc[1:], self.total_forcing, 'k--')
            handles.append(Line2D([0], [0], color='k', linestyle='--', label='forcing'))

        if disp_legend:
            ax1.legend(handles=handles, loc='upper left')
            if show_equilibrium:
                ax2.legend(loc='upper left')


class ClimateModelUI:
    def __init__(self, model):
        self.model = model

    def display(self, fancy_ui=True):
        controls = {}
        future_scenario = widgets.Dropdown(
            options=[None, 'SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5'],
            value=None,
            description='Future scenaro:',
            disabled=False,
        )
        controls['future_scenario'] = future_scenario
        cbs = []
        self.sliders = []
        cb_layout = widgets.Layout(width='200px')

        # for column in self.model.forcings.columns[1:11]:
        for column in self.model.control_forcings.keys():
            checkbox = widgets.Checkbox(
                value=True,
                description=f'{column}',
                disabled=False,
                layout=cb_layout,
                # indent=False
            )
            controls[column] = checkbox
            cbs.append(checkbox)

        for k, v in self.model.default_vars.items():
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
            self.sliders.append(slider)

        for control in controls.values():
            # control.layout=Layout(height='auto', width='auto')
            pass
        self.controls = controls

        grid = GridspecLayout(5, 4, width='400px')
        grid[0, :2] = future_scenario
        for i in range(3):
            grid[1, i] = cbs[:3][i]
            grid[2, i] = cbs[3:6][i]
        for i in range(2):
            grid[3, i] = cbs[6:][i]
        for i in range(4):
            grid[4, i] = self.sliders[i]
        if fancy_ui:
            # Fancy UI with buttons for setting all anthro/nat forcings on/off, resetting sliders.
            # Needs a button to run the model as well.
            blayout = widgets.Layout(width='40px')
            anthro = widgets.Label('Anthro.', layout=widgets.Layout(width='80px'))
            nat = widgets.Label('Natural', layout=widgets.Layout(width='80px'))

            self.anthro_on = widgets.Button(description='on', layout=blayout)
            self.anthro_off = widgets.Button(description='off', layout=blayout)
            self.nat_on = widgets.Button(description='on', layout=blayout)
            self.nat_off = widgets.Button(description='off', layout=blayout)
            self.reset_vars = widgets.Button(description='reset vars.', layout=Layout(width='160px'))

            self.anthro_cbs = cbs[:6]
            self.nat_cbs = cbs[6:]
            self.run_model_btn = widgets.Button(description='Run model')

            for btn in [self.anthro_on, self.anthro_off, self.nat_on, self.nat_off, self.reset_vars, self.run_model_btn]:
                btn.on_click(self.btn_clicked)

            ui = widgets.VBox([
                future_scenario,
                widgets.HBox([anthro, self.anthro_on, self.anthro_off] + cbs[:3]),
                widgets.HBox([
                    widgets.Label('Anthro.', layout=widgets.Layout(width='80px')),
                    widgets.Label('', layout=widgets.Layout(width='40px')),
                    widgets.Label('', layout=widgets.Layout(width='40px')),
                ] + cbs[3:6]),
                widgets.HBox([nat, self.nat_on, self.nat_off] + cbs[6:]),
                widgets.HBox([self.reset_vars] + self.sliders),
                self.run_model_btn,
            ])
            self.ui = ui
            self.output = widgets.Output()
            display(ui, self.output)
        else:
            # Simple way of doing this: https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html#More-control-over-the-user-interface:-interactive_output
            ui = widgets.VBox([
                future_scenario,
                widgets.HBox(cbs[:4]),
                widgets.HBox(cbs[4:8]),
                widgets.HBox(cbs[8:]),
                widgets.HBox(self.sliders),
            ])

            out = widgets.interactive_output(self.run_model, controls)
            display(ui, out)

        # Stop flickering? https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html#Flickering-and-jumping-output
        ui.layout.height = '200px'

    def btn_clicked(self, btn):
        if btn == self.anthro_on:
            for cb in self.anthro_cbs:
                cb.value = True
        elif btn == self.anthro_off:
            for cb in self.anthro_cbs:
                cb.value = False
        elif btn == self.nat_on:
            for cb in self.nat_cbs:
                cb.value = True
        elif btn == self.nat_off:
            for cb in self.nat_cbs:
                cb.value = False
        elif btn == self.reset_vars:
            for v, slider in zip(self.model.default_vars.values(), self.sliders):
                slider.value = v
        elif btn == self.run_model_btn:
            with self.output:
                interaction.clear_output(wait=True)
                self.run_model(**{k: ctrl.value for k, ctrl in self.controls.items()})
                interaction.show_inline_matplotlib_plots()

    def run_model(self, **control_values):
        self.model.run_model(**control_values)
        self.model.plot()

