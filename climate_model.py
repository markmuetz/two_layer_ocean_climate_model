from pathlib import Path

import ipywidgets as widgets
import ipywidgets.widgets.interaction as interaction
from ipywidgets import Layout
from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class ClimateModel:
    default_vars = {
        'lam': 0.8,  # K (W m-2)-1
        'd_m': 100,  # m
        'd_d': 900,  # m
        'K': 1e-4,  # m2 s-1
        'aer-cld scaling': 1,  # unitless
    }
    consts = dict(
        rho=1000,  # kg m-3
        c_p=4218,  # J kg-1 K-1
        dt=365.2422 * 86400,  # s
    )
    # Forcings to use: CO2, other WMGHGs, O3, aerosol-radiation, aerosol-cloud, other anthropogenic (sum of everything else), solar, volcanic.
    # Forcings in CSV: co2, ch4, n2o, other_wmghg, o3, h2o_stratospheric, contrails, aerosol-radiation_interactions, aerosol-cloud_interactions,
    #                  bc_on_snow, land_use, volcanic, solar, nonco2_wmghg, aerosol, chapter2_other_anthro, total_anthropogenic, total_natural, total
    # Mapping
    control_forcings = {
        'CO2': ['co2'],
        'other_WMGHGs': ['ch4', 'n2o', 'other_wmghg'],
        'O3': ['o3'],
        'aerosol_radiation': ['aerosol-radiation_interactions'],
        'aerosol_cloud': ['aerosol-cloud_interactions'],
        'other_anthropogenic': ['h2o_stratospheric', 'contrails', 'bc_on_snow', 'land_use'],
        'solar': ['solar'],
        'volcanic': ['volcanic'],
    }

    def __init__(self, debug=False):
        self.debug = debug
        self.forcings = pd.read_csv(Path('data/AR6_ERF_1750-2019.csv'))
        self.scenario_forcings = {}
        self.scenarios = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
        for scenario in self.scenarios:
            scenario_key = scenario.replace('-', '').replace('.', '').lower()
            self.scenario_forcings[scenario] = pd.read_csv(Path(f'data/ERF_{scenario_key}_1750-2500.csv'))
        self.hadcrut5 = pd.read_csv(Path('data/gmt_HadCRUT5.csv'))
        self.hadcrut5 = self.hadcrut5[self.hadcrut5.Year <= 2019]
        self.hadcrut5['HadCRUT5 1850-1899 anom (degC)'] = (
            self.hadcrut5['HadCRUT5 (degC)']
            - self.hadcrut5[(self.hadcrut5.Year >= 1850) & (self.hadcrut5.Year <= 1899)]['HadCRUT5 (degC)'].mean()
        )

    def _print_debug(self, msg):
        if self.debug:
            print(msg)

    def update_T(self, dTm, dTd, dF, dt, lam, d_m, d_d, K, rho, c_p):
        C_m = rho * c_p * d_m
        C_d = rho * c_p * d_d
        D = K * rho * c_p * ((dTm - dTd) / (0.5 * (d_m + d_d)))
        dTm = dTm + dt / C_m * (dF - dTm / lam - D)
        dTd = dTd + dt / C_d * D
        return dTm, dTd

    def run_model(self, total_forcing=None, **control_values):
        # Use difference forcings based on whether SSP is selected.
        if scenario := control_values.get('future_scenario', None):
            max_col = 11
            forcings = self.scenario_forcings[scenario]
        else:
            max_col = 14
            forcings = self.forcings
            self.current_control_values = control_values
        self._print_debug(control_values)

        # Map control values.
        # E.g., if 'other WMGHGs' in control_values, mapped_control_values: ['ch4', 'n2o', 'other_wmghg']
        # will have their values set.
        mapped_control_values = {}
        for k, v in control_values.items():
            if k in self.control_forcings:
                for forcing in self.control_forcings[k]:
                    mapped_control_values[forcing] = v
            else:
                mapped_control_values[k] = v
        self.control_values = mapped_control_values
        self._print_debug(mapped_control_values)

        # Use default values if no values applied.
        var_values = {k: control_values.get(k, v) for k, v in self.default_vars.items()}
        forcing_columns = [c for c in forcings.columns[1:max_col] if self.control_values.get(c, True)]
        self._print_debug(forcing_columns)

        aer_cld_scaling = var_values.pop('aer-cld scaling')
        # If total_forcing is supplied, just use, otherwise calc. from selected forcing_columns.
        if total_forcing is not None:
            self.total_forcing = total_forcing
        else:
            forcings = forcings.copy()
            self._print_debug(forcings.columns)
            forcings['aerosol-cloud_interactions'] *= aer_cld_scaling
            # Set forcings to all forcings, without first year.
            self.total_forcing = forcings.loc[1:][forcing_columns].sum(axis=1)

        # ICs.
        years = [1750]
        dT = [(0, 0)]
        self._print_debug(var_values)
        # Run time loop. Note, skip the first year for years, and total forcings first year has
        # been removed.
        for year, F in list(zip(forcings.year.loc[1:], self.total_forcing)):
            dT.append(self.update_T(dT[-1][0], dT[-1][1], F, **{**self.consts, **var_values}))
            years.append(int(year))

        # Put resuts into nice DataFrame.
        dT = np.array(dT)
        dT = pd.DataFrame(data={'year': years, 'dTm': dT[:, 0], 'dTd': dT[:, 1]})
        self.dT = dT

        # Calc. anomaly_baseline and RMSE
        self.anomaly_baseline = dT[(dT.year >= 1850) & (dT.year <= 1899)].mean()
        dTm_match_obs = dT[(dT.year >= 1850) & (dT.year <= 2019)].dTm - self.anomaly_baseline.dTm
        self.rmse = np.sqrt(
            ((self.hadcrut5['HadCRUT5 1850-1899 anom (degC)'].values - dTm_match_obs.values) ** 2).mean()
        )

    def optimize_lambda(self, **control_values):
        # Function to optimize. Note closure over control_values.
        def func(lam):
            control_values['lam'] = lam.item()
            self.run_model(**control_values)
            return self.rmse

        res = minimize(func, 0.8)
        self._print_debug(res)
        return res.fun, res.x.item()

    def configure_axes(self, axes):
        axes = [ax for ax in axes if ax is not None]
        for ax in axes:
            if self.control_values.get('future_scenario', None):
                ax.set_xlim((1850, 2110))
                ax.set_xticks(range(1850, 2110, 10))
            else:
                ax.set_xlim((1850, 2020))
                ax.set_xticks(range(1850, 2030, 10))

        axes[-1].tick_params(axis="x", rotation=90)
        for ax in axes[:-1]:
            ax.set_xticks([])

    def plot(
        self,
        ax0=None,
        ax1=None,
        ax2=None,
        show_forcing=True,
        show_equilibrium=True,
        show_shallow=True,
        show_deep=False,
        show_obs=True,
    ):
        disp_legend = ax1 is None
        if ax1 is None:
            if show_forcing and show_equilibrium:
                fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
            elif show_forcing:
                fig, (ax0, ax1) = plt.subplots(2, 1)
            elif show_equilibrium:
                fig, (ax1, ax2) = plt.subplots(2, 1)
            else:
                fig, ax1 = plt.subplots()
            fig.set_size_inches((18, 8))

            if show_obs:
                if self.control_values:
                    lam, d_d, d_m, K, aer_cld_scaling = [
                        self.control_values.get(k, v) for k, v in self.default_vars.items()
                    ]
                    title = (
                        f'$\lambda = ${lam:.2f}, $d_d = ${d_d:.2f}, $d_m = ${d_m:.2f}, '
                        f'$\kappa = ${K:.5f}, aer-cld scaling: {aer_cld_scaling:.2f}. RMSE: {self.rmse:.6f}'
                    )
                else:
                    title = f'RMSE: {self.rmse:.3f}'
            else:
                lam, d_d, d_m, K, aer_cld_scaling = [
                    self.control_values.get(k, v) for k, v in self.default_vars.items()
                ]
                title = (
                    f'$\lambda = ${lam:.2f}, $d_d = ${d_d:.2f}, $d_m = ${d_m:.2f}, '
                    f'$\kappa = ${K:.5f}, aer-cld scaling: {aer_cld_scaling:.2f}'
                )
            ax = [ax for ax in [ax0, ax1, ax2] if ax is not None][0]
            ax.set_title(title)

        # Note, ONLY values from 1850 on are plotted (index=100) so that autoscale works as expected.
        if show_shallow:
            ax1.plot(self.dT.year[100:], self.dT.dTm[100:] - self.anomaly_baseline.dTm, label='$\Delta$Tm')
        if show_deep:
            ax1.plot(self.dT.year[100:], self.dT.dTd[100:] - self.anomaly_baseline.dTd, label='$\Delta$Td')
        if show_obs:
            ax1.plot(
                self.hadcrut5.Year, self.hadcrut5['HadCRUT5 1850-1899 anom (degC)'], color='r', label='HadCRUT 5 obs.'
            )
            ax1.fill_between(
                self.hadcrut5.Year,
                self.hadcrut5['HadCRUT5 1850-1899 anom (degC)'] - self.hadcrut5['HadCRUT5 uncertainty'] / 2,
                self.hadcrut5['HadCRUT5 1850-1899 anom (degC)'] + self.hadcrut5['HadCRUT5 uncertainty'] / 2,
                alpha=0.4,
                facecolor='r',
                label='obs. uncert.',
            )
        ax1.set_ylabel('temperature anomaly 1850-1899 ($^\circ$C)')

        if show_forcing:
            # N.B. total_forcing is missing the first year, hence difference in start index.
            ax0.plot(self.dT.year[100:], self.total_forcing[99:], 'k--', label='forcing')
            ax0.set_ylabel('forcing (W m$^{-2}$)')

        if show_equilibrium:
            ax2.plot(self.dT.year[100:], self.dT.dTm[100:] - self.anomaly_baseline.dTm, label='transient')
            # N.B. total_forcing is missing the first year, hence difference in start index.
            ax2.plot(self.dT.year[100:], self.total_forcing[99:] * self.control_values['lam'], label='equilibrium')
            ax2.tick_params(axis="x", rotation=90)
            ax2.set_ylabel('($^\circ$C)')

        self.configure_axes([ax0, ax1, ax2])

        if disp_legend:
            ax1.legend(loc='upper left')
            if show_forcing:
                ax0.legend(loc='upper left')
            if show_equilibrium:
                ax2.legend(loc='upper left')


class ClimateModelUI:
    var_slider_kwargs = {
        'lam': {'min': 0.4, 'max': 3.2, 'step': 0.05, 'readout_format': '.2f'},
        'd_m': {'min': 10, 'max': 300, 'step': 10, 'readout_format': '.0f'},
        'd_d': {'min': 100, 'max': 2000, 'step': 50, 'readout_format': '.0f'},
        'K': {'min': 0, 'max': 4e-4, 'step': 5e-5, 'readout_format': '.5f'},
        'aer-cld scaling': {'min': 0, 'max': 2, 'step': 0.01, 'readout_format': '.2f'},
    }

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
        cb_layout = Layout(width='200px')
        slider_layout = Layout(width='300px')

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

        for k, value in self.model.default_vars.items():
            slider_kwargs = self.var_slider_kwargs[k]
            slider = widgets.FloatSlider(
                value=value,
                description=f'{k}: ',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                layout=slider_layout,
                **slider_kwargs,
            )
            controls[k] = slider
            self.sliders.append(slider)

        self.controls = controls

        if fancy_ui:
            blayout = Layout(width='40px')
            label_layout = Layout(width='80px')
            spacer_layout = Layout(width='40px')
            # Fancy UI with buttons for setting all anthro/nat forcings on/off, resetting sliders.
            # Needs a button to run the model as well.
            anthro = widgets.Label('Anthro.', layout=label_layout)
            nat = widgets.Label('Natural', layout=label_layout)

            self.anthro_on = widgets.Button(description='on', layout=blayout)
            self.anthro_off = widgets.Button(description='off', layout=blayout)
            self.nat_on = widgets.Button(description='on', layout=blayout)
            self.nat_off = widgets.Button(description='off', layout=blayout)
            self.reset_vars = widgets.Button(description='reset vars.', layout=slider_layout)

            self.anthro_cbs = cbs[:6]
            self.nat_cbs = cbs[6:]
            self.run_model_btn = widgets.Button(description='Run model', layout=slider_layout)

            for btn in [
                self.anthro_on,
                self.anthro_off,
                self.nat_on,
                self.nat_off,
                self.reset_vars,
                self.run_model_btn,
            ]:
                btn.on_click(self.btn_clicked)

            ui = widgets.VBox(
                [
                    future_scenario,
                    widgets.HBox([anthro, self.anthro_on, self.anthro_off] + cbs[:3]),
                    widgets.HBox(
                        [
                            widgets.Label('Anthro.', layout=label_layout),
                            widgets.Label('', layout=spacer_layout),
                            widgets.Label('', layout=spacer_layout),
                        ]
                        + cbs[3:6]
                    ),
                    widgets.HBox([nat, self.nat_on, self.nat_off] + cbs[6:]),
                    widgets.HBox([self.reset_vars] + self.sliders[:2]),
                    widgets.HBox(self.sliders[2:]),
                    self.run_model_btn,
                ]
            )
            self.ui = ui
            self.output = widgets.Output()
            display(ui, self.output)
        else:
            # Simple way of doing this: https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html#More-control-over-the-user-interface:-interactive_output
            ui = widgets.VBox(
                [
                    future_scenario,
                    widgets.HBox(cbs[:3]),
                    widgets.HBox(cbs[3:6]),
                    widgets.HBox(cbs[6:]),
                    widgets.HBox(self.sliders),
                ]
            )

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
                self.run_model(**self.get_values())
                interaction.show_inline_matplotlib_plots()

    def get_values(self):
        return {k: ctrl.value for k, ctrl in self.controls.items()}

    def run_model(self, **control_values):
        self.model.run_model(**control_values)
        self.model.plot()
