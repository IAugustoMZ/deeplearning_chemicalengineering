import numpy as np
import pandas as pd
from scipy.integrate import odeint

class MixtureTank:

    scenario_codes = {
        0: {
            'step': 0.0,
            'var': None
        },
        1: {
            'step': 0.05,
        },
        2: {
            'step': 0.2,
        },
        3: {
            'step': -0.05,
        },
        4: {
            'step': -0.2,
        }

    }

    def __init__(self, state0: tuple) -> None:
        """
        class to simulate a mixture tank

        Parameters
        ----------
        state0 : tuple
            initial state tuple
        """
        # initial conditions
        # [0] = V0
        # [1] = Ca0
        # [2] = T0
        self.state0 = list(state0)

        # max volume of the tank (L)
        self.Vmax = 15

        # fixed out flow rate (L/min)
        self.q_out = 5

        # integration points
        self.N_int = 1000

    def mixture_tank(self,
                     x: np.array,
                     t: np.array,
                     q: np.array,
                     qf: np.array,
                     Caf: np.array,
                     Tf: np.array) -> list:
        """
        performs the balances (material and energy)

        Parameters
        ----------
        x : np.array
            vector with state variables
        t : np.array
            time array
        q : np.array
            outlet flowrate (L/min)
        qf : np.array
            feed flowrate (L/min)
        Caf : np.array
            feed concentration (mol/L)
        Tf : np.array
            feed temperature (K)

        Returns
        -------
        list
            list of derivatives
        """

        # states (3)
        V = x[0]    # volume (L)
        Ca = x[1]   # concentration (mol/L)
        T = x[2]    # temperature (K)

        # mass balance: volume derivative
        dVdt = qf - q

        # species balance: concentration derivative
        dCadt = (((qf * Caf) - (q * Ca)) / V) - ((Ca * dVdt) / V)

        # energy balance: temperature derivative
        dTdt = (((qf * Tf) - (q * T)) / V) - ((T * dVdt) / V)

        return [dVdt, dCadt, dTdt]
    
    def simulate(self, tsim: int, scenario: int=0) -> pd.DataFrame:
        """
        executes the simulation

        Parameters
        ----------
        tsim : int
            simulation time, minutes
        scenario : int, optional
            scenario code, by default 0

        Returns
        -------
        pd.DataFrame
            dataframe with simulation results.
        """

        # time interval
        t = np.linspace(0, tsim, self.N_int)

        # create simulation scenario
        q, qf, Caf, Tf = self.create_scenario(scenario, len(t))

        # store results
        V = np.ones(len(t)) * self.state0[0]
        Ca = np.ones(len(t)) * self.state0[1]
        T = np.ones(len(t)) * self.state0[2]

        for i in range(len(t)-1):

            # simulate
            inputs = (q[i], qf[i], Caf[i], Tf[i])
            ts = [t[i], t[i+1]]
            y = odeint(self.mixture_tank, self.state0, ts, args=inputs)

            # store results
            V[i+1] = y[-1][0]
            Ca[i+1] = y[-1][1]
            T[i+1] = y[-1][2]

            # adjust initial condition
            y0 = y[-1]

        # create data frame
        data = pd.DataFrame()
        data['q_feed_L_min'] = qf
        data['q_out_L_min'] = q
        data['temp_feed_C'] = Tf - 273.15
        data['conc_feed_mol_L'] = Caf
        data['level'] = V / self.Vmax
        data['conc_out_mol_L'] = Ca
        data['temp_out_C'] = T - 273.15

        # generate noise level
        data = self.add_noise(data)

        return data

    def create_scenario(self, scenario: int, size:int) -> tuple:
        """
        create simulation scenario,
        based in a code

        Parameters
        ----------
        scenario : int
            scenario code
        size : int
            size of the time vector

        Returns
        -------
        tuple
            tuple with initial conditions
        """
        # output flow rate is always constant
        q = np.ones(size) * self.q_out
        qf = q.copy()
        Caf = np.ones(size) * self.state0[1]
        Tf = np.ones(size) * self.state0[2]

        if scenario != 0:
            final_step = self.scenario_codes[scenario]['step']
            half = size // 2
            m = (final_step) / ((size - 1) - half)
            qf[half:] = (m * self.q_out * np.linspace(0, half, half)) + self.q_out
            Caf[half:] = (m * self.state0[1] * np.linspace(0, half, half)) + self.state0[1]
            Tf[half:] = (m * self.state0[2] * np.linspace(0, half, half)) + self.state0[2]

        return q, qf, Caf, Tf
    
    def add_noise(self,
                  data: pd.DataFrame,
                  noise_level: float=0.01) -> pd.DataFrame:
        """
        adds normal noise to the data

        Parameters
        ----------
        data : pd.DataFrame
            clean data
        noise_level : float, optional
            noise / signal ratio, by default 0.01

        Returns
        -------
        pd.DataFrame
            data with noise
        """
        df = data.copy()

        for col in df.columns:
            if col not in ['t_min']:
                df[col] += np.random.normal(
                    loc=0,
                    scale=noise_level * data[col].mean(),
                    size=data[col].shape[0]
                )

        return df



if __name__ == '__main__':

    import os

    tank = MixtureTank(state0=(2.0, 0.2, 350.0))

    path = 'D:\\Portfolio\\ProjetosPessoais\\deeplearning_chemicalengineering\\0_data\\som_studies'

    for i in range(5):
        if i == 0:
            data = tank.simulate(tsim=5*600, scenario=i)
            data.to_csv(os.path.join(path, 'data_normal.csv'), index=False)
        else:
            aux = tank.simulate(tsim=600, scenario=i)

            if i == 1:
                data = aux.copy()
            else:
                data = pd.concat([data, aux])
            
            data.reset_index(inplace=True)
            data.drop(['index'], axis=1, inplace=True)
            data.to_csv(os.path.join(path, 'data_faulty.csv'), index=False)

