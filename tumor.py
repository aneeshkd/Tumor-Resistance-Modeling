import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import islice 
from scipy.integrate import solve_ivp

class Tumor:
    """
    A class used to represent a system of differential equations, where x 
    represents the susceptible tumor population and y represents the drug-resistant
    tumor population.

    Attributes
    -----
    k: int
        Carrying capacity of the tumor
    r1: float
        Susceptible replication rate 
    r2: float
        Drug-resistant replication rate 
    dmax: 
        Susceptible maximum rate of death
    d2: float
        Drug-resistant rate of death
    elimination_thres: 
        Threshold for a population to be considered eliminated
    x0: int
        Initial number of susceptible cells
    y0: int
        Initial number of drug-resistant cells 
    max: float
        Maximum treatment dosage, equivalent to dmax
    moderate: float
        Moderate treatment dosage, equivalent to dmax/2
    low: float
        Low treatment dosage, equivalent to dmax/4
    none: float
        No treatment condition, set to 0

    """
    def __init__(self, k: int, r1: float, r2: float, dmax: float, d2: float, 
                 elimination_thres: int=1, x0: int=95, y0: int=5):
        """
        Parameters 
        -----
        k: int
        Carrying capacity of the tumor
        r1: float
            Susceptible replication rate 
        r2: float
            Drug-resistant replication rate 
        dmax: float
            Susceptible maximum rate of death
        d2: float
            Drug-resistant rate of death
        elimination_thres: 
            Threshold for a population to be considered eliminated
        x0: int
            Initial number of susceptible cells
        y0: int
            Initial number of drug-resistant cells 
        """

        # Cancer params 
        self.k = k
        self.r1 = r1
        self.r2 = r2
        self.dmax = dmax
        self.d2 = d2

        # Initial conditions 
        self.x0 = x0
        self.y0 = y0
        self.elimination_thres = elimination_thres

        # Fixed Treatments  
        self.max = dmax
        self.moderate = dmax/2
        self.low = dmax/4
        self.none = 0
        self.treatments = [self.max, self.moderate, self.low, self.none]


    # Binary treatment, returns dtr during treatment period, d2 otherwise 
    def __d1_piecewise(self, t: float, dtr: float, ttr: int, t0: int) -> float:
        """ Piecewise function returning a death rate depending on whether treatment is taking place 

        Parameters
        -----
        t: float
            Time of treatment 
        dtr: float
            Rate of death corresponding to the treatment dosage level
        ttr: int
            Length of on-treatment 
        t0: int
            Length of off-treatment 
        
        Returns
        -----
        float
            returns dtr during on-treatment and d2 during off-treatment 
        """

        cycle_len = ttr + t0
        return dtr if t % cycle_len < ttr else self.d2
    

    # Adaptive treatment, determine treatment depending on the value of y
    def __d1_adaptive(self, t: int, z: np.ndarray, ttr: int, t0: int) -> float:
        """ Function returning the maximum dosage if resistant cells are eliminated otherwise 
        low treatment during the on-treament period and d2 otherwise

        Parameters
        -----
        t: float
            Time of treatment
        z: np.ndarray
            A list containing two elements:
            - z[0] is x, the susceptible population
            - z[1] is y, the drug-resistant population  
        ttr: int
            Length of on-treatment 
        t0: int
            Length of off-treatment 

        Returns
        -----
        float
            Returns max if resistant cells eliminated and low otherwise during on-treatment
            and d2 during off-treatment 
        """

        cycle_len = ttr + t0
        x, y = z

        if y < self.elimination_thres:
            dose = self.max 
        else:
            dose = self.low

        dtr = dose if t % cycle_len < ttr else self.d2

        return dtr


    def __tumor_ODE(self, t: float, z: np.ndarray, dtr: float, ttr: int, t0: int, adaptive: bool): 
        """ System of differential equations describing tumor cell dynamics, checking to see
        if a population of cells has been eliminated 

        Parameters
        -----
        t: float
            Time of treatment
        z: np.ndarray
            A list containing two elements:
            - z[0] is x, the susceptible population
            - z[1] is y, the drug-resistant population
        dtr: float
            Rate of death corresponding to the treatment dosage level
        ttr: int
            Length of on-treatment 
        t0: int
            Length of off-treatment 
        adaptive: bool
            True if adaptive treatment strategy is used, False for constant dosage

        Returns
        -----
        list of float
            A list containing two elements:
            - dx_dt (float): Rate of change of the susceptible population
            - dy_dt (float): Rate of change of the drug-resistant population
        """

        x = max(0, z[0])
        y = max(0, z[1])

        if x <= self.elimination_thres:
            dx_dt = 0
        else:
            if adaptive:
                d1 = self.__d1_adaptive(t, z, ttr, t0)
            else:
                d1 = self.__d1_piecewise(t, dtr, ttr, t0)
            dx_dt = self.r1 * x * (1 - (x + y) / self.k ) - d1 * x
            
        if y <= self.elimination_thres:
            dy_dt = 0
        else:
            dy_dt = self.r2 * y * (1 - (x + y) / self.k) - self.d2 * y

        return [dx_dt, dy_dt]
    
    
    def solve_IVP(self, adaptive: bool, ttr: int, t0: int, t_span: list[int], dtr: float=None):
        """ Solve the ODE described by tumor_ODE

        Parameters
        -----
        adaptive: bool
            True if adaptive treatment strategy is used, False for constant dosage
        ttr: int
            Length of on-treatment 
        t0: int
            Length of off-treatment 
        t_span: list[int]
            A list containing two elements:
            - Starting time of the system
            - Ending time of the system
        dtr: float
            Rate of death corresponding to the treatment dosage level

        Returns
        -----
        Bunch
            A scipy Bunch object containing information of the solved IVP
        """

        t_eval = np.arange(t_span[0], t_span[1] + 1, 1)

        res = solve_ivp(
            self.__tumor_ODE,
            t_span,
            [self.x0, self.y0],  
            args=(dtr, ttr, t0, adaptive),
            t_eval=t_eval,
            method='BDF'
        )

        # Post-process the solution to ensure values below elimination_thres are set to 0
        for i in range(len(res.t)):
            if res.y[0, i] <= self.elimination_thres:
                res.y[0, i] = 0
            if res.y[1, i] <= self.elimination_thres:
                res.y[1, i] = 0

        return res 
    
    def calculate_burden_elimination(self, res):
        """Function returning peak tumor burden and time to total tumor elimination

        Parameters
        -----
        res: Bunch
            A scipy Bunch object containing information of the solved IVP

        Returns
        -----
        list of float
            A list containing two elements:
            - peak_burden: the peak tumor burden described by x + y
            - time: the value of t for which x and y first reach 0

        """

        x = res.y[0]
        y = res.y[1]
        t = res.t

        peak_burden = np.max(x + y)
        zero_idx = np.intersect1d(np.where(y < 1e-3), np.where(x < 1e-3))
        time = t[zero_idx[0]] if zero_idx.size > 0 else np.inf

        return [peak_burden, time]
    

    def plot_res(self, res, treatment: str):
        """Plotting function for the solved IVP

        Parameters
        -----
        res: Bunch
            A scipy Bunch object containing information of the solved IVP
        treatment: str
            A string describing the treatment used
        """

        x = res.y[0]
        y = res.y[1]
        total_burden = x + y
        t = res.t
        
        sns.set(font='Arial')
        plt.plot(t, x, color='blue', label='Susceptible')
        plt.plot(t, y, color='red', label='Resistant')
        plt.plot(t, total_burden, color='green', linestyle=':', label='Total Tumor Burden')
        plt.axhline(y=self.k, color='black', linestyle='--', linewidth=2)
        
        plt.xlabel('Time (days)', size=12, color='black')
        plt.ylabel('Cells (10^4)', size=12, color='black')
        plt.xticks(size=10, color='black')
        plt.yticks(size=10, color='black')
        plt.title(f"Treatment: {treatment}", size=14, color='black')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.show()