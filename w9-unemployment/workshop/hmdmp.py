import numpy as np
from scipy.stats import truncnorm, norm
from scipy import optimize
import Tasmanian
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numba import jit
import sys
from sys import exit

class hmdmp:
    """Python class for solving the Hagedorn-Manovskii version of
     Diamond-Mortensen Pissarides model using Deterministic 
     Parametrized Expectations Approach (PEA). 
    
    { Min Weight Resids: Global, Finite Elements: Local}

    Note:
    
    The usage of the term GRID here is consistent with TASMANIAN 
    and the Sparse Grid literature's definition: It is a collection 
    of sparse coordinates (interpolation nodes) approximating 
    multidimensional domains of (unknown) functions, tensor basis 
    functions, and, quadrature weights for integration on those nodes.

    (c) 2020++, T. Kam (URL: phantomachine.github.io)
    """

    def __init__(self, 
                    LaborForceSize = 1.0,      # Normalized LF population
                    β = 0.99**(1.0/12.0),      # Discount factor - weekly freq
                    b = 0.955,                 # utility value of outside option
                    ι = 0.407,                 # Match elasticity w.r.t θ
                    η = 0.052,                 # Bargaining power: worker
                    ξ = 0.449,                 # vacancy cost - power term
                    κ_k = 0.474,               # vacancy cost - capital cost
                    κ_w = 0.11,                # vacancy cost - labor cost
                    s = 0.0081,                # Match destruction probability
                    ρ=0.9895,                  # log TFP - AR1 persistence
                    σ=0.0034,                  # log TFP - AR1 shock s.d.
                    logZ_ss=0.0,               # log TFP - determ. constant
                    e_min_scalelog=-4.0,       # log TFP - AR1 shock l.b.
                    e_max_scalelog=4.0,        # log TFP - AR1 shock u.b.
                    MAXITER=500,               # Max. iteration of TI-PEA
                    TOL=1e-8):                 # Convergence tol. TI-PEA

                # ------------------- Properties ----------------------------------

        # Smolyak Sparse Grid + Chebychev Interpolation Scheme
        # Or Leja sequential sparse grid rules + global or local Lagrange polys
        # Dimension of function input (domain)
        # self.iNumInputs = iNumInputs # using two inputs for testing

        # Dimension of function output (codomain)
        # self.NumOutput = NumOutput # 1D (real line)
        # self.NumOutput = NumOutput

        # Model parameters
        self.LaborForce = LaborForceSize
        self.β = β   
        self.b = b
        self.ι = ι
        self.η = η
        self.ξ = ξ
        self.κ_k = κ_k
        self.κ_w = κ_w  
        self.s = s
        self.ρ = ρ   
        self.σ = σ   
        self.logZ_ss = logZ_ss  
        self.ε_mean = 0.0
        self.Zss = np.exp(((1.0-ρ)*logZ_ss + σ*self.ε_mean)/(1.0-ρ))

        # Bounds on the level of log-normal shock e
        self.e_min_scalelog = e_min_scalelog
        self.e_max_scalelog = e_max_scalelog
        self.e_min = np.exp(e_min_scalelog*self.σ)
        self.e_max = np.exp(e_max_scalelog*self.σ)
        self.bounds_shocks = np.array([[self.e_min, self.e_max],])

        # Bounds on the level of TFP (Z)
        self.Z_min = np.exp(e_min_scalelog*σ/np.sqrt(1.0-ρ**2.0))
        self.Z_max = np.exp(e_max_scalelog*σ/np.sqrt(1.0-ρ**2.0))
        self.bounds_exo = np.array([[self.Z_min, self.Z_max],]) # Note 2D array!
        self.NumOutput_exo = 1 # Dimension (co-domain)
        self.iNumInputs_exo = self.bounds_exo.shape[0] # Dimension (domain)

        # Exogenous state space of Z (for quadrature)
        # σ is s.d. of normal distro of AR(1) shock for log(Z)
        self.σ_lognormal = (np.exp(σ**2.0)*(np.exp(σ**2.0)-1.0))**0.5

        # Joint state space (X = Z) (for interpolations)
        self.bounds = np.array([[self.Z_min, self.Z_max],])
        self.NumOutput = 1 # Dimension (co-domain)
        self.iNumInputs = 1 # Dimension (domain)

        # Precision settings
        self.MAXITER_value = MAXITER
        self.TOL_value = TOL       # stopping criterion: operator iteration

        # Smoothing parameter(default=1.0)
        self.SMOOTH = 1.0

    ## ----- MODEL PRIMITIVES -------------------------------------
    def q(self, θ):
        """Probability of filling a vacancy by the end of one period.
        Function is derived analytically from the assumption of a 
        Telegraph-line matching process. Key parameter is ι 
        (Greek: "iota") which governs the elasticity of matching with
        respect to market tightness."""
        if θ <= 0.0:
            qval = 1.0
        else:
            qval = (1.0 + θ**self.ι)**(-1.0/self.ι)
        # if qval < 0.0:
        #     qval = 1.0
        return qval

    def q_inv(self, qval):
        """The inverse function of q. Given qval, compute implied market 
        tightness value θval at qval. Function q is the probability of 
        filling a vacancy by the end of one period. Function is derived 
        analytically from the assumption of a Telegraph-line matching 
        process. Key parameter is ι (Greek: "iota") which governs the 
        elasticity of matching with respect to market tightness."""
        # if qval >= 1.0:
        #     θval = 0.0
        # elif qval <= 0.0:
        #     θval = 0.0
        # else:
        #     # if qval <= 0.0:
        #     #     qval = 1e-12
        if (qval <= 0.0) or (qval >= 1.0):
            qval = 1.0
        θval = (qval**(-self.ι) - 1.0)**(1.0/self.ι)
        return θval

    def cost(self, Z):
        """Hagedorn-Manovskii's reduced-form, per-period vacancy-posting
        cost function. Given state of technology Z, we have date-t cost of
        posting vacancy as κ_t = cost(Z).

        Note: 
        If parameters are such that κ_k = 0 and ξ = 0, 
        then κ_t = κ_w is the textbook per-period fixed cost in DMP."""
        return self.κ_k*Z + self.κ_w*Z**self.ξ 

    def WS(self, Z, θ):
        """Nash bargaining solution - Wage Setting equation, under the 
        assumption of linear production technology in labor (so marginal 
        product of labor is given by technology X), and, no disutility of 
        labor supply."""
        κ = self.cost(Z)
        return self.η*(Z + κ*θ) + (1.0 - self.η)*self.b

    def Bathtub(self, N, θ):
        """Employment dynamics given initial employment state N,
        and, givem market tightness θ"""
        U = (self.LaborForce-N)
        V = U*θ
        N_next = (1.0 - self.s)*N + self.q(θ)*V
        return {"employment": N_next, "unemployment": U, "vacancy": V}

    def Output(self, Z, N):
        return Z*N

    def SteadyState(self):
        """Non-stochastic steady-state equilibrium. 
        Used as reference point for solution space and also 
        for simulations."""
                    
        Zss = self.Zss
        # Employment/posting cost
        κss = self.cost(Zss)
        # Exists unique steady state θss, assuming λss = 0
        jcws = lambda θ: (1.0-self.β*(1.0-self.s))/self.q(θ) \
                - self.β*((1.0-self.η)*(Zss - self.b) - self.η*κss*θ)
        θss = optimize.brentq(jcws, 1e-6, 1e+3)
        # Beveridge curve
        bc = lambda N: self.s*N/(1.0-N) - self.q(θss)*θss
        Nss = optimize.brentq(bc, 1e-6, self.LaborForce-1e-6)
        # Identities
        Uss = self.LaborForce - Nss
        Vss = θss*Uss
        # Assume V = 0 not binding at SS
        λss = 0.0
        # Wage setting
        Wss = self.WS(Zss, θss)
        # Production function
        Yss = Zss*Nss

        out = { "technology"    : Zss,
                "tightness"     : θss,
                "employment"    : Nss, 
                "unemployment" : Uss,
                "vacancy"       : Vss, 
                "multiplier"    : λss,
                "wage"          : Wss,
                "output"        : Yss,
            }

        return out

    ## ----- TOOLS ------------------------------------------------
    def supnorm(self, function1, function2):
        """Returns the absolute maximal (supremum-norm) distance between
        two arbitrary NumPy ND-arrays (function coordinates)"""
        return (np.abs(function1 - function2)).max()

    def logNormal_cdf(self, x, μ, σ):
        """
        Univariate log-Normal cumulative density function.

        Parameters:
        -----------
        x (float) the argument of the PDF. x > 0
        μ, σ (floats) parameters of the normal PDF of log values, 
            i.e., log(x). σ > 0

        Returns
        -------
        cdf (float, array), the value of the cumulative density function at each x.

        Depends on SciPy stats.norm.
        """
        if (x <= 0.0):
            cdf = 0.0
        else:
            logx = np.log(x)
            cdf = norm.cdf(logx, loc=μ, scale=σ)
        return cdf

    def logNormal_pdf (self, x, μ, σ):
        """
        Univariate log-Normal probability density function.

        Also known as the Cobb-Douglas PDF or the Anti log-Normal PDF. The Log 
        Normal PDF describes a random variable X whose logarithm, log(X), is 
        normally distributed.

        Parameters:
        -----------
        x (float) the argument of the PDF. x > 0
        μ, σ (floats) parameters of the normal PDF of log values, 
            i.e., log(x). σ > 0

        Returns
        -------
        pdf (float, array), the value of the cumulative density function at each x.
        """
        if (x <= 0.0):
            pdf = 0.0
        else:
            denominator = x*σ*np.sqrt(2.0*np.pi)
            pdf = np.exp(-0.5*((np.log(x) - μ)/σ)**2.0)/denominator
        return pdf

    def logNormalTruncated_pdf(self, x, μ, σ, a, b):
        """
        Univariate Truncated log-Normal probability density function. 
        Support of this distribution is [a,b].

        Parameters:
        -----------
        x (float) the argument of the PDF. x > 0
        μ, σ (floats) parameters of the log normal PDF of log values, 
            i.e., log(x). σ > 0
        a, b (floats) the lower and upper truncation limits: a < x < b. 
            Note that a >= 0, since x > 0.

        Returns
        -------
        pdf (float, array), the value of the prob density function at each x.
        """
        # Check for illegal inputs
        if (self.σ <= 0.0):
            print("logNormalTruncated_pdf: Must have sigma > 0!")
            exit
        if (b <= a):
            print("logNormalTruncated_pdf: Illegal bounds. Must have b > a!")
            exit
        if (a < 0.0):
            print("logNormalTruncated_pdf: Illegal bounds. Must have a >= 0!")
            exit

        # Evaluate pdf
        if (x <= a) or (b <= x):
            pdf = 0.0
        else:
            lncdf_a = self.logNormal_cdf(a, μ, σ)
            lncdf_b = self.logNormal_cdf(b, μ, σ)
            lnpdf_x = self.logNormal_pdf(x, μ, σ)
            pdf = lnpdf_x / (lncdf_b - lncdf_a)
        return pdf

    def ar1_density_conditional(self, z):
        """
        AR(1) model for log(Z). Get pdf of Z_next conditional on knowing 
        current Z. Default assumes that steady-state (unconditional mean) of
        log(Z) = 0. 

        Model is:
        Z_next = exp(logZ_ss)**(1-ρ) * (Z**ρ) * exp(σ*ε_next)
        ε_next ~ TruncNorm(0, 1)

        Parameters:
        -----------
        z (float) the level of *realized* random variable Z
        logZ_ss (float) the unconditional mean of log(Z)
        ρ, σ (floats) persistence and std deviation parameters. |ρ| < 1, σ > 0.

        Returns
        -------
        pdf_z (float, array), the value of the conditional prob density 
        function of z_next, i.e, conditional on a given z.
        """
        # Mean of log(Z_next) process conditional on log(Z)
        μ = (1-self.ρ)*self.logZ_ss + self.ρ*np.log(z)
        # Bounds on Z_next - BOUNDS_EXO defined in 2D array format for TASMANIAN
        # a, b = float(self.bounds_exo[:, 0]), float(self.bounds_exo[:, 1])
        # Assumes shocks are log-Normal so, conditional distro also log-Normal
        # Example: y := z_next
        pdf_znext = lambda y: self.logNormal_pdf(y, μ, self.σ)
        return pdf_znext

    def ar1_density_conditional_truncnorm(self, z):
        """
        AR(1) model for log(Z). Get pdf of Z_next conditional on knowing 
        current Z. Default assumes that steady-state (unconditional mean) of
        log(Z) = 0. 

        Model is:
        Z_next = exp(logZ_ss)**(1-ρ) * (Z**ρ) * exp(σ*ε_next)
        ε_next ~ TruncNorm(0, 1)

        Parameters:
        -----------
        z (float) the level of *realized* random variable Z
        logZ_ss (float) the unconditional mean of log(Z)
        ρ, σ (floats) persistence and std deviation parameters. |ρ| < 1, σ > 0.

        Returns
        -------
        pdf_z (float, array), the value of the conditional prob density 
        function of z_next, i.e, conditional on a given z.
        """
        # Mean of log(Z_next) process conditional on log(Z)
        μ = (1-self.ρ)*self.logZ_ss + self.ρ*np.log(z)
        # Bounds on Z_next - BOUNDS_EXO defined in 2D array format for TASMANIAN
        a, b = float(self.bounds_exo[:, 0]), float(self.bounds_exo[:, 1])
        # Assumes shocks are Trunc log-Normal so, conditional distro also 
        # Trunc log-Normal
        # Example: y := z_next
        pdf_znext = lambda y: self.logNormalTruncated_pdf(y, μ, self.σ, a, b)
        return pdf_znext

    def ar1(self, z, ε, log_level=False, shock_scale ="sd"):
        """
        Define AR(1) model for TFP shock process.
        """
        ρ = self.ρ
        σ = self.σ
        Zss = self.Zss
        if shock_scale == "unit":
            shock = ε
        elif shock_scale == "sd":
            shock = σ*ε
        else:
            # option to introduce float scaling
            shock = shock_scale*ε

        # Option to log-linear ar1 or exponential form:
        if log_level==False:
            znext = (Zss**(1.0-ρ))*(z**ρ)*np.exp(shock)
        else:
            znext = (1.0-ρ)*np.log(Zss) + ρ*z + shock
        return znext
        
    def StatusBar(self, iteration, iteration_max, stats1, width=15):
        percent = float(iteration)/iteration_max
        sys.stdout.write("\r")
        progress = ""
        for i in range(width):
            if i <= int(width * percent):
                progress += "="
            else:
                progress += "-"
        sys.stdout.write(
            "[ %s ] %.2f%% %i/%i, error = %0.10f    "
            % (progress,percent*100,iteration,iteration_max,stats1)
            )
        sys.stdout.flush()

    def MakeAllPolyGrid(self, Depth, Order, 
                            sRule="localp", TypeDepth="level"):
        """Create sparse grid given parameters and interpolant type. Used for 
        interpolation of functions over joint state space containing both 
        *current* endogenous and exogenous states. See MakeExoPolyGrid() for 
        grid subspace defined over exogenous states only (used for integration 
        w.r.t. distribution of exogenous states). 
        
        Dependencies: Uses TASMANIAN.

        Parameters:
        -----------
        iNumInputs (int),   Dimension of function input (domain)
        NumOutput  (int),   Dimension of function output (codomain)
        Depth (int),        Non-negative integer controls the density of grid 
                            The initial 
                            construction of the local grids uses tensor 
                            selection equivalent to TasGrid::type_level
                            Depth is the L parameter in the formula in TASMANIAN
                            manual; i.e., the "level" in Smolyak. 
        Order (int),        Integer no smaller than -1.
                            1 : indicates the use of constant and linear
                                functions only
                            2 : would allow quadratics (if enough points are 
                                present)
                           -1 : indicates using the largest possible order for 
                                each point.
        
        sRule (str),        Choose from Tasmanian.lsTsgGlobalRules. 
                            (Know your onions!)

        TypeDepth (str),    Choose from Tasmanian.lsTsgGlobalRules. 
                            (Know your onions!)

        Returns
        -------
        grid (obj),         Grid scheme: domain, quadrature weights, etc. See 
                            TASMANIAN manual.
        """
        
        # Step 1. Define sparse grid and local poly rule
        if sRule=="localp":
            grid = Tasmanian.makeLocalPolynomialGrid(self.iNumInputs, 
                                                    self.NumOutput,
                                                        Depth, Order, sRule)
        elif sRule=="chebyshev":
            grid = Tasmanian.makeGlobalGrid(self.iNumInputs, 
                                            self.NumOutput,
                                            Depth, 
                                            TypeDepth,
                                            sRule)

        # Step 2. Transform to non-canonical domain.
        # self.bounds is np array of domain bounds
        # e.g., np.array([[self.K_min, self.K_max], [self.Z_min, self.Z_max]])
        grid.setDomainTransform(self.bounds)
        return grid

    def MakeExoPolyGrid(self, Depth, Order, 
                        sRule="localp", TypeDepth="level"):
        """Like MakeAllPolyGrid() this method defines only sparse grid
        over exogenous random variables' state space. Used for computing 
        conditional expectations (integrals) of functions of endo- and 
        exo-genous states, over subspace of *exogenous*, *future* random 
        variables. 
        
        Dependencies: Uses TASMANIAN.
        
        Parameters:
        -----------
        iNumInputs (int),   Dimension of function input (domain)
        NumOutput  (int),   Dimension of function output (codomain)
        Depth (int),        Non-negative integer controls the density of grid 
                            The initial 
                            construction of the local grids uses tensor 
                            selection equivalent to TasGrid::type_level
                            Depth is the L parameter in the formula in TASMANIAN
                            manual; i.e., the "level" in Smolyak. 
        Order (int),        Integer no smaller than -1.
                            1 : indicates the use of constant and linear
                                functions only
                            2 : would allow quadratics (if enough points are 
                                present)
                           -1 : indicates using the largest possible order for 
                                each point.
        
        sRule (str),        Choose from Tasmanian.lsTsgGlobalRules. 
                            (Know your onions!)

        TypeDepth (str),    Choose from Tasmanian.lsTsgGlobalRules. 
                            (Know your onions!)

        Returns
        -------
        grid_exo (obj),     Grid scheme: domain, quadrature weights, etc. See 
                            TASMANIAN manual.
        """

        # Step 1. Define sparse grid and local poly rule
        # Step 2. Transform to non-canonical domain.
        # self.bounds is np array of domain bounds for exog vars
        # e.g., np.array([[self.Z_min, self.Z_max]])
        if sRule=="localp":
            grid_exo = Tasmanian.makeLocalPolynomialGrid(self.iNumInputs_exo, 
                                                        self.NumOutput_exo,
                                                          Depth, Order, sRule)
            
            grid_exo.setDomainTransform(self.bounds_exo)
        elif sRule=="chebyshev" or sRule=="gauss-chebyshev1":
            grid_exo = Tasmanian.makeGlobalGrid(self.iNumInputs, 
                                                self.NumOutput,
                                                    Depth, 
                                                    TypeDepth,
                                                    sRule)
            grid_exo.setDomainTransform(self.bounds_exo)
        elif sRule=="gauss-hermite":
            # Optimized for integration on [-∞, ∞]
            grid_exo = Tasmanian.makeGlobalGrid(self.iNumInputs, 
                                                self.NumOutput,
                                                    Depth, 
                                                    TypeDepth,
                                                    sRule)
            # grid_exo.setDomainTransform(self.bounds_exo) 
        elif sRule=="gauss-legendre":
            # Optimized for integration on [a, b]
            grid_exo = Tasmanian.makeGlobalGrid(self.iNumInputs, 
                                                self.NumOutput,
                                                    Depth, 
                                                    TypeDepth,
                                                    sRule)
            grid_exo.setDomainTransform(self.bounds_exo)
        return grid_exo

    def InterpSparse(self, grid_all, Y, Xi):
        """Given raw data points Y use local, piecewise polynomials for 
           Lagrange Interpolation over localized sparse grid. Then calculate 
           batch interpolated values for new data Yi outside of the grid. 
           GRID_ALL is obtained from MakeAllPolyGrid(). 

        Parameters:
        -----------
        grid_all (obj),    sparse grid object (TASMANIAN). grid_all.getPoints()
                           give (Npoints x Nvars) sparse grid domain of 
                           future exogenous states
        Y (array),        (Npoints x Nvars) Function values (data) defined on 
                           grid_all 
        Xi (array),       (Npoints_i x Nvars) coordinates on domain not 
                           defined in grid_all (to be interpolated over)

        Returns
        -------
        Yi (array),        (Npoints_i x Nvars) interpolated values on Xi 
        """

        # Step 1. Lagrange Interpolation to define interpolant over supplied 
        # needed data points Y defined over elements in X := grid.getPoints()
        grid_all.loadNeededPoints(Y)

        # Step 2. Interpolate values get Yi values over Xi points not on grid X
        # Data must be np.array of size (N_obs, iNumInputs) where 
        # iNumInputs = dim(Domain) = bounds.shape[0]
        Yi = grid_all.evaluateBatch(Xi)

        return Yi

    def PEA_deterministic(self, grid_all, grid_exo, efun_old):
        """Take as given data for expectation function evaluated at sparse 
        GRID_ALL, efun_old. Evaluate the Euler operator once to get update of 
        this as efun_new. This evaluation over grid points involves: 
            (1) Checking for KKT complementary slackness condition(s)
            (2) Quadrature integral approximation to construct efun_new
        Implements a version of Christiano and Fisher's non-stochastic PEA 
        method. Benefit is that we don't need to do costly nonlinear 
        Newton-Raphson type local root solvers. (This is a special feature of 
        this model though!)

        This is also where most custom model definition is specified, in terms 
        of its Recursive Equilibrium (Euler, Constraints) conditions. 

        Note to selves: 
        ~~~~~~~~~~~~~~~
        Future devs should make this as model-free as possible, in the style of 
        Miranda and Fackler's CompEcon MATLAB toolbox or the incomplete DOLO!

        Parameters:
        -----------
        grid_all (obj),    sparse grid object (TASMANIAN). grid_all.getPoints()
                           give (Npoints x Nvars) sparse grid domain of 
                           future exogenous states
        grid_exo (obj),    sparse grid object (TASMANIAN). grid_exo.getPoints()
                           give (Npoints_exo x Nvars_exo) sparse grid domain of 
                           future exogenous states
        efun_old (array),  (Npoints x 1) Expectation Function values on 
                           grid_all 

        Returns
        -------
        efun_new (array), (Npoints x 1) updated Expectation Function values 
                          on grid_all 
        θ  (array),       (Npoints x 1) updated market tightness function 
                           values on grid_all 
        λ  (array),       (Npoints x 1) updated market tightness function
                            values on grid_all                   
        """

        # Note: grid_exo is "pre-computed" outside of PEA iteration!
        Z_next_grid = grid_exo.getPoints()
        zWeights = grid_exo.getQuadratureWeights()
        w = np.atleast_2d(zWeights) # Special case 1D shock, convert to 
                                    # 2D array. Keep TASMANIAN happy!

        # Array C (control) is of dimensionality: grid_all.shape
        X = grid_all.getPoints()
        θ = np.zeros(X.shape[0])
        λ = θ.copy()

        # Pre-allocate - we'll update this at end of loop (in Step 4)
        efun_new = efun_old.copy()

        # print(efun_old.shape)

        # Loop over all current states
        for idx_state, state in enumerate(X):

            # Current state :=: idx_state
            z = state[0]
    
            # STEP 1.
            # Guess of current qval = q(θ) given efun_old(.)
            m = efun_old[idx_state]
            
            κ = self.cost(z)
            qval = κ/m
            λ[idx_state] = 0.0
            if (qval <= 0.0) or (qval >= 1.0):
                qval = 1.0
                λ[idx_state] = κ - m           
            θ[idx_state] = self.q_inv(qval)
            

            # STEP 3. - Interpolate efun_old, Evaluate RHS integrand components
            #            
            # Conditional pdf of z_next **given** current Z = z:
            # WARNING: Currently support only truncnorm option!
            # (LAMBDA function): Evaluate Qz at any point z_next as: Qz(z_next)
            Qz = self.ar1_density_conditional_truncnorm(z)
            Intg = np.zeros(Z_next_grid.shape)
            
            for idx_znext, znext in enumerate(Z_next_grid):
                # Interpolation - 2D array format for TASMANIAN
                # Next period value of expectation function at znext
                zi = np.zeros((1,1))
                zi[:,0] = znext
                m_next = self.InterpSparse(grid_all, efun_old, zi)
                if m_next > efun_old.max():
                    m_next = efun_old.max()
                if m_next < efun_old.min():
                    m_next = efun_old.min()
                # Next-period employment cost at state znext
                κ_next = self.cost(znext)
                λ_next = 0.0
                # Next period: Match prob., tightness, multiplier
                qval_next = κ_next/m_next
                # KKT check - override if V = 0 binding
                if (qval_next <= 0.0) or (qval_next >= 1.0):
                    qval_next = 1.0
                    λ_next = κ_next - m_next
                θ_next = self.q_inv(qval_next)
                    
                # Define integrand on RHS of Euler eqn
                gval = znext - self.WS(znext, θ_next) \
                        + (1.0-self.s)*(κ_next/qval_next - λ_next)
                Intg[idx_znext] = self.β*gval*Qz(znext)
            
            # STEP 4.
            # Evaluate integral using quadrature scheme on grid_exo 
            Integrand = Intg*w.T
            # RHS_Euler = np.sum(Integrand), update expectation fn guess
            efun_new[idx_state] = np.sum(Integrand)

        return efun_new, θ, λ

    def Solve_PEA_TimeIteration(self, grid_all, grid_exo, 
                                                efun_old=None, DISPLAY=True):
        """Start with an initial guess of the expectation function (typically 
        the RHS of your Euler equation), efun_old. Iterate on a version of the 
        (Wilbur Coleman III)-cum-(Kevin Reffett) operator in terms of a 
        PEA_deterministic() operator defined in this same class, until 
        successive approximations of the equilibrium expectations function 
        converge to a limit. This allow you to then back out the solution's 
        implied equilibrium policy function(s).

        Note to selves: 
        ~~~~~~~~~~~~~~~
        Next-phase development: Endogenize grid_all to exploit adaptive sparse 
        grid capabilities starting from the most dense grid_all. Idea: as we 
        iterate closer to a solution, there may be redundant grid elements (in 
        terms of a surplus criterion). Then we might consider updating grid_all 
        with successively sparser grid_all refinements.

        Parameters:
        -----------
        Let X := grid_all.getPoints() from TASMANIAN.

        grid_all (obj),    sparse grid object (TASMANIAN). grid_all.getPoints()
                           give (Npoints x Nvars) sparse grid domain of 
                           future exogenous states
        grid_exo (obj),    sparse grid object (TASMANIAN). grid_exo.getPoints()
                           give (Npoints_exo x Nvars_exo) sparse grid domain of 
                           future exogenous states
        efun_old (array), (Npoints x Nvars) Expectation Function values on 
                          grid_all 

        Returns
        -------
        efun_new (array), (Npoints x 1) fixed-point Expectation Function 
                          values on grid_all 
        θ (array),        (Npoints x 1) fixed-point policy (market tightness) 
                          values on grid_all 
        λ (array),       (Npoints x 1) fixed-point multiplier function 
                          values on grid_all (for vacancy constraint)
        """
        if DISPLAY==True:
            print("\n****** Solve_PEA_TimeIteration ***********************\n")

            print("For function interpolations")
            print("----------------------------------")
            print("\tRule: %s" % grid_all.getRule())
            print("\tInterpolation Nodes: %i" % grid_all.getNumPoints()) 
            print("\tLocal Polynomial basis (Finite-Element Method)? %s" 
                                        % grid_all.isLocalPolynomial())
            print("\tGlobal Polynomial basis (Weighted Residuals Method)? %s" 
                                        % grid_all.isGlobal())
            print("\tInterpolation Nodes: %i" % grid_all.getNumPoints())
            print("\tMax. order of polynomials: %i" % grid_all.getOrder())

            print("\nFor quadrature/integration")
            print("----------------------------------")
            print("\tRule: %s" % grid_exo.getRule())
            print("\tInterpolation Nodes: %i" % grid_exo.getNumPoints()) 
            print("\tLocal Polynomial basis (Finite-Element Method)? %s" 
                                        % grid_exo.isLocalPolynomial())
            print("\tGlobal Polynomial basis (Weighted Residuals Method)? %s" 
                                        % grid_exo.isGlobal())
            print("\tInterpolation Nodes: %i" % grid_exo.getNumPoints())
            print("\tMax. order of polynomials: %i" % grid_exo.getOrder())

            print("\n\t\t請稍微等一下 ...")
            print("\t\tதயவுசெய்து ஒரு கணம் காத்திருங்கள் ...")
            print("\t\tしばらくお待ちください ...")
            print("\t\tPlease wait ...\n")

        for j in range(self.MAXITER_value): 
            # Evaluate Euler operator once given efun_old
            efun_new, θ, λ = \
                self.PEA_deterministic(grid_all, grid_exo, efun_old)
            # Compute distance between guesses
            error = self.supnorm(efun_new, efun_old)
            # Update expectation function
            efun_old = self.SMOOTH*efun_new + (1.0-self.SMOOTH)*efun_old
            # Progress bar
            self.StatusBar(j+1, self.MAXITER_value, error)

            # Stopping rules
            if (j == self.MAXITER_value-1) and (error >= self.TOL_value):
                print("\nSolve_PEA_TimeIteration: Max interation reached.")
                print("\t\tYou have not converged below tolerance!")
            if error < self.TOL_value:
                print("\nSolve_PEA_TimeIteration: Convergence w.r.t. TOL_value attained.")
                break

        return efun_new, θ, λ

    def getInitialGuess(self, grid_all):
        # STEP 3: Initial guess of expectation function
        χ = 24.48875328038525
        c = -22.0
        efun_init = c + χ*grid_all.getPoints()
        return efun_init


    def diagnostics_EulerError(self, grid_all, efun, θ, λ,
                                method="informal", PLOT=True):
        """Given numerical solution for policy C and expectation function efun,
        Calculate the Den Haan and Marcet Euler equation errors.
        There are two methods:
        1.  DHM informal method - evaluates error between LHS and RHS of 
            Euler equation over a fine grid.
        2.  DHM formal method - This is akin to a GMM orthogonality conditions. 
            Idea: 
            * Simulate artificial time series of length T. 
            * Compute test statistic: J(T) := T*M'*inv(W)*M, where 
                M = (h(s)f(x, y, y')).sum()/T   # sum over length-T outcomes
                                                # h(s) can be 1, or any vector 
                                                # of variables.
                W = (f(x, y, y')h(s)'h(s)f(x, y, y')).sum()/T
              Test stat J(T) is distributed as χ**2 with |h(s)| degree of 
              freedom
        3. Repeat step 2, N independent times.
        4. Calculate the fraction of times the stat is in the lower- and 
           upper-5% range.
        """
        # policy = self.getPolicy(efun_new, C, mu)
        print("diagnostics_EulerError:")
        if method=="informal":
            
            print("\tCurrently performing 'informal' error diagnostic.")
            print("\tEuler equation error (percentage consumption changes).")
            # Sparse grid domain nodes
            gridPoints = grid_all.getPoints()

            # Define Lambda function objects via InterpSparse()
            θ_interp = self.getPolicy(grid_all, θ)
            λ_interp = self.getPolicy(grid_all, λ)
            e_interp = self.getPolicy(grid_all, (efun.flatten()).T)

            # Uniform rectangular grid for plotting
            iTestGridSize = 1000
            dX = np.linspace(self.Z_min, self.Z_max, iTestGridSize)
            
            # Approximant evaluated over fine mesh
            θ_approxval = θ_interp(dX)
            λ_approxval = λ_interp(dX)
            # Implied tightness function evaluated over same mesh
            θ_implied = self.q_inv(self.cost(dX)/(e_interp(dX) + λ_approxval))
                                   
            # Percentage errors in consumption errors
            error_pcons = np.abs((θ_approxval - θ_implied)/θ_implied)
            error_pcons_max = error_pcons.max()
            error_pcons_avg = error_pcons.mean()

            print("\tMax. Euler (consumption) error = %6.5f percent" % \
                                                    (error_pcons_max*100))
            print("\tMean Euler (consumption) error = %6.5f percent" % \
                                                    (error_pcons_avg*100))

            # Plot figure of errors
            if PLOT==True:
                Xmat = dX
                Ymat = error_pcons

                fig = plt.figure()
                # Surface plot of interpolated values
                plt.plot(Xmat, Ymat)

                # Econ101 label yer bloody axes!
                plt.xlabel(r"$Z$")
                plt.title("Euler Error")
                plt.show(block=False)
            # else:
            #     plt = []
        else:
            print("Currently performing Den-Haan and Marcet's J-test error diagnostic.")
            print("Sorry! This feature is not yet available in your region.")
        return [error_pcons_max, error_pcons, error_pcons_avg, plt]

    def ImpulseResponseFunction(self, grid_all, θ, λ,
                                Horizon=16, 
                                shock_scale="sd", 
                                shock_date=1,
                                experiment="deterministic", 
                                Burn=1000,
                                PLOT=True,
                                irf_percent=False,
                                show_det_steady=False,
                                Display=False,
                                state_init=[],
                                ):

        # Interpolant objects: C and mu functions
        θ_interp = self.getPolicy(grid_all, θ)
        λ_interp = self.getPolicy(grid_all, λ)
        # Initial state at steady state
        # state = np.atleast_2d([self.Kss, self.Zss])
        # kss, zss = state[:, 0].copy(), state[:, 1].copy()
        # Shock series
        if experiment=="deterministic":
            ε = np.zeros(Horizon)
            ε[shock_date] = 1.0
        else:
            e_min = np.log(self.e_min)
            e_max = np.log(self.e_max)
            ε = truncnorm.rvs(e_min, e_max, size=Burn+Horizon)

   
        # Initiate lists of simulated outcomes
        Zss = self.SteadyState()["technology"]
        θss = self.SteadyState()["tightness"]
        Nss = self.SteadyState()["employment"]
        Uss = self.SteadyState()["unemployment"]
        Vss = self.SteadyState()["vacancy"]
        λss = self.SteadyState()["multiplier"]
        Wss = self.SteadyState()["wage"]
        Yss = self.SteadyState()["output"]
        Zs, θs, Ns, Us, Vs, λs, Ws, Ys = [], [], [], [], [], [], [], []

        # Initialize state
        if not state_init:
            z, N = Zss.copy(), Nss.copy()
        else:
            z, N = state_init[0], state_init[1]

        # Make at least 2D for TASMANIAN Evaluate() or EvaluateBatch()
        state = np.atleast_2d([z, N])
        
        # Recursively generate outcomes
        if experiment=="deterministic":
            T = Horizon
        else:
            T = Burn + Horizon
        
        for t in range(T-1):
            # Store current states (k,z)
            Zs.append(z)
            Ns.append(N)
            U = self.LaborForce - N
            Us.append(U)
            # pack into 2d Numpy array to suit interpolation
            # Current outcomes
            y = z*N
            Ys.append(y)      
            # Current consumption
            if t == 0:
                θ = θss
            else:
                θ = θ_interp(state)
            # Wage
            Ws.append(self.WS(z, θ))
            # check for KKT
            λ = λ_interp(state)
            if (λ > self.TOL_value) and (self.q(θ) >= 1.0):
                # Binding constraint case
                λs.append(λ)
                θ = 0.0
                θs.append(θ)
            else:
                # Non-binding constraint case
                λs.append(0.0)
                θs.append(θ)
            # Vacancy
            Vs.append(θ*U)
            # Draw next-period TFP state, update states
            znext = self.ar1(z, ε[t+1])
            z = znext
            # Induce next period N state
            Nnext = self.Bathtub(N, θ)
            state[:,0], state[:,1] = z, N
            # Report progress to screen (default=False)
            if Display:
                self.StatusBar(t+1, T-1, 0.0, width=15)
        
        # Flattened numpy array
        Zs = np.asarray(Zs).flatten()
        θs = np.asarray(θs).flatten()
        Ns = np.asarray(Ns).flatten()
        Us = np.asarray(Us).flatten()
        Vs = np.asarray(Vs).flatten()
        λs = np.asarray(λs).flatten()
        Ws = np.asarray(Ws).flatten()
        Ys = np.asarray(Ys).flatten()

        if irf_percent==True:
            Zs = (Zs - Zss)/Zss
            θs = (θs - θss)/θss
            Ns = (Ns - Nss)/Nss
            Us = (Us - Uss)/Uss
            Vs = (Vs - Vss)/Vss
            Ws = (Ws - Wss)/Wss
            Ys = (Ys - Yss)/Yss
            # Note λss = 0!

        # For time-series sim, remove initial burn-in obs
        if experiment != "deterministic":
            Zs = Zs[Burn::]
            θs = θs[Burn::]
            Ns = Ns[Burn::]
            Us = Us[Burn::]
            Vs = Vs[Burn::]
            λs = λs[Burn::]
            Ws = Ws[Burn::]
            Ys = Ys[Burn::]
            print("\n\tBurn-in sims. of length BURN=%i discarded ..." % Burn)

        # Pack away
        sims = {    
                    "technology"  : { "path" : Zs, "point" : Zss },
                    "tightness"   : { "path" : θs, "point" : θss },
                    "employment"  : { "path" : Ns, "point" : Nss },
                    "unemployment": { "path" : Us, "point" : Uss },
                    "vacancy"     : { "path" : Vs, "point" : Vss },
                    "multiplier"  : { "path" : λs, "point" : λss },
                    "wage"        : { "path" : Ws, "point" : Wss },
                    "output"      : { "path" : Ws, "point" : Yss },
                }

        # Default option to PLOT time-series/IRF figures
        if PLOT == True:
            nvars = len(sims)
            ncol = 2
            nrow = int((nvars + np.mod(nvars, ncol))/ncol)

            fig = plt.figure(facecolor="white", tight_layout=True)
            for idx_key, (key, series) in enumerate(sims.items()):
                T_series = len(series["path"])
                plt.subplot(nrow, ncol, idx_key+1)
                #plt.subplots_adjust(hspace = .001)
                plt.plot(np.arange(T_series), series["path"], 'k.--')
                if (show_det_steady==True):
                    if (irf_percent):
                        plt.plot(np.arange(T_series), np.zeros(T_series), 'r')
                    else:
                        plt.plot(np.arange(T_series), \
                                        series["point"]*np.ones(T_series), 'r')
                
                ax = fig.gca()
                ax.set_ylabel(key)
                # ax.set_xlabel('t')
            plt.show(block=False)
        return sims



