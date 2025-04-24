# Compare the bias correction methods from SBCK. This script computes the bias correction for different RCM
# outputs and saves the results as netCDF files. The script bias_correction_plot_comparison_methods_sbck can be used
# to plot the results.
# The methods are:
# - 'QM': Quantile Mapping
# - 'RBC': Random Bias Correction
# - 'IdBC': Identity Bias Correction
# - 'CDFt': Quantile Mapping bias corrector, taking account of an evolution of the distribution
# - 'OTC': Optimal Transport bias Corrector
# - 'dOTC': Dynamical Optimal Transport bias Corrector
# - 'ECBC': Empirical Copula Bias Correction
# - 'QMrs': Quantile Mapping bias corrector with multivariate rankshuffle
# - 'R2D2': Non stationnary Quantile Mapping bias corrector with multivariate rankshuffle
# - 'QDM': QDM Bias correction method
# - 'MBCn': MBCn Bias correction method
# - 'MRec': MRec Bias correction method
# - 'TSMBC': Time Shifted Multivariate Bias Correction
# - 'dTSMBC': Time Shifted Multivariate Bias Correction where observations are unknown.
# - 'AR2D2': Multivariate bias correction with quantiles shuffle

from deepdown.config import Config
from deepdown.bias_correction_sbck import run_bias_correction

# Options for bias correction methods and RCMs
methods = ['QM', 'RBC', 'IdBC', 'CDFt', 'OTC', 'dOTC', 'ECBC', 'QMrs', 'R2D2',
           'QDM', 'MBCn', 'MRec', 'TSMBC', 'dTSMBC', 'AR2D2']
models = ['ALADIN63_CNRM-CM5', 'ALADIN63_MPI-ESM-LR', 'CCLM4-8-17_MIROC5', 'CCLM4-8-17_MPI-ESM-LR',
          'RegCM4-6_CNRM-CM5', 'RegCM4-6_MPI-ESM-LR', 'REMO2015_MIROC5']

# Load the configuration
conf = Config().get()

path_inputs = conf.path_inputs
path_output = conf.path_output

# Iterate over each model and method
for model in models:
    for method in methods:
        print (f"Running bias correction for {model} using {method}")

        # Adjust the input and output paths for the current model
        conf.path_inputs = [
            f'{path_inputs[0]}/{model}',
            f'{path_inputs[1]}/{model}'
        ]
        conf.path_output = f'{path_output}/{method}/{model}'

        # Run the bias correction for each method
        run_bias_correction(conf, method)

print("Bias correction completed for all models and methods.")
