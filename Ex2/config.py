import os
import platform

if platform.system() in ["Darwin", "Linux"]:
    SYSTEM_PATH_SEP = "/"
else: # platform.system() == "Windows":
    SYSTEM_PATH_SEP = "\\"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + f"{SYSTEM_PATH_SEP}"

# DATASET CONFIGS
## houseproices (small debug set)
DATASET_TEST_HOUSEPRICES = PROJECT_ROOT + f"test{SYSTEM_PATH_SEP}test_datasets{SYSTEM_PATH_SEP}test_housePrices.csv"

# moneyball
DATASET_MONEYBALL = PROJECT_ROOT + "Datasets" + SYSTEM_PATH_SEP + "Moneyball" + SYSTEM_PATH_SEP + "baseball.csv"
MONEYBALL_TARGET = "W"
MONEYBALL_FEATURES = [
    "Team",
    "League",
    "Year",
    "RS",
    "RA",
    "W",
    "OBP",
    "SLG",
    "BA",
    "Playoffs",
    "RankSeason",
    "RankPlayoffs",
    "G",
    "OOBP",
    "OSLG"
    ]

# Metro Interstace Traffic
DATASET_METRO = PROJECT_ROOT + "Datasets" + SYSTEM_PATH_SEP + "Metro" + SYSTEM_PATH_SEP + "Metro_Interstate_Traffic_Volume.csv"
METRO_TARGET = "traffic_volume"
METRO_FEATURES = [
    'holiday',
    'temp',
    'rain_1h',
    'snow_1h',
    'clouds_all',
    'weather_main',
    'weather_description',
    'date_time'
    ]

# Superconductivity
## TODO: properly configure Superconductivity set
DATASET_SUPERCONDUCTIVITY = PROJECT_ROOT + "Datasets" + SYSTEM_PATH_SEP + "Superconduct" + SYSTEM_PATH_SEP + "train.csv"
SUPERCONDUCTIVITY_TARGET = "critical_temp"
SUPERCONDUCTIVITY_FEATURES = ['number_of_elements', 'mean_atomic_mass', 'wtd_mean_atomic_mass',
       'gmean_atomic_mass', 'wtd_gmean_atomic_mass', 'entropy_atomic_mass',
       'wtd_entropy_atomic_mass', 'range_atomic_mass', 'wtd_range_atomic_mass',
       'std_atomic_mass', 'wtd_std_atomic_mass', 'mean_fie', 'wtd_mean_fie',
       'gmean_fie', 'wtd_gmean_fie', 'entropy_fie', 'wtd_entropy_fie',
       'range_fie', 'wtd_range_fie', 'std_fie', 'wtd_std_fie',
       'mean_atomic_radius', 'wtd_mean_atomic_radius', 'gmean_atomic_radius',
       'wtd_gmean_atomic_radius', 'entropy_atomic_radius',
       'wtd_entropy_atomic_radius', 'range_atomic_radius',
       'wtd_range_atomic_radius', 'std_atomic_radius', 'wtd_std_atomic_radius',
       'mean_Density', 'wtd_mean_Density', 'gmean_Density',
       'wtd_gmean_Density', 'entropy_Density', 'wtd_entropy_Density',
       'range_Density', 'wtd_range_Density', 'std_Density', 'wtd_std_Density',
       'mean_ElectronAffinity', 'wtd_mean_ElectronAffinity',
       'gmean_ElectronAffinity', 'wtd_gmean_ElectronAffinity',
       'entropy_ElectronAffinity', 'wtd_entropy_ElectronAffinity',
       'range_ElectronAffinity', 'wtd_range_ElectronAffinity',
       'std_ElectronAffinity', 'wtd_std_ElectronAffinity', 'mean_FusionHeat',
       'wtd_mean_FusionHeat', 'gmean_FusionHeat', 'wtd_gmean_FusionHeat',
       'entropy_FusionHeat', 'wtd_entropy_FusionHeat', 'range_FusionHeat',
       'wtd_range_FusionHeat', 'std_FusionHeat', 'wtd_std_FusionHeat',
       'mean_ThermalConductivity', 'wtd_mean_ThermalConductivity',
       'gmean_ThermalConductivity', 'wtd_gmean_ThermalConductivity',
       'entropy_ThermalConductivity', 'wtd_entropy_ThermalConductivity',
       'range_ThermalConductivity', 'wtd_range_ThermalConductivity',
       'std_ThermalConductivity', 'wtd_std_ThermalConductivity',
       'mean_Valence', 'wtd_mean_Valence', 'gmean_Valence',
       'wtd_gmean_Valence', 'entropy_Valence', 'wtd_entropy_Valence',
       'range_Valence', 'wtd_range_Valence', 'std_Valence', 'wtd_std_Valence',
       ]


FP_TOLERANCE = 1e-10 # Tolerance for floating point operations for unit tests