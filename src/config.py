# src/config.py

import os 

# Pfad zum Basisverzeichnes des Projekts (von diesem Skript aus gesehen)
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Konfiguration der Dateipfade ---
DATA_DIR = os.path.join(BASE_PATH,'data','ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
RELEVANT_ECG_PATH = os.path.join(DATA_DIR, 'ekg_zu_scp_codes.csv')
DATABASE_PATH = os.path.join(DATA_DIR, 'ptbxl_database.csv') #database table 

print(DATA_DIR)
print(RELEVANT_ECG_PATH)
print(DATABASE_PATH)
# --- EKG-Verarbeitungsparameter ---
SAMPLING_RATE = 500  # PTB-XL Sampling Rate high resolution
SNIPPET_LENGTH_BEFORE_R = 100 #200ms vor R-Peak
SNIPPET_LENGTH_AFTER_R = 200 #400ms nach R-Peak
MIN_SNIPPETS_PER_FILE = 5 # Mindestanzahl von Snippets pro Datei

# --- Autoencoder-Parameter ---
LATENT_DIMENSIONS_TO_TEST = [32] # Oder [8, 16, 32, 64, 128, 256]
AE_EPOCHS = 50 #von 10 auf 100, dann auf 50 geändert
AE_BATCH_SIZE = 32
 
# --- VAE Spezifische Parameter ---
BETA = 1.0  # oder z.B. 0.1 / 0.5 / 1.0 testen




CATEGORY_COLORS = {
    "NORM": '#1f77b4', "LVH": '#ff7f0e', "CLBBB": '#2ca02c', "CRBBB": '#d62728',
    "IRBBB": '#9467bd', "LAFB": '#8c564b', "WPW": '#e377c2', "1AVB": '#7f7f7f',
    "IVCD": '#bcbd22', "AFIB": '#17becf', "AFLT": '#aec7e8'
}
