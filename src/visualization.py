# src/visualization.pyparams_info = params_str

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.manifold import TSNE
try:
    import umap
except Exception:
    umap = None
import numpy as np
import tensorflow as tf # TensorFlow ist jetzt hier notwendig für tf.gather und tf.reshape
from typing import Optional, Dict, Any, Tuple, Iterable
from .config import CATEGORY_COLORS # Sicherstellen, dass dies korrekt ist, ggf. direkt die Farben definieren
# from .models import VQVAE # Importieren Sie Ihre VQVAE-Klasse, falls Sie sie für Typ-Hints benötigen.
#                         # In der Regel übergeben Sie eine Instanz, so dass der direkte Import nicht zwingend ist.



# ---- Funktion zur Visualisierung des Latenzraums ----
def visualize_latent_space(model, data, labels, n_components=3, method="TSNE", filename="", params_info=""):
    """
    Visualisiert den Latent Space eines (Gaussian) VAE.
    Wir verwenden mu (nicht das gesampelte z), mitteln über die Zeitachse und reduzieren dann per t-SNE/UMAP.
    """
    if not isinstance(model, tf.keras.Model) or not hasattr(model, "encoder"):
        raise ValueError("model muss ein Keras Model sein und ein 'encoder' Attribut besitzen (VAE).")

    print(f"Starte Latent-Space-Visualisierung mit {method} (n_components={n_components})...")

    # --- 1) mu berechnen ---
    if hasattr(model, "get_latent_mu"):
        mu = model.get_latent_mu(data)
        mu = np.asarray(mu)
    else:
        # Fallback: encoder liefert (B, T, 2*latent_dim) => split in mu/logvar
        enc = model.encoder.predict(data, verbose=0)
        enc = np.asarray(enc)
        if enc.shape[-1] % 2 != 0:
            raise ValueError("Encoder-Output hat ungerade letzte Dimension, kann nicht in (mu, logvar) splitten.")
        mu, _ = np.split(enc, 2, axis=-1)

    # mu: (B, T_latent, latent_dim)  -> pro Snippet mitteln
    if mu.ndim != 3:
        raise ValueError(f"Erwartete mu mit ndim=3 (B,T,D), erhalten: shape={mu.shape}")
    X = mu.mean(axis=1)  # (B, latent_dim)

    labels = np.asarray(labels).astype(str)
    if len(X) <= 1:
        print("Nicht genügend Samples für t-SNE/UMAP.")
        return

    # --- 2) Dimensionsreduktion ---
    if method.upper() == "UMAP":
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(X)
        title = f"UMAP Latent Space (mu-mean){params_info}"
    else:
        perplexity_val = min(30, max(5, len(X) - 1))
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=perplexity_val)
        reduced = reducer.fit_transform(X)
        title = f"t-SNE Latent Space (mu-mean, perp={perplexity_val}){params_info}"

    # Farben (optional / fallback schwarz)
    category_colors = {
        "NORM": "#1f77b4", "LVH": "#ff7f0e", "CLBBB": "#2ca02c", "CRBBB": "#d62728",
        "IRBBB": "#9467bd", "LAFB": "#8c564b", "WPW": "#e377c2", "1AVB": "#7f7f7f",
        "AFIB": "#17becf", "AFLT": "#aec7e8", "IVCD": "#bcbd22",
    }
    point_colors = [category_colors.get(lab, "#000000") for lab in labels]

    # --- 3) Plot: 2D oder 3D ---
    if n_components == 2:
        fig = go.Figure(data=[go.Scatter(
            x=reduced[:, 0],
            y=reduced[:, 1],
            mode="markers",
            marker=dict(size=7, color=point_colors),
            text=labels,
            hoverinfo="text",
        )])
        fig.update_layout(
            title=title,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            margin=dict(l=0, r=0, b=0, t=40),
        )
    elif n_components == 3:
        fig = go.Figure(data=[go.Scatter3d(
            x=reduced[:, 0],
            y=reduced[:, 1],
            z=reduced[:, 2],
            mode="markers",
            marker=dict(size=6, color=point_colors),
            text=labels,
            hoverinfo="text",
        )])
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
    else:
        raise ValueError("n_components muss 2 oder 3 sein.")

    if filename:
        pio.write_html(fig, file=filename, include_plotlyjs="cdn", auto_open=False)
        print(f"Latent-Space-Visualisierung gespeichert unter: {filename}")
    else:
        print("[WARN] Kein filename angegeben -> Plot wurde nicht gespeichert.")


# ---- Funktion zur Visualisierung der EKG-Rekonstruktionen ----
# Diese Funktion bleibt gleich, da sie nur Original und Rekonstruktion plottet
def plot_ecg_reconstructions(original_ecg_snippets, reconstructed_ecg_snippets, num_examples=5, filename="", params_info=""):
    """
    Plottet eine Vergleichsansicht von originalen und rekonstruierten EKG-Snippets.

    Args:
        original_ecg_snippets (np.array): Die ursprünglichen EKG-Snippets.
        reconstructed_ecg_snippets (np.array): Die vom Autoencoder rekonstruierten EKG-Snippets.
        num_examples (int): Anzahl der Beispiele, die geplottet werden sollen.
        filename (str): Dateiname für die Speicherung des Plots.
    """
    # Sicherstellen, dass die Anzahl der Kanäle korrekt ist (z.B. 12 für PTB-XL)
    # Und dass die Snippets die Form (num_samples, snippet_length, num_channels) haben
    
    total_channels = original_ecg_snippets.shape[2] # Annahme: letzter Dim ist Kanäle
    snippet_length = original_ecg_snippets.shape[1] # Annahme: mittlere Dim ist Länge

    # Erstelle Subplots: num_examples Zeilen, 2*total_channels Spalten (Original + Rekonstruktion pro Kanal)
    fig, axes = plt.subplots(num_examples, total_channels * 2, figsize=(20, num_examples * 2), squeeze=False)

    for i in range(num_examples):
        original_snippet = original_ecg_snippets[i]
        reconstructed_snippet = reconstructed_ecg_snippets[i]

        for channel in range(total_channels):
            # Original EKG
            ax_orig = axes[i, channel * 2]
            ax_orig.plot(original_snippet[:, channel])
            if i == 0: # Titel nur für die erste Zeile
                ax_orig.set_title(f'Orig. Ch {channel+1}', fontsize=8)
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])
            
            # Rekonstruiertes EKG
            ax_recon = axes[i, channel * 2 + 1]
            ax_recon.plot(reconstructed_snippet[:, channel])
            if i == 0: # Titel nur für die erste Zeile
                ax_recon.set_title(f'Rec. Ch {channel+1}', fontsize=8)
            ax_recon.set_xticks([])
            ax_recon.set_yticks([])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()






