#!/usr/bin/env python3
"""
Generador de gráficos para presentación - Heritage Segmentation Research
Genera visualizaciones profesionales de resultados POC5.5 y proyecciones POC6
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuración estética
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLORS = {
    'MaxViT': '#FF6B6B',  # Rojo (winner)
    'Swin': '#4ECDC4',    # Turquesa
    'ConvNeXt': '#45B7D1', # Azul
    'Hybrid': '#FF6B6B',
    'ViT': '#4ECDC4',
    'CNN': '#45B7D1'
}

# Crear directorio para gráficos
OUTPUT_DIR = Path(__file__).parent / 'graficos'
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Cargar todos los CSVs"""
    base_path = Path(__file__).parent
    return {
        'tabla1': pd.read_csv(base_path / 'tabla1_resultados_multiclass_poc55.csv'),
        'tabla2': pd.read_csv(base_path / 'tabla2_performance_per_class.csv'),
        'tabla3': pd.read_csv(base_path / 'tabla3_domain_generalization_proyectado.csv'),
        'tabla4': pd.read_csv(base_path / 'tabla4_dg_techniques_ablation.csv'),
        'tabla5': pd.read_csv(base_path / 'tabla5_dataset_progression.csv'),
        'tabla6': pd.read_csv(base_path / 'tabla6_poc_evolution.csv'),
    }

def grafico1_comparativa_arquitecturas(df):
    """
    Gráfico 1: Comparativa 3 arquitecturas en 3 niveles jerárquicos
    Barras agrupadas mostrando Binary/Coarse/Fine mIoU
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Preparar datos
    models = df['Modelo'].str.replace('-Tiny', '')
    x = np.arange(len(models))
    width = 0.25
    
    # Convertir porcentajes a float
    binary = df['mIoU_Binary'].str.rstrip('%').astype(float)
    coarse = df['mIoU_Coarse'].str.rstrip('%').astype(float)
    fine = df['mIoU_Fine'].str.rstrip('%').astype(float)
    
    # Crear barras
    bars1 = ax.bar(x - width, binary, width, label='Binary (2 clases)', 
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, coarse, width, label='Coarse (4 grupos)', 
                   color='#f39c12', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, fine, width, label='Fine (16 clases)', 
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # Añadir valores en las barras
    def add_values(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_values(bars1)
    add_values(bars2)
    add_values(bars3)
    
    # Styling
    ax.set_ylabel('mIoU (%)', fontsize=14, fontweight='bold')
    ax.set_title('Rendimiento Jerárquico: CNN vs ViT vs Hybrid\n(POC5.5 - 418 muestras, RTX 3050)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 80)
    
    # Añadir línea de target
    ax.axhline(y=22, color='red', linestyle='--', linewidth=2, alpha=0.5, 
               label='Target mIoU (22%)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grafico1_comparativa_arquitecturas.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico 1 generado: Comparativa arquitecturas")

def grafico2_per_class_heatmap(df):
    """
    Gráfico 2: Heatmap de IoU por clase y modelo
    Muestra qué clases son fáciles/difíciles para cada arquitectura
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Preparar matriz de datos
    classes = df['Clase'].values
    maxvit = df['MaxViT_IoU'].str.rstrip('%').astype(float).values
    swin = df['Swin_IoU'].str.rstrip('%').astype(float).values
    convnext = df['ConvNeXt_IoU'].str.rstrip('%').astype(float).values
    
    data_matrix = np.column_stack([maxvit, swin, convnext])
    
    # Crear heatmap
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Añadir valores en las celdas
    for i in range(len(classes)):
        for j, model in enumerate(['MaxViT', 'Swin', 'ConvNeXt']):
            value = data_matrix[i, j]
            color = 'white' if value < 50 else 'black'
            text = ax.text(j, i, f'{value:.1f}%',
                          ha='center', va='center', color=color, 
                          fontsize=10, fontweight='bold')
    
    # Configurar ejes
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(['MaxViT\n(Hybrid)', 'Swin\n(ViT)', 'ConvNeXt\n(CNN)'], 
                       fontsize=12, fontweight='bold')
    ax.set_yticklabels(classes, fontsize=10)
    
    # Título y colorbar
    ax.set_title('Detección por Clase de Daño: IoU (%) Heatmap\n' + 
                 'Verde = Buena detección, Rojo = Pobre detección',
                 fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('IoU (%)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grafico2_per_class_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico 2 generado: Heatmap per-class IoU")

def grafico3_domain_generalization(df):
    """
    Gráfico 3: Domain Generalization Gap
    Barras mostrando in-domain vs LOMO vs LOContent
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = df['Modelo'].str.replace('-Tiny', '')
    x = np.arange(len(models))
    width = 0.25
    
    # Convertir a float
    in_domain = df['In_Domain_mIoU'].str.rstrip('%').astype(float)
    lomo = df['LOMO_mIoU'].str.rstrip('%').astype(float)
    locontent = df['LOContent_mIoU'].str.rstrip('%').astype(float)
    
    # Crear barras
    bars1 = ax.bar(x - width, in_domain, width, label='In-Domain', 
                   color='#3498db', alpha=0.9, edgecolor='black')
    bars2 = ax.bar(x, lomo, width, label='LOMO (Leave-One-Material-Out)', 
                   color='#e67e22', alpha=0.9, edgecolor='black')
    bars3 = ax.bar(x + width, locontent, width, label='LOContent (Leave-One-Content-Out)', 
                   color='#95a5a6', alpha=0.9, edgecolor='black')
    
    # Añadir valores
    def add_values_dg(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    add_values_dg(bars1)
    add_values_dg(bars2)
    add_values_dg(bars3)
    
    # Styling
    ax.set_ylabel('mIoU (%)', fontsize=14, fontweight='bold')
    ax.set_title('Domain Generalization Gap (Proyección POC6)\n' +
                 'Caída de rendimiento en dominios no vistos',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 25)
    
    # Añadir anotación de gap
    for i, model in enumerate(models):
        gap = in_domain.iloc[i] - lomo.iloc[i]
        ax.annotate(f'Gap: -{gap:.1f}%', 
                   xy=(i, lomo.iloc[i] - 1), 
                   fontsize=9, ha='center', color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grafico3_domain_generalization.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico 3 generado: Domain Generalization Gap")

def grafico4_dg_techniques_ablation(df):
    """
    Gráfico 4: Ablation de técnicas DG
    Barras horizontales mostrando ganancia de cada técnica
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Preparar datos
    techniques = df['Tecnica_DG'].str.replace('_', ' ')
    baseline = df['MaxViT_Baseline'].str.rstrip('%').astype(float)
    improved = df['MaxViT_Mejorado'].str.rstrip('%').astype(float)
    gain = improved - baseline
    
    # Ordenar por ganancia
    sorted_idx = gain.argsort()
    techniques = techniques.iloc[sorted_idx]
    gain = gain.iloc[sorted_idx]
    improved_sorted = improved.iloc[sorted_idx]
    
    # Colores según ganancia
    colors = ['#2ecc71' if g > 0 else '#95a5a6' for g in gain]
    
    # Crear barras horizontales
    bars = ax.barh(range(len(techniques)), gain, color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Añadir valores
    for i, (bar, g, imp) in enumerate(zip(bars, gain, improved_sorted)):
        if g > 0:
            ax.text(g + 0.2, i, f'+{g:.1f}% → {imp:.1f}%', 
                   va='center', fontsize=10, fontweight='bold', color='darkgreen')
        else:
            ax.text(0.1, i, f'{imp:.1f}%', 
                   va='center', fontsize=10, color='gray')
    
    # Styling
    ax.set_yticks(range(len(techniques)))
    ax.set_yticklabels(techniques, fontsize=11)
    ax.set_xlabel('Ganancia mIoU (%)', fontsize=14, fontweight='bold')
    ax.set_title('Técnicas Domain Generalization: Ablation Study (Proyección)\n' +
                 'MaxViT-Tiny baseline 16.8% → mejorado con DG techniques',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.axvline(x=0, color='black', linewidth=2)
    
    # Resaltar mejor técnica
    best_idx = gain.argmax()
    ax.get_yticklabels()[best_idx].set_color('red')
    ax.get_yticklabels()[best_idx].set_fontweight('bold')
    ax.get_yticklabels()[best_idx].set_fontsize(12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grafico4_dg_techniques_ablation.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico 4 generado: DG Techniques Ablation")

def grafico5_dataset_evolution_timeline(df):
    """
    Gráfico 5: Timeline de evolución del dataset
    Línea mostrando crecimiento POC5 → POC5.5 → POC5.8 → POC6
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Preparar datos
    pocs = ['POC5\nBinary', 'POC5.5\nLaptop', 'POC5.8\nServer', 'POC6\nTarget']
    samples = [50, 418, 1464, 11000]
    classes = [2, 16, 16, 16]
    
    # Gráfico 1: Samples
    ax1.plot(pocs, samples, marker='o', linewidth=3, markersize=12, 
            color='#3498db', label='Samples')
    ax1.fill_between(range(len(pocs)), samples, alpha=0.3, color='#3498db')
    
    for i, (poc, sample) in enumerate(zip(pocs, samples)):
        ax1.text(i, sample + 500, f'{sample:,}', ha='center', 
                fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Número de Muestras', fontsize=14, fontweight='bold')
    ax1.set_title('Evolución del Dataset: Escalabilidad', 
                 fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 12000)
    
    # Gráfico 2: Classes
    colors_classes = ['#95a5a6', '#e74c3c', '#e74c3c', '#e74c3c']
    bars = ax2.bar(pocs, classes, color=colors_classes, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    
    for i, (bar, cls) in enumerate(zip(bars, classes)):
        ax2.text(i, cls + 0.5, f'{cls} clases', ha='center', 
                fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Número de Clases', fontsize=14, fontweight='bold')
    ax2.set_title('Complejidad de la Tarea', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 20)
    
    # Añadir anotación
    ax2.annotate('Binary → Multiclass', 
                xy=(0.5, 9), xytext=(1, 12),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=11, fontweight='bold', color='red')
    
    plt.suptitle('Progresión de la Investigación: POC5 → POC6', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grafico5_dataset_evolution.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico 5 generado: Dataset Evolution Timeline")

def grafico6_throughput_optimization(df):
    """
    Gráfico 6: Optimización de throughput POC5 → POC5.8
    Mostrar el impacto de las optimizaciones (laptop vs server)
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Datos extraídos de tabla6
    pocs = ['POC5\nBinary\n(Laptop)', 'POC5.5\nMulticlass\n(Laptop)', 
            'POC5.8\nPreloading\n(Server)']
    throughput = [8.5, 4.2, 97.0]
    vram = [2.1, 0.84, 0.41]
    
    # Crear barras
    x = np.arange(len(pocs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, throughput, width, label='Throughput (imgs/s)', 
                   color='#2ecc71', alpha=0.9, edgecolor='black')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, vram, width, label='VRAM Usage (GB)', 
                    color='#e74c3c', alpha=0.9, edgecolor='black')
    
    # Añadir valores
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.1f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Styling
    ax.set_ylabel('Throughput (imgs/s)', fontsize=14, fontweight='bold', color='#2ecc71')
    ax2.set_ylabel('VRAM Usage (GB)', fontsize=14, fontweight='bold', color='#e74c3c')
    ax.set_xlabel('POC Version', fontsize=14, fontweight='bold')
    ax.set_title('Optimización de Rendimiento: Laptop → Server + RAM Preloading\n' +
                 '23x speedup en throughput, 5x reducción en VRAM',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(pocs, fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#2ecc71')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Leyenda combinada
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
             fontsize=11, framealpha=0.9)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grafico6_throughput_optimization.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico 6 generado: Throughput Optimization")

def grafico7_radar_comparison(df):
    """
    Gráfico 7: Radar chart comparando 3 arquitecturas
    En múltiples dimensiones: mIoU, params, throughput, VRAM
    """
    from math import pi
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Categorías
    categories = ['mIoU Fine', 'mIoU Coarse', 'mIoU Binary', 
                 'Throughput', 'VRAM Efficiency']
    N = len(categories)
    
    # Preparar datos (normalizar a 0-100)
    models_data = []
    for idx, row in df.iterrows():
        model_name = row['Modelo'].replace('-Tiny', '')
        values = [
            float(row['mIoU_Fine'].rstrip('%')),
            float(row['mIoU_Coarse'].rstrip('%')),
            float(row['mIoU_Binary'].rstrip('%')),
            float(row['Throughput_imgs_s']) / 4.5 * 100,  # Normalizar a 100
            (1 - float(row['VRAM_Usage_GB']) / 1.0) * 100  # Invertir (menos es mejor)
        ]
        models_data.append({'name': model_name, 'values': values})
    
    # Ángulos para cada eje
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Colores
    colors_radar = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Plotear cada modelo
    for model_dict, color in zip(models_data, colors_radar):
        values = model_dict['values']
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model_dict['name'], 
               color=color, markersize=8)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    # Configurar ejes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Título y leyenda
    plt.title('Comparativa Multi-dimensional: CNN vs ViT vs Hybrid\n' +
             '(Mayor área = Mejor rendimiento)',
             fontsize=14, fontweight='bold', pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
              fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grafico7_radar_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico 7 generado: Radar Chart Comparison")

def grafico8_class_difficulty_scatter(df):
    """
    Gráfico 8: Scatter plot Frecuencia vs Performance
    Mostrar correlación entre frecuencia de clase y IoU
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Mapear frecuencia a valores numéricos
    freq_map = {'Muy_Baja': 1, 'Baja': 2, 'Media': 3, 'Alta': 4}
    df['Freq_Numeric'] = df['Frecuencia_Dataset'].map(freq_map)
    
    # Convertir IoU a float
    maxvit_iou = df['MaxViT_IoU'].str.rstrip('%').astype(float)
    
    # Colores según dificultad
    diff_colors = {
        'Facil': '#2ecc71',
        'Media': '#f39c12',
        'Dificil': '#e67e22',
        'Muy_Dificil': '#e74c3c'
    }
    colors = df['Dificultad'].map(diff_colors)
    
    # Crear scatter
    scatter = ax.scatter(df['Freq_Numeric'], maxvit_iou, 
                        c=colors, s=200, alpha=0.7, 
                        edgecolors='black', linewidth=2)
    
    # Añadir labels para cada punto
    for idx, row in df.iterrows():
        ax.annotate(row['Clase'], 
                   (row['Freq_Numeric'], float(row['MaxViT_IoU'].rstrip('%'))),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)
    
    # Línea de tendencia
    z = np.polyfit(df['Freq_Numeric'], maxvit_iou, 1)
    p = np.poly1d(z)
    ax.plot(df['Freq_Numeric'], p(df['Freq_Numeric']), 
           "--", color='gray', linewidth=2, alpha=0.5, 
           label=f'Tendencia: y = {z[0]:.1f}x + {z[1]:.1f}')
    
    # Styling
    ax.set_xlabel('Frecuencia en Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('IoU MaxViT (%)', fontsize=14, fontweight='bold')
    ax.set_title('Relación Frecuencia-Performance: Class Imbalance Impact\n' +
                 'Clases raras (Lightleak, Burn marks) tienen IoU <1%',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['Muy Baja', 'Baja', 'Media', 'Alta'], fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')
    
    # Leyenda de colores
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=diff, edgecolor='black') 
                      for diff, color in diff_colors.items()]
    ax.legend(handles=legend_elements, title='Dificultad', 
             loc='lower right', fontsize=10, title_fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grafico8_class_difficulty_scatter.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico 8 generado: Class Difficulty Scatter")

def main():
    """Generar todos los gráficos"""
    print("🎨 Generando gráficos para presentación...")
    print("="*60)
    
    # Cargar datos
    data = load_data()
    
    # Generar gráficos
    print("\n📊 Generando visualizaciones...")
    grafico1_comparativa_arquitecturas(data['tabla1'])
    grafico2_per_class_heatmap(data['tabla2'])
    grafico3_domain_generalization(data['tabla3'])
    grafico4_dg_techniques_ablation(data['tabla4'])
    grafico5_dataset_evolution_timeline(data['tabla5'])
    grafico6_throughput_optimization(data['tabla6'])
    grafico7_radar_comparison(data['tabla1'])
    grafico8_class_difficulty_scatter(data['tabla2'])
    
    print("\n" + "="*60)
    print(f"✅ Todos los gráficos generados en: {OUTPUT_DIR}")
    print("\nGráficos disponibles:")
    for i in range(1, 9):
        print(f"  - grafico{i}_*.png")
    print("\n💡 Recomendaciones para presentación:")
    print("  • Gráfico 1: Slide principal de resultados")
    print("  • Gráfico 2: Análisis detallado per-class")
    print("  • Gráfico 3: Domain Generalization (trabajo futuro)")
    print("  • Gráfico 5: Timeline de progreso POC5→POC6")
    print("  • Gráfico 7: Comparativa visual impactante (radar)")

if __name__ == '__main__':
    main()
