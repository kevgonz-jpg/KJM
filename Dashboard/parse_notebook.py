import json
import os
def parse_notebook(path):
    print(f"\n=== Analizando: {os.path.basename(path)} ===")
    if not os.path.exists(path):
        print("El archivo no existe.")
        return
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cells = data.get('cells', [])
    plotly_cells = []
    for idx, cell in enumerate(cells):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            if 'plotly' in source.lower() or 'go.Figure' in source:
                plotly_cells.append((idx, source))
    
    with open("notebooks_extracted.txt", "a", encoding="utf-8") as out:
        out.write(f"\n=== Analizando: {os.path.basename(path)} ===\n")
        out.write(f"Total de celdas con Plotly: {len(plotly_cells)}\n")
        if len(plotly_cells) > 0:
            out.write("\n--- PRIMERA CELDA CON PLOTLY ---\n")
            out.write(f"Index: {plotly_cells[0][0]}\n")
            out.write(plotly_cells[0][1])
            out.write("\n\n--- ÚLTIMA CELDA CON PLOTLY ---\n")
            out.write(f"Index: {plotly_cells[-1][0]}\n")
            out.write(plotly_cells[-1][1])
            out.write("\n\n" + "="*50 + "\n")
    return plotly_cells
if os.path.exists("notebooks_extracted.txt"):
    os.remove("notebooks_extracted.txt")
# Analizar notebook de sistema_procesamiento
parse_notebook(r"C:\Users\ADMON\Documents\Universidad\Proyecto KMJ\codigo\sistema_procesamiento\notebooks\experimentacion_curva_fase1_famas1.ipynb")
# Analizar notebook de TES_colombia
parse_notebook(r"C:\Users\ADMON\Documents\Universidad\Proyecto KMJ\codigo\TES_colombia\Limpieza_TESCOL_y_Dinamica.ipynb")