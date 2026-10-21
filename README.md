# lector-mdl

Analizador de modelos de Vensim (`.mdl`) para leerlos **por capas**. Es la herramienta que usa la skill `lector-modelos-vensim`: extrae la estructura del modelo sin interpretarla y lo simula con PySD, para que la explicación se apoye en hechos del archivo y no en suposiciones.

## Qué produce

`python mdl_reader.py modelo.mdl --out salida` deja en `salida/`:

| Archivo | Contenido |
| --- | --- |
| `estructura.json` | Inventario de variables (tipo, unidades, comentario, ecuación, vista), stocks con sus flujos, frontera endógena/exógena, enlaces con polaridad estimada numéricamente, bucles con su tipo (R o B), bucles esenciales, familias de bucles, retardos ocultos, tablas, datos externos, subíndices y avisos |
| `resumen.md` | Lo mismo en tablas legibles |
| `sim_<escenario>.csv` | Trayectorias simuladas de cada escenario (si PySD está instalado) |

## Instalación y uso

```bash
pip install -r requirements.txt
python mdl_reader.py modelo.mdl --out salida \
    --scenario "Base:" \
    --scenario "Alta demanda:AUMENTO DE DEMANDA=0.9" \
    --scenario "Refactor:POLITICA DE REFACTOR=1,RECORTE DE REFACTOR POR PRESION=0"
```

- Los nombres de variables van tal como están en el `.mdl`, con espacios.
- `--data archivo.DAT` o `--data archivo.csv`: datos externos (variables sin ecuación o `GET XLS`/`GET DATA`).
- `--no-sim`: solo estructura. `--max-loops N`: limita la búsqueda de bucles en modelos grandes.
- El nombre del escenario se normaliza en el archivo: `"Alta demanda"` queda como `sim_Alta_demanda.csv`.

## Comprobar que funciona

```bash
pytest -q
```

Las pruebas leen los modelos de `ejemplos/`, comprueban los stocks y la coherencia de los bucles (número par de enlaces negativos = R), y verifican que la simulación coincide con una corrida directa de PySD.

## Límites

- PySD no implementa todas las funciones de Vensim; las que faltan se traducen a `not_implemented_function()` y el analizador lo avisa.
- La polaridad de los enlaces se estima numéricamente alrededor de las trayectorias simuladas: en enlaces no lineales puede cambiar según el estado (`±`).
- Con subíndices, la lectura es parcial.
