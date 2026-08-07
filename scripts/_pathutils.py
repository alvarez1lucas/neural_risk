# scripts/_pathutils.py
"""
Utilidad compartida para resolver rutas de forma robusta al directorio
desde el que se invoque el script, Y para que los imports de
'neural_risk.*' funcionen sin importar desde dónde se corra el script.

FIX (#3 de la auditoria de VS Code): los scripts usaban rutas relativas
fijas ("config/config.yaml", "./data/trained_models.pkl") que solo
funcionan si se corre "python scripts/run_engine.py" desde la RAIZ del
proyecto. Si se corre desde otro directorio (ej. parado dentro de
scripts/), fallaba con FileNotFoundError.

FIX #2 (encontrado por el usuario vía Copilot -- ModuleNotFoundError:
No module named 'neural_risk'): cuando se ejecuta "python scripts/algo.py",
Python agrega SOLO el directorio del script (scripts/) a sys.path[0] --
NUNCA la raíz del proyecto. Como el paquete 'neural_risk/' vive un nivel
arriba (junto a scripts/, no adentro), "from neural_risk.engine import..."
fallaba SIEMPRE al correr los scripts directamente, sin importar el CWD.

Por eso, ahora, con solo IMPORTAR este módulo (import _pathutils o
from _pathutils import resolve_path), como efecto colateral se inserta
la raíz real del proyecto en sys.path -- así los imports de neural_risk.*
que vengan DESPUÉS en el mismo archivo encuentran el paquete.

IMPORTANTE: el import de _pathutils tiene que ser el PRIMERO de todos
en cada script (antes de cualquier "from neural_risk... import..."),
para que el parche de sys.path ya esté aplicado cuando Python llegue a
esas líneas. Ver el orden correcto en run_engine.py/backtest.py/etc.
"""
import os
import sys


def project_root() -> str:
    """Raiz del proyecto = un nivel arriba de la carpeta scripts/."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Efecto colateral al importar este módulo: garantiza que la raíz del
# proyecto esté en sys.path, para que 'import neural_risk...' funcione
# sin importar desde qué directorio se haya invocado el script.
_root = project_root()
if _root not in sys.path:
    sys.path.insert(0, _root)


def resolve_path(relative_path: str) -> str:
    """
    Si 'relative_path' existe relativo al CWD actual (convencion normal:
    correr desde la raiz del proyecto), se devuelve tal cual -- no rompe
    nada de lo que ya funcionaba. Si no existe ahi, se resuelve relativo
    a la raiz REAL del proyecto (funciona aunque se invoque el script
    desde otro directorio).
    """
    if os.path.exists(relative_path):
        return relative_path
    return os.path.join(project_root(), relative_path)