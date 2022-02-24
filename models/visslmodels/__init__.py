
from pathlib import Path
from classy_vision.generic.registry_utils import import_all_modules

FILE_ROOT = Path(__file__).parent
print(FILE_ROOT)
import_all_modules(FILE_ROOT, 'models.visslmodels')
