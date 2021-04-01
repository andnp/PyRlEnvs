from glob import glob
import os

import PyRlEnvs.Category as Cat

# categories are stored along with the modules
# we don't really want to load all of the modules whenever the library is loaded,
# so we'll make that an opt-in feature
_categories_built = False
def _buildCategories():
    global _categories_built

    if _categories_built:
        return

    _categories_built = True

    domain_paths = glob(os.path.join(os.path.dirname(__file__), 'domains', '*.py'))
    domain_paths = (path for path in domain_paths if '__init__.py' not in path)
    domain_paths = (path for path in domain_paths if os.path.isfile(path))

    for path in domain_paths:
        __import__(path)


def getCategory(category: str):
    _buildCategories()

    return Cat.getCategory(category)
