from typing import Dict, Iterator, List, Optional, NamedTuple, Type
from PyRlEnvs.BaseEnvironment import BaseEnvironment

class EnvironmentMetadata(NamedTuple):
    name: str
    Env: Type[BaseEnvironment]
    description: Optional[str] = None


# TODO: generate documentation based off of this information
# TODO: generate experiment descriptions based off of this (not in this repo, but this is best place to store this todo right now)
class Category:
    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description

        self.environments: List[EnvironmentMetadata] = []

    def addEnvironment(self, name: str, Env: Type[BaseEnvironment], description: Optional[str] = None):
        metadata = EnvironmentMetadata(name, Env, description)
        self.environments.append(metadata)

        return metadata

    def __len__(self) -> int:
        return len(self.environments)

    def __iter__(self) -> Iterator[EnvironmentMetadata]:
        return self.environments.__iter__()

    def __getitem__(self, idx: int) -> EnvironmentMetadata:
        return self.environments.__getitem__(idx)

_categories: Dict[str, Category] = {}

def addToCategory(category: str, Env: Type[BaseEnvironment], description: Optional[str] = None):
    cat = _categories.get(category)

    if cat is None:
        cat = Category(category)

    cat.addEnvironment(Env.__name__, Env, description)

    return cat

def createCategory(category: str, description: Optional[str] = None):
    if category in _categories:
        print('Overwritting this category ', category)

    _categories[category] = Category(category, description)

    return _categories[category]

def getCategory(category: str):
    cat = _categories.get(category)

    if cat is None:
        raise Exception("Nope, this category doesn't exist", category)

    return cat
