from selectors.random_selector import RandomSelector
from selectors.entropy_selector import EntropySelector
from selectors.max_selector import MaxSelector

#@author melkherj

selectors_registry = {
    'random':RandomSelector,
    'entropy':EntropySelector,
    'max':MaxSelector,
}
