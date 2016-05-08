from selectors.random_selector import RandomSelector
from selectors.entropy_selector import EntropySelector

#@author melkherj

selectors_registry = {
    'random':RandomSelector,
    'entropy':EntropySelector,
}
