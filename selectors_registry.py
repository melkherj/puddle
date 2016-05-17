from selectors.random_selector import RandomSelector
from selectors.entropy_selector import EntropySelector
from selectors.adaptive_selector import AdaptiveSelector
from selectors.top_selector import TopSelector
from selectors.empty_selector import EmptySelector

#@author melkherj

selectors_registry = {
    'random':RandomSelector,
    'entropy':EntropySelector,
    'adaptive':AdaptiveSelector,
    'top':TopSelector,
    'empty':EmptySelector,
}
