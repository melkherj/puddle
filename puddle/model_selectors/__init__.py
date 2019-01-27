from .simple import UncertaintySelector, RandomSelector, FisherSelector

selectors = {
    'random':RandomSelector(),
    'uncertainty':UncertaintySelector(),
    'FisherSelector': FisherSelector()
    }