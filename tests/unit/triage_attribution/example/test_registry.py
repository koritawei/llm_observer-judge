"""The classifiers_registry plus triage_attribution.registry eager-import registration path."""
from benchmark.attribution.service import TriageClassifier
from benchmark.attribution.classifiers_registry import triage_classifiers
from business.agents.example.classifier import ExampleTriageClassifier


def test_registry_returns_dict_keyed_by_logic_name() -> None:
    classifiers = triage_classifiers()
    assert isinstance(classifiers["example"], ExampleTriageClassifier)


def test_registry_values_are_triage_classifier_subclasses() -> None:
    for classifier in triage_classifiers().values():
        assert isinstance(classifier, TriageClassifier)


def test_old_function_removed_from_example_module() -> None:
    """triage_classifiers is no longer defined in example/classifier.py."""
    import business.agents.example.classifier as example_mod

    assert not hasattr(example_mod, "triage_classifiers")
