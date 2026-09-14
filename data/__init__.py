"""CFlow-LM Datasets package."""

from .wikitext import WikiTextDataset, build_wikitext_loader
from .pg19 import PG19Dataset, build_pg19_loader
from .codeparrot import CodeParrotDataset, build_codeparrot_loader

DATASET_REGISTRY = {
    "wikitext103": build_wikitext_loader,
    "pg19":        build_pg19_loader,
    "codeparrot":  build_codeparrot_loader,
}


def build_dataloader(dataset_name: str, **kwargs):
    """Factory function for dataset dataloaders."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[dataset_name](**kwargs)


__all__ = [
    "WikiTextDataset",
    "PG19Dataset",
    "CodeParrotDataset",
    "build_wikitext_loader",
    "build_pg19_loader",
    "build_codeparrot_loader",
    "build_dataloader",
]
