"""Tests for data loading and preprocessing."""

import pytest
from finetune_gpt2.data import load_pdf_data, download_data


def test_load_pdf_data_placeholder():
    """Placeholder test for load_pdf_data function."""
    download_data("document.pdf")
    assert load_pdf_data("document.pdf") is not None 


def test_prepare_dataset_placeholder():
    """Placeholder test for prepare_dataset function."""
    # TODO: Implement actual tests
    assert True
