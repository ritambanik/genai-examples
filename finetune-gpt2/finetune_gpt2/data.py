"""
Data loading and preprocessing utilities for GPT-2 fine-tuning.
"""

import re
from typing import Optional, Union
from pathlib import Path
import PyPDF2
import requests
from datasets import load_dataset, DatasetDict

def download_data(save_path: Union[str, Path], url: str = "https://gita-society.com/wp-content/uploads/PDF/bluebook15.pdf") -> None:
    """
    Download data from a specified URL and save it to a local path.

    Args:
        save_path: Local path to save the downloaded file
        url: URL to download the data from
    Returns:
        None
    """
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Data downloaded successfully and saved to {save_path}")        
    except requests.RequestException as e:
        print(f"Error downloading data: {e}")

def load_pdf_data(file_path: Union[str, Path]) -> str:
    """
    Load text data from a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        The content of the file as a string
    """
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page_num in range(len(reader.pages)):
          if page_num > 3:
              page = reader.pages[page_num]
              text += page.extract_text()
    return text

def perform_eda(text: str) -> str:
    """
    Perform the following EDA steps.

    Args:
        text (str): Input text to be sanitized
        
    Returns:
        str: Sanitized text after EDA
    """
    
    text = text.replace("\n", '').replace("\r", '')
    text = " ".join(text.split())
    text = re.sub(r' \d+ International Gita Society', '', text)
    text = re.sub(r' Bhagavad -Gita \d+', '', text)
    
    new_text = split_after_n_words(text, 100)  # Split text after every 100 words for better readability
    
    return new_text

def split_after_n_words(text: str, n: int) -> str:
    """
    Split the text after every n words.

    Args:
        text (str): Input text to be split
        n (int): Number of words after which to split the text

    Returns:
        str: Text with newlines inserted after every n words
    """
    words = text.split()
    return "\n".join(" ".join(words[i:i+n]) for i in range(0, len(words), n))

def split_into_train_validation(text: str, file_path: Union[str, Path], train_split: float = 0.8) -> None:
    """
    Split the text into training and validation sets.

    Args:
        text (str): Input text to be split
        file_path (Union[str, Path], optional): Path to save the train and validation files
        train_split (float): Proportion of the data to be used for training
        

    Returns:
        None
    """
    split_index = int(train_split * len(text))
    train_text = text[:split_index]
    validation_text = text[split_index:]
    
    train_file_path = Path(file_path) / "train.txt"
    validation_file_path = Path(file_path) / "validation.txt"
    
    with open(train_file_path, "w") as f:
        f.write(train_text)
        
    with open(validation_file_path, "w") as f:
        f.write(validation_text)


def prepare_dataset(
    data_path: Union[str, Path]
) -> DatasetDict:
    """
    Prepare and tokenize a dataset for GPT-2 training.

    Args:
        data_path: Path to the training data

    Returns:
        Dataset ready for training
    """
    
    doc_path = Path(data_path) / "document.pdf"
    download_data(doc_path)
    text = load_pdf_data(doc_path)
    sanitized_text = perform_eda(text)
    split_into_train_validation(sanitized_text, data_path)
    
    train_file_path = Path(data_path) / "train.txt"
    validation_file_path = Path(data_path) / "validation.txt"
    
    dataset = load_dataset("text", data_files={"train": str(train_file_path), "validation": str(validation_file_path)})
       
    return dataset


__all__ = ["load_pdf_data", "prepare_dataset"]
