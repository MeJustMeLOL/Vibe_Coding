import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from torch.utils.data import Dataset, DataLoader
import json
# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
nltk.download('punkt')


# ------------------------
# Web Scraper Class
# ------------------------
class WebScraper:
    """Handles fetching HTML from a webpage using Selenium."""
    def __init__(self, proxy=None, max_retries=3, retry_delay=2):
        self.proxy = proxy
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch(self, url):
        """Fetch HTML content from a webpage."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        if self.proxy:
            chrome_options.add_argument(f"--proxy-server={self.proxy}")

        for attempt in range(self.max_retries):
            try:
                with webdriver.Chrome(options=chrome_options) as driver:
                    driver.get(url)
                    logging.info(f"Successfully fetched the page: {url}")
                    return driver.page_source
            except WebDriverException as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(self.retry_delay)
        raise Exception(f"Failed to fetch {url} after {self.max_retries} attempts.")


# ------------------------
# Text Classification Models
# ------------------------
class TextFNN(nn.Module):
    """A simple feed-forward neural network for text classification."""
    def __init__(self, input_size, hidden_size, num_classes):
        super(TextFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TextCNN(nn.Module):
    """A simple CNN for text classification."""
    def __init__(self, vocab_size, embedding_dim=100, num_filters=100, filter_sizes=[3, 4, 5]):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)          # [batch_size, seq_len, embedding_dim]
        x = x.unsqueeze(1)             # [batch_size, 1, seq_len, embedding_dim]
        conv_results = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(result, dim=2)[0] for result in conv_results]
        features = torch.cat(pooled, dim=1)
        features = self.dropout(features)
        out = self.fc(features)        # [batch_size, 1]
        return out


# ------------------------
# Dataset Classes
# ------------------------
class ReviewDataset(Dataset):
    """Custom Dataset for review samples."""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        # For FNN (CrossEntropyLoss) we use integer labels.
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label


# ------------------------
# Utility Functions for Scraping and Preprocessing
# ------------------------
def scrape_text_from_url(url, proxy=None):
    """Scrapes text from a webpage."""
    scraper = WebScraper(proxy=proxy)
    html = scraper.fetch(url)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(strip=True)


def preprocess_text(text):
    """Tokenizes text into sentences using NLTK."""
    return nltk.sent_tokenize(text)


def split_into_samples(text, sample_size=1000):
    """Splits text into chunks of 'sample_size' characters."""
    return [text[i:i+sample_size] for i in range(0, len(text), sample_size)]


def build_dataset(pos_urls: list, neg_urls: list, proxy: str = None,
                  max_len: int = 50, vocab_size: int = 1000, model: str = "FNN"):
    """
    Scrapes text from lists of positive and negative URLs,
    builds fixed-length sequences, and returns the dataset along with a vocabulary mapping.
    """
    texts = []
    labels = []

    for url in pos_urls:
        try:
            text = scrape_text_from_url(url, proxy)
            samples = split_into_samples(text)
            texts.extend(samples)
            labels.extend([1] * len(samples))
            logging.info(f"Scraped {len(samples)} samples from positive URL: {url}")
        except Exception as e:
            logging.error(f"Error scraping positive URL {url}: {e}")

    for url in neg_urls:
        try:
            text = scrape_text_from_url(url, proxy)
            samples = split_into_samples(text)
            texts.extend(samples)
            labels.extend([0] * len(samples))
            logging.info(f"Scraped {len(samples)} samples from negative URL: {url}")
        except Exception as e:
            logging.error(f"Error scraping negative URL {url}: {e}")

    # Build vocabulary from texts
    tokenized_texts = []
    all_tokens = []
    from nltk_utils import stem
    for text in texts:
        tokens = [word.lower() for word in nltk.word_tokenize(text)]
        stemmed_tokens = [stem(word) for word in tokens]  # Stem tokens here!
        tokenized_texts.append(stemmed_tokens)
        all_tokens.extend(stemmed_tokens)
    freq_dist = nltk.FreqDist(all_tokens)
    vocab_common = freq_dist.most_common(vocab_size - 1)  # Reserve index 0 for padding
    word_to_idx = {'<PAD>': 0}
    for word, _ in vocab_common:
        word_to_idx[word] = len(word_to_idx)

    sequences = []
    for tokens in tokenized_texts:
        seq = [word_to_idx.get(word, 0) for word in tokens][:max_len]
        if len(seq) < max_len:
            seq += [0] * (max_len - len(seq))
        sequences.append(seq)

    # Reshape data based on model type.
    if model.upper() == "FNN":
        # For FNN, use BoW representation (vocab_size features)
        X = np.zeros((len(texts), vocab_size), dtype=np.int32)
        for i, tokens in enumerate(tokenized_texts):
            for word in tokens:
                idx = word_to_idx.get(word, 0)
                if idx < vocab_size:
                    X[i, idx] += 1  # Count word occurrences
    elif model.upper() == "CNN":
        # For CNN, add an extra channel dimension.
        X = np.array(sequences)
        X = X[..., np.newaxis]
    else:
        logging.warning(f"Model type '{model}' not recognized. Returning unmodified sequence data.")
    y = np.array(labels)
    return X, y, word_to_idx
def flatten_html_json(node, default_label=None):
    """
    Recursively traverse a nested HTML JSON tree and extract text and labels.
    Assumes that nodes may have a "text" key (or you can derive text from them),
    and optionally a "label" key. If no "label" exists, it uses default_label or
    the node's tag name as a fallback.
    """
    texts = []
    labels = []

    # If the node has a "text" field, extract it.
    # (You might modify this if your structure stores text differently.)
    text = node.get("text", None)
    if text:
        texts.append(text)
        # Use the provided label if available, otherwise fall back on default_label or the tag.
        label = node.get("label", default_label if default_label is not None else node.get("tag"))
        labels.append(label)

    # Process children recursively.
    for child in node.get("children", []):
        child_texts, child_labels = flatten_html_json(child, default_label=default_label)
        texts.extend(child_texts)
        labels.extend(child_labels)

    return texts, labels
def build_dataset_from_json(json_file, max_len=50, vocab_size=1000, model="FNN"):
    import json
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk_utils import stem  # assuming you have a stemming utility
    import numpy as np
    import logging

    # Load JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = []
    labels = []
    # Case 1: The JSON is structured as {"data": [ ... ]}
    if isinstance(data, dict) and "data" in data:
        data_list = data["data"]
    # Case 2: The JSON is a list of entries with "text" and "label" keys
    elif isinstance(data, list):
        data_list = data
    # Case 3: The JSON is a nested HTML tree exported by your parser
    elif isinstance(data, dict) and "tag" in data and "children" in data:
        # Flatten the nested tree. Adjust default_label as needed.
        texts, labels = flatten_html_json(data, default_label="unknown")
        data_list = None  # We've already extracted texts and labels.
    else:
        raise ValueError("JSON file format is invalid. It must be a list, contain a 'data' key with a list, or be a nested tree.")

    # If we have a list structure, extract text and label from each entry.
    if data_list is not None:
        for entry in data_list:
            # Use .get() to safely extract keys; log a warning if missing.
            text = entry.get("text")
            label = entry.get("label")
            if text is None or label is None:
                logging.warning(f"Skipping entry due to missing fields: {entry}")
                continue
            texts.append(text)
            labels.append(label)

    if len(texts) == 0:
        raise ValueError("No valid entries found in JSON file.")

    # Process text: tokenization and stemming
    tokenized_texts = []
    all_tokens = []
    for text in texts:
        tokens = [word.lower() for word in word_tokenize(text)]
        stemmed_tokens = [stem(word) for word in tokens]  # Apply stemming if needed
        tokenized_texts.append(stemmed_tokens)
        all_tokens.extend(stemmed_tokens)

    # Build vocabulary based on frequency distribution
    freq_dist = nltk.FreqDist(all_tokens)
    vocab_common = freq_dist.most_common(vocab_size - 1)  # Reserve index 0 for padding
    word_to_idx = {'<PAD>': 0}
    for word, _ in vocab_common:
        word_to_idx[word] = len(word_to_idx)

    # Create sequences: pad/truncate to max_len
    sequences = []
    for tokens in tokenized_texts:
        seq = [word_to_idx.get(word, 0) for word in tokens][:max_len]
        if len(seq) < max_len:
            seq += [0] * (max_len - len(seq))
        sequences.append(seq)

    # Prepare dataset based on model type
    if model.upper() == "FNN":
        # For FNN, use a bag-of-words (BoW) representation with vocab_size features.
        X = np.zeros((len(texts), vocab_size), dtype=np.int32)
        for i, tokens in enumerate(tokenized_texts):
            for word in tokens:
                idx = word_to_idx.get(word, 0)
                if idx < vocab_size:
                    X[i, idx] += 1  # Count occurrences of each word
    elif model.upper() == "CNN":
        # For CNN, keep the sequence format (and add an extra channel dimension if needed)
        X = np.array(sequences)
        X = X[..., np.newaxis]  # Add channel dimension for CNN
    else:
        raise ValueError("Model type not recognized. Choose either 'FNN' or 'CNN'.")

    # Convert labels to a NumPy array
    y = np.array(labels)
    return X, y, word_to_idx

# ------------------------
# HTML Parser and Explorer
# ------------------------
class HTMLParser:
    """Parses HTML into a navigable tree structure."""
    def __init__(self, html):
        self.soup = BeautifulSoup(html, "html.parser")
        self.root = self._build_tree(self.soup)

    def _build_tree(self, element, parent=None):
        if not hasattr(element, "name") or element.name is None:
            return None

        node = {
            "tag": element.name,
            "classes": element.get("class", []),
            "children": [],
            "parent": parent,
        }

        for child in element.children:
            child_node = self._build_tree(child, node)
            if child_node:
                node["children"].append(child_node)
        return node

    def get_tree(self):
        return self.root

    def export_json(self, filename):
        # Remove the circular parent references before exporting
        def remove_parent(node):
            # Create a new node excluding the "parent" key
            new_node = {k: v for k, v in node.items() if k != "parent"}
            # Process children recursively if available
            if "children" in new_node:
                new_node["children"] = [remove_parent(child) for child in new_node["children"]]
            return new_node

        tree_without_parents = remove_parent(self.root)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(tree_without_parents, f, indent=2)

    def extract_text_by_blocks(self):
        blocks = {}
        for block in self.soup.find_all(True):
            if block.name in ['div', 'p', 'span']:
                block_text = block.get_text(strip=True)
                if block_text:
                    blocks.setdefault(block.name, []).append(block_text)
        return blocks
def format_node_identifier(node):
    """Format a node's identifier (tag and classes)."""
    classes = " ".join(node.get("classes", []))
    return f"{node['tag']} ({classes})" if classes else node["tag"]


def tree_to_string(node, prefix="", is_last=True):
    """Convert an HTML tree into a visual tree-like string."""
    classes_str = f" ({' '.join(node['classes'])})" if node["classes"] else ""
    tree_str = f"{prefix}{'└── ' if is_last else '├── '}{node['tag']}{classes_str}"
    children = node.get("children", [])
    for i, child in enumerate(children):
        new_prefix = prefix + ("    " if is_last else "│   ")
        tree_str += "\n" + tree_to_string(child, new_prefix, i == len(children) - 1)
    return tree_str


def get_breadcrumbs(node):
    """Generate breadcrumb-style path for the current location in the tree."""
    path = []
    while node:
        path.append(format_node_identifier(node))
        node = node.get("parent")
    return " > ".join(reversed(path))


class HTMLExplorer:
    """Interactive CLI explorer for navigating an HTML tree."""
    def __init__(self, root):
        self.current_node = root

    def list_children(self):
        children = self.current_node.get("children", [])
        if children:
            print("\n📂 Children:")
            for idx, child in enumerate(children):
                print(f"  [{idx}] {format_node_identifier(child)}")
        else:
            print("\n⚠️ No children.")

    def change_directory(self, arg):
        children = self.current_node.get("children", [])
        if arg == "..":
            if self.current_node.get("parent"):
                self.current_node = self.current_node["parent"]
            else:
                print("⛔ Already at the root.")
        else:
            try:
                idx = int(arg)
                if 0 <= idx < len(children):
                    self.current_node = children[idx]
                else:
                    print("⛔ Index out of range.")
            except ValueError:
                matching_children = [child for child in children if arg in " ".join(child.get("classes", []))]
                if matching_children:
                    self.current_node = matching_children[0]
                else:
                    print("⛔ No matching class found.")

    def expand(self):
        print("\n📜 Subtree from current node:\n" + tree_to_string(self.current_node))

    def search(self, query):
        query = query.lower()
        results = []

        def search_tree(node):
            identifier = format_node_identifier(node).lower()
            if query in identifier:
                results.append(node)
            for child in node.get("children", []):
                search_tree(child)

        search_tree(self.current_node)
        if results:
            print("\n🔍 Search Results:")
            for i, node in enumerate(results):
                print(f"  [{i}] {format_node_identifier(node)}")
        else:
            print("❌ No matches found.")

    def start(self):
        while True:
            print(f"\n📍 Path: {get_breadcrumbs(self.current_node)}")
            self.list_children()
            command = input("\n🔹 Command (ls, cd <index/class>, cd .., expand, search <text>, exit): ").strip()
            if command.lower() == "ls":
                continue
            elif command.lower().startswith("cd "):
                self.change_directory(command[3:].strip())
            elif command.lower() == "expand":
                self.expand()
            elif command.lower().startswith("search "):
                self.search(command[7:].strip())
            elif command.lower() == "exit":
                print("👋 Exiting explorer.")
                break
            else:
                print("⛔ Unknown command.")


# ------------------------
# Training Functions
# ------------------------
def trainFNN(model, dataset, num_epochs, device, batch_size=16, lr=0.001):
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for words, labels in train_loader:
            # Cast inputs to float for FNN
            words = words.to(device).float()
            labels = labels.to(device)
            outputs = model(words)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    logging.info(f"Final loss: {loss.item():.4f}")


def trainCNN(model, dataset, num_epochs, device, batch_size=16, lr=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.float().to(device).unsqueeze(1)  # Adjust for BCEWithLogitsLoss
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")


def save_model_checkpoint(model, word_to_idx, additional_info, filename="model_checkpoint.pth"):
    """Saves the model state along with the vocabulary mapping and any additional info."""
    checkpoint = {
        "model_state": model.state_dict(),
        "word_to_idx": word_to_idx,
    }
    checkpoint.update(additional_info)
    torch.save(checkpoint, filename)
    logging.info(f"Model checkpoint saved to {filename}")


# ------------------------
# Main Execution
# ------------------------
def set_seed(seed):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Seed set to {seed}")

def main(args):
    set_seed(args.seed)

    # Check input: Either json_file or url must be provided based on mode.
    if args.mode != "scrape" and not (args.json_file or args.url):
        logging.error("For training modes, please provide either a JSON file or a URL.")
        return

    # JSON file based data loading takes precedence
    if args.json_file:
        logging.info("Loading dataset from JSON file.")
        try:
            X, y, word_to_idx = build_dataset_from_json(
                args.json_file,
                max_len=args.max_len,
                vocab_size=args.vocab_size,
                model=args.mode.upper()
            )
        except Exception as e:
            logging.error(f"Failed to build dataset from JSON: {e}")
            return

        dataset = ReviewDataset(X, y)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.mode.lower() == "fnn":
            input_size = X.shape[1]
            hidden_size = 128
            output_size = 2
            model_obj = TextFNN(input_size, hidden_size, output_size)
            trainFNN(model_obj, dataset, args.epochs, device, args.batch_size, args.lr)
            additional_info = {
                "input_size": input_size, "hidden_size": hidden_size,
                "output_size": output_size, "max_len": args.max_len,
                "word_to_idx": word_to_idx, "tags": ["positive", "negative"],
                "model_type": "FNN"
            }
            save_model_checkpoint(model_obj, word_to_idx, additional_info, filename="fnn_checkpoint.pth")
        elif args.mode.lower() == "cnn":
            model_obj = TextCNN(vocab_size=args.vocab_size)
            trainCNN(model_obj, dataset, args.epochs, device, args.batch_size, args.lr)
            additional_info = {
                "max_len": args.max_len, "vocab_size": args.vocab_size,
                "model_type": "CNN"
            }
            save_model_checkpoint(model_obj, word_to_idx, additional_info, filename="cnn_checkpoint.pth")
        else:
            logging.error("Invalid model type. Choose 'fnn' or 'cnn'.")

    elif args.mode == "scrape":
        if not args.url:
            logging.error("URL is required for scraping.")
            return
        try:
            text = scrape_text_from_url(args.url, args.proxy)
            logging.info("Extracted raw text from the webpage.")
            processed_text = preprocess_text(text)
            samples = split_into_samples(text)
            logging.info(f"Generated {len(samples)} text samples.")

            scraper = WebScraper(proxy=args.proxy)
            html = scraper.fetch(args.url)
            parser_obj = HTMLParser(html)
            class_tree = parser_obj.get_tree()
            tree_str = tree_to_string(class_tree)
            mapped_text = parser_obj.extract_text_by_blocks()
            parser_obj.export_json("output.json")

            output_filename = "scraped_output.txt"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write("Processed Text:\n")
                for sentence in processed_text:
                    f.write(sentence + "\n")
                f.write("\n---\nExtracted Text Blocks:\n")
                for block, texts in mapped_text.items():
                    f.write(f"{block}:\n")
                    for txt in texts:
                        f.write(f"  - {txt}\n")
                    f.write("-" * 50 + "\n")
                f.write("\nClass Tree:\n" + tree_str)
            logging.info(f"Results written to {output_filename}")

            browser_choice = input("Do you want to use the browser to explore the HTML? (yes/no): ").strip().lower()
            if browser_choice == "yes":
                explorer = HTMLExplorer(class_tree)
                explorer.start()
            else:
                print("Exiting without browser exploration.")
        except Exception as e:
            logging.error(f"An error occurred during scraping: {e}")

    elif args.mode.lower() == "cnn":
        logging.info("CNN mode selected. Building dataset and training the CNN model.")
        pos_urls = [args.url]
        neg_urls = []
        X, y, word_to_idx = build_dataset(pos_urls, neg_urls, proxy=args.proxy,
                                          max_len=args.max_len, vocab_size=args.vocab_size, model="CNN")
        dataset = ReviewDataset(X, y)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_obj = TextCNN(vocab_size=args.vocab_size)
        trainCNN(model_obj, dataset, args.epochs, device, args.batch_size, args.lr)
        additional_info = {"max_len": args.max_len, "vocab_size": args.vocab_size, "model_type": "CNN"}
        save_model_checkpoint(model_obj, word_to_idx, additional_info, filename="cnn_checkpoint.pth")

    elif args.mode.lower() == "fnn":
        logging.info("FNN mode selected. Building dataset and training the FNN model.")
        pos_urls = [args.url]
        neg_urls = []
        X, y, word_to_idx = build_dataset(pos_urls, neg_urls, proxy=args.proxy,
                                          max_len=args.max_len, vocab_size=args.vocab_size, model="FNN")
        dataset = ReviewDataset(X, y)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_size = X.shape[1]
        hidden_size = 128
        output_size = 2
        model_obj = TextFNN(input_size, hidden_size, output_size)
        trainFNN(model_obj, dataset, args.epochs, device, args.batch_size, args.lr)
        additional_info = {
            "input_size": input_size, "hidden_size": hidden_size,
            "output_size": output_size, "max_len": args.max_len,
            "word_to_idx": word_to_idx, "tags": ["positive", "negative"],
            "model_type": "FNN"
        }
        save_model_checkpoint(model_obj, word_to_idx, additional_info, filename="fnn_checkpoint.pth")
    else:
        logging.error("Invalid mode selected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Web scraping and HTML exploration with optional FNN/CNN training"
    )
    parser.add_argument('--mode', choices=['scrape', 'cnn', 'fnn'], default='scrape',
                        help="Choose 'scrape' to scrape HTML, 'cnn' for CNN training, or 'fnn' for FNN training.")
    parser.add_argument('--url', type=str, help="The URL to scrape or use for training")
    parser.add_argument('--json_file', type=str, help="Path to the JSON file for training data")
    parser.add_argument('--proxy', type=str, default=None, help="Optional proxy address for scraping")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--max_len', type=int, default=50, help="Maximum sequence length")
    parser.add_argument('--vocab_size', type=int, default=1000, help="Vocabulary size")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)
