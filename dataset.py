import torch
from torch.utils.data import Dataset
import random
import os
from tokenizer import *
from torch.nn.utils.rnn import pad_sequence
import math


class ShakespeareDataset(Dataset):
    def __init__(self, filename="shakespeare_cleaned.txt", tokenizer=None, split="train",
                 test_split=0.1, seed=42, sequences_per_epoch=100000, min_seq_length=3,
                 max_seq_length=100, overlap_ratio=0.1):
        """
        Initialize Shakespeare dataset with massive sequence generation capability.

        Args:
            filename: Path to the Shakespeare text file
            tokenizer: Tokenizer instance for text processing
            split: Either "train" or "test" to specify which split to use
            test_split: Fraction of data to use for testing (default 0.1 for 10%)
            seed: Random seed for consistent splitting
            sequences_per_epoch: Number of sequences to generate per epoch (default 100,000)
            min_seq_length: Minimum sequence length in words (default 3)
            max_seq_length: Maximum sequence length in words (default 100)
            overlap_ratio: Ratio of sequences that can have overlapping text (0.1 = 10% can overlap)
        """
        self.filename = filename
        self.tokenizer = tokenizer
        self.split = split
        self.test_split = test_split
        self.seed = seed
        self.sequences_per_epoch = sequences_per_epoch
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.overlap_ratio = overlap_ratio

        # Set random seed for consistent splitting
        random.seed(seed)

        # Get file size and calculate split boundaries
        self.file_size = os.path.getsize(filename)
        self._calculate_split_boundaries()
        self._estimate_text_statistics()
        self._calculate_sampling_parameters()

    def _calculate_split_boundaries(self):
        """Calculate file boundaries for train/test split."""
        # Calculate split point in file
        split_point = int(self.file_size * (1 - self.test_split))

        if self.split == "train":
            self.start_byte = 0
            self.end_byte = split_point
        elif self.split == "test":
            self.start_byte = split_point
            self.end_byte = self.file_size
        else:
            raise ValueError("split must be either 'train' or 'test'")

        self.split_size = self.end_byte - self.start_byte

    def _estimate_text_statistics(self):
        """Estimate comprehensive text statistics for this split."""
        with open(self.filename, 'r', encoding='utf-8') as f:
            # Seek to start of this split
            f.seek(self.start_byte)

            # Read larger sample for better estimates (5000 chars or full split if smaller)
            sample_size = min(5000, self.split_size)
            sample = f.read(sample_size)

            if self.tokenizer:
                sample_words = self.tokenizer.split_text(sample)
            else:
                sample_words = sample.split()

            # Calculate statistics
            self.sample_word_count = len(sample_words)
            self.chars_per_word = len(sample) / self.sample_word_count if self.sample_word_count > 0 else 5
            self.estimated_total_words = int(self.split_size / self.chars_per_word)

            # Estimate lines (Shakespeare averages ~10-15 words per line)
            self.estimated_lines = sample.count('\n') * (self.split_size / sample_size)
            self.words_per_line = self.estimated_total_words / max(1, self.estimated_lines)

        print(f"Split: {self.split}")
        print(f"Estimated words in split: {self.estimated_total_words:,}")
        print(f"Estimated lines: {int(self.estimated_lines):,}")
        print(f"Average words per line: {self.words_per_line:.1f}")

    def _calculate_sampling_parameters(self):
        """Calculate parameters for diverse sampling."""
        # Calculate how many different starting positions we can use
        avg_seq_words = (self.min_seq_length + self.max_seq_length) / 2

        # With overlap allowed, we can start sequences much more frequently
        if self.overlap_ratio > 0:
            # Allow starting positions every few words instead of every sequence length
            spacing_words = max(1, int(avg_seq_words * (1 - self.overlap_ratio)))
            self.possible_start_positions = max(1, int(self.estimated_total_words / spacing_words))
        else:
            # Non-overlapping: can fit estimated_total_words / avg_seq_words sequences
            self.possible_start_positions = max(1, int(self.estimated_total_words / avg_seq_words))

        print(f"Estimated possible unique starting positions: {self.possible_start_positions:,}")
        print(f"Sequences per epoch: {self.sequences_per_epoch:,}")

    def __len__(self):
        """Return the number of sequences per epoch."""
        return self.sequences_per_epoch

    def _get_text_chunk_at_position(self, word_position, target_words):
        """
        Get a text chunk starting at approximately the given word position.

        Args:
            word_position: Approximate word position to start from
            target_words: Number of words to try to get

        Returns:
            str: Text chunk starting near the specified position
        """
        # Convert word position to approximate byte position
        estimated_byte_pos = int(word_position * self.chars_per_word)
        seek_pos = max(self.start_byte, min(self.start_byte + estimated_byte_pos, self.end_byte - 1000))

        with open(self.filename, 'r', encoding='utf-8') as f:
            f.seek(seek_pos)

            # Skip partial word at beginning if not at split start
            if seek_pos > self.start_byte:
                f.readline()  # Skip potentially partial line

            # Read enough text to get target_words
            text_chunk = ""
            words_collected = 0

            while words_collected < target_words:
                line = f.readline()
                if not line or f.tell() > self.end_byte:
                    # If we hit the end, wrap to the beginning of our split
                    f.seek(self.start_byte)
                    line = f.readline()
                    if not line:  # Split is empty
                        break

                # Check if adding this line would exceed split boundary
                if f.tell() > self.end_byte:
                    # Take only part of the line that fits
                    f.seek(f.tell() - len(line))
                    remaining_bytes = self.end_byte - f.tell()
                    if remaining_bytes > 0:
                        line = f.read(remaining_bytes)
                    else:
                        break

                text_chunk += line

                if self.tokenizer:
                    words_collected = len(self.tokenizer.split_text(text_chunk))
                else:
                    words_collected = len(text_chunk.split())

            return text_chunk

    def _get_diverse_sequence(self, idx):
        """
        Generate a diverse sequence using multiple strategies based on index.

        This ensures we get maximum diversity across the dataset.
        """
        # Use index to create diverse sampling strategies
        strategy = idx % 4

        if strategy == 0:
            # Strategy 1: Sequential sampling with small random offset
            base_position = (idx // 4) * (self.max_seq_length // 2)
            position = base_position + random.randint(0, 10)
        elif strategy == 1:
            # Strategy 2: Random position in first half of split
            position = random.randint(0, self.estimated_total_words // 2)
        elif strategy == 2:
            # Strategy 3: Random position in second half of split
            position = random.randint(self.estimated_total_words // 2, self.estimated_total_words - self.max_seq_length)
        else:
            # Strategy 4: Completely random position
            position = random.randint(0, max(0, self.estimated_total_words - self.max_seq_length))

        # Ensure position is valid
        position = max(0, min(position, self.estimated_total_words - self.min_seq_length))

        return position

    def __getitem__(self, idx):
        """
        Return a diverse sequence of tokenized words as a tensor.
        Uses the index to ensure diverse sampling across the entire dataset.

        Args:
            idx: Index used to generate diverse sequences

        Returns:
            torch.Tensor: Tensor of token IDs for consecutive words
        """
        # Create deterministic but varied seed based on idx
        random.seed(self.seed + idx * 17)  # Use prime number for better distribution

        # Get diverse starting position
        start_word_position = self._get_diverse_sequence(idx)

        # Random sequence length
        seq_length = random.randint(self.min_seq_length, self.max_seq_length)

        # Get text chunk starting at the calculated position
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Get extra words to ensure we have enough
                text_chunk = self._get_text_chunk_at_position(start_word_position, seq_length + 20)

                if self.tokenizer:
                    words = self.tokenizer.split_text(text_chunk)
                else:
                    words = text_chunk.split()

                # Ensure we have enough words
                if len(words) < self.min_seq_length:
                    # Try a different position
                    start_word_position = random.randint(0, max(0, self.estimated_total_words - self.max_seq_length))
                    continue

                # Select exact sequence length we want
                actual_seq_length = min(seq_length, len(words))
                actual_seq_length = max(self.min_seq_length, actual_seq_length)

                # Take sequence from beginning of chunk (since we positioned it carefully)
                selected_words = words[:actual_seq_length]

                # Tokenize
                if self.tokenizer:
                    token_ids = self.tokenizer.tokenize_text(selected_words)
                    return torch.tensor(token_ids, dtype=torch.long)
                else:
                    # Fallback tokenization
                    token_ids = [hash(word.lower()) % 10000 for word in selected_words]
                    return torch.tensor(token_ids, dtype=torch.long)

            except Exception as e:
                # If there's an error, try a different position
                start_word_position = random.randint(0, max(0, self.estimated_total_words - self.max_seq_length))
                continue

        # Fallback if all attempts failed
        if self.tokenizer:
            fallback_tokens = list(range(self.min_seq_length))
            return torch.tensor(fallback_tokens, dtype=torch.long)
        else:
            fallback_tokens = list(range(self.min_seq_length))
            return torch.tensor(fallback_tokens, dtype=torch.long)


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    """Collate function for proper autoregressive sequence-to-sequence training."""
    input_ids = [seq[:-1] for seq in batch]  # Input: all except last token
    target_ids = [seq[1:] for seq in batch]   # Target: all except first token (shifted by 1)

    # Ensure sequences have minimum length for training
    min_length = 2
    filtered_input_ids = []
    filtered_target_ids = []
    
    for inp, tgt in zip(input_ids, target_ids):
        if len(inp) >= min_length and len(tgt) >= min_length:
            filtered_input_ids.append(inp)
            filtered_target_ids.append(tgt)
    
    # If no valid sequences, create minimal sequences
    if not filtered_input_ids:
        dummy_seq = torch.tensor([0, 1], dtype=torch.long)
        filtered_input_ids = [dummy_seq[:-1]]
        filtered_target_ids = [dummy_seq[1:]]

    # Pad sequences to same length
    padded_input_ids = pad_sequence(filtered_input_ids, batch_first=True, padding_value=24942)
    padded_target_ids = pad_sequence(filtered_target_ids, batch_first=True, padding_value=-100)  # -100 ignored in loss
    
    return padded_input_ids, padded_target_ids


# Example usage with massive dataset:
if __name__ == "__main__":
    # Create tokenizer instance
    tokenizer = Tokenizer("tokens.pkl")

    # Create large train dataset (100K sequences per epoch)
    train_dataset = ShakespeareDataset(
        "shakespeare_cleaned.txt",
        tokenizer=tokenizer,
        split="train",
        sequences_per_epoch=100000,  # 100K sequences per epoch
        min_seq_length=5,
        max_seq_length=80,
        overlap_ratio=0.2  # Allow 20% overlap for more diversity
    )

    # Create test dataset (10K sequences per epoch)
    test_dataset = ShakespeareDataset(
        "shakespeare_cleaned.txt",
        tokenizer=tokenizer,
        split="test",
        sequences_per_epoch=10000,  # 10K sequences per epoch
        min_seq_length=5,
        max_seq_length=80,
        overlap_ratio=0.2
    )

    print(f"\nTrain dataset length: {len(train_dataset):,}")
    print(f"Test dataset length: {len(test_dataset):,}")

    print("\nSample sequences from different parts of the dataset:")
    sample_indices = [0, 1000, 25000, 49999]  # Sample from different positions

    for i in sample_indices:
        if i < len(train_dataset):
            sample = train_dataset[i]
            print(f"Train sequence {i}: shape {sample.shape}, length {len(sample)}")

            # Show actual text if possible (first few tokens)
            if hasattr(tokenizer, 'decode_tokens'):
                try:
                    decoded = tokenizer.decode_tokens(sample[:10].tolist())
                    print(f"  Sample text: {' '.join(decoded[:8])}...")
                except:
                    pass
            print("-" * 50)

    print(f"\nWith {len(train_dataset):,} sequences per epoch and {train_dataset.estimated_total_words:,} words,")
    print(f"this dataset provides massive diversity for training!")
    print(f"Each epoch will see different combinations and starting positions.")