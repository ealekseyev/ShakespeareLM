import torch
import torch.nn.functional as F
from model_transformer_revised import ShakespeareLM
from tokenizer import Tokenizer
from time import sleep


model_path = "checkpoints/transformer_dev_e11_b250.pt"#"models/kafka_e980_checkpoint.pt"

def is_punctuation(word):
    return word in {'.', ',', ':', '!', ';', "'", '-', '?', '"'}


# Unicode + ASCII variants for each token class
_APOSTROPHES  = {"'", '\u2019', '\u2018'}          # ' ' '
_OPEN_QUOTES  = {'"', '\u201C', '\u00AB'}           # " « (unambiguous openers)
_CLOSE_QUOTES = {'"', '\u201D', '\u00BB'}           # " » (unambiguous closers)
_DASHES       = {'-', '\u2013', '\u2014', '\u2212'} # - – — −


def needs_space_before(prev_word, curr_word, quote_open):
    """Return True if a space should be printed before curr_word."""
    if prev_word is None:
        return False
    if curr_word in {'.', ',', ':', '!', '?', ';'}:
        return False
    if curr_word in {')', ']', '}'}:
        return False
    if prev_word in {'(', '[', '{'}:
        return False
    if curr_word in _APOSTROPHES or prev_word in _APOSTROPHES:
        return False
    if curr_word in _DASHES or prev_word in _DASHES:
        return False
    # Unambiguous Unicode closing quote — no space before
    if curr_word in _CLOSE_QUOTES:
        return False
    # Unambiguous Unicode opening quote — no space after
    if prev_word in _OPEN_QUOTES:
        return False
    # ASCII " toggle: quote_open is state *before* toggling for curr_word
    if curr_word == '"' and quote_open:      # closing
        return False
    if prev_word == '"' and quote_open:      # word after opening
        return False
    return True


def sample_with_rules(prob_distribution, previous_word, tokenizer, top_p=0.5):
    """
    Sample using top-p but keep sampling until rules are satisfied
    """
    probs = prob_distribution[0]  # Get first batch

    # Ensure probabilities are valid
    probs = torch.clamp(probs, min=1e-8)
    probs = probs / probs.sum()

    max_attempts = 50  # Prevent infinite loops

    for attempt in range(max_attempts):
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        # Find cutoff point where cumulative probability exceeds p
        cutoff_idx = torch.where(cumulative_probs > top_p)[0]
        if len(cutoff_idx) > 0:
            cutoff_idx = min(cutoff_idx[0].item(), len(sorted_probs) - 1)
        else:
            cutoff_idx = len(sorted_probs) - 1

        cutoff_idx = max(0, cutoff_idx)

        # Keep only top-p tokens
        top_p_probs = sorted_probs[:cutoff_idx + 1]
        top_p_indices = sorted_indices[:cutoff_idx + 1]

        # Renormalize probabilities
        top_p_probs = top_p_probs / top_p_probs.sum()

        # Sample from the filtered distribution
        try:
            sampled_idx = torch.multinomial(top_p_probs, 1).item()
            token_id = top_p_indices[sampled_idx].item()

            # Bounds check
            if token_id < 0 or token_id >= len(tokenizer.tokens):
                continue

            # Get the word
            try:
                word = tokenizer.untokenize_text([token_id])[0]
                if word == '???':
                    continue
            except:
                continue

            # Apply rules
            is_current_punct = is_punctuation(word)
            is_prev_punct = is_punctuation(previous_word) if previous_word else False

            # Rule 1: No adjacent punctuation
            if is_current_punct and is_prev_punct:
                continue

            # Rule 2: No duplicate words
            if word == previous_word:
                continue

            # If we get here, all rules are satisfied
            return token_id, word

        except:
            continue

    # If we can't find a valid token after max_attempts, return the most probable one
    return sorted_indices[0].item(), tokenizer.untokenize_text([sorted_indices[0].item()])[0]


def generate_text(model, tokenizer, prompt, max_length=100, top_p=0.5, device='cpu'):
    """
    Generate text using the trained model with formatting rules
    """
    model.eval()

    # Split the prompt into words (as expected by the tokenizer)
    words = tokenizer.split_text(prompt)

    # Tokenize the prompt
    tokens = tokenizer.tokenize_text(words)

    # Get vocab size from tokenizer and model
    tokenizer_vocab_size = len(tokenizer.tokens)
    model_vocab_size = model.embedding_layer.num_embeddings
    safe_vocab_size = min(tokenizer_vocab_size, model_vocab_size)

    # Handle unknown tokens (-1) and out of bounds tokens
    valid_tokens = []
    valid_words = []
    for i, token in enumerate(tokens):
        if token == -1:
            print(f"Warning: Unknown word '{words[i]}' found, skipping")
            continue
        elif token >= safe_vocab_size or token < 0:
            print(f"Warning: Token {token} is out of bounds (>= {safe_vocab_size}), skipping")
            continue
        else:
            valid_tokens.append(token)
            valid_words.append(words[i])

    if not valid_tokens:
        print("No valid tokens found. Using default start token.")
        valid_tokens = [0]
        valid_words = ["the"]

    # Start with the valid tokens
    current_tokens = valid_tokens.copy()
    generated_words = valid_words.copy()


    # Print initial prompt
    quote_open = False
    capitalize_next = True
    prev_display = None
    for word in valid_words:
        display_word = word
        if capitalize_next and not is_punctuation(word):
            display_word = 'I' if word.lower() == 'i' else word.capitalize()
            capitalize_next = False
        elif word.lower() == 'i' and not is_punctuation(word):
            display_word = 'I'

        if word in {'.', '!', '?'}:
            capitalize_next = True

        space = " " if needs_space_before(prev_display, word, quote_open) else ""
        print(space + display_word, end="", flush=True)
        prev_display = word
        if word == '"':   # toggle AFTER space decision
            quote_open = not quote_open
        #sleep(0.05)

    with torch.no_grad(), torch.autocast(device_type=device.type if hasattr(device, 'type') else device, dtype=torch.float16):
        for step in range(max_length):
            try:
                input_ids = torch.tensor([current_tokens], dtype=torch.long).to(device)
                logits = model(input_ids)
                last_logits = logits[0, -1, :]
                prob_distribution = F.softmax(last_logits.unsqueeze(0), dim=-1)

                previous_word = generated_words[-1] if generated_words else None
                next_token_id, next_word = sample_with_rules(
                    prob_distribution, previous_word, tokenizer, top_p
                )

                # Capitalize if needed
                display_word = next_word
                if capitalize_next and not is_punctuation(next_word):
                    display_word = 'I' if next_word.lower() == 'i' else next_word.capitalize()
                    capitalize_next = False
                elif next_word.lower() == 'i' and not is_punctuation(next_word):
                    display_word = 'I'

                if next_word in {'.', '!', '?'}:
                    capitalize_next = True

                space = " " if needs_space_before(prev_display, next_word, quote_open) else ""
                print(space + display_word, end="", flush=True)
                prev_display = next_word
                if next_word == '"':   # toggle AFTER space decision
                    quote_open = not quote_open

                #sleep(0.1)

                current_tokens.append(next_token_id)
                generated_words.append(next_word)

                max_seq_len = 100
                if len(current_tokens) > max_seq_len:
                    current_tokens = current_tokens[-max_seq_len:]
                    generated_words = generated_words[-max_seq_len:]

            except RuntimeError as e:
                print(f"\nError at step {step}: {e}")
                break

    return generated_words


def pick_checkpoint():
    """Ask the user for an epoch number and return the latest batch checkpoint for that epoch."""
    import os
    import re
    checkpoint_dir = "checkpoints"
    pattern = re.compile(r"transformer_dev_e(\d+)_b(\d+)\.pt")

    # Build mapping: epoch -> highest batch number seen
    epoch_max_batch = {}
    for f in os.listdir(checkpoint_dir):
        m = pattern.match(f)
        if m:
            e, b = int(m.group(1)), int(m.group(2))
            if e not in epoch_max_batch or b > epoch_max_batch[e]:
                epoch_max_batch[e] = b

    if not epoch_max_batch:
        print("No checkpoints found in checkpoints/.")
        return None

    available = sorted(epoch_max_batch)
    print(f"Available epochs: {available}")
    while True:
        raw = input("Enter epoch number to load: ").strip()
        try:
            epoch = int(raw)
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if epoch not in epoch_max_batch:
            print(f"Epoch {epoch} not found. Available: {available}")
            continue
        batch = epoch_max_batch[epoch]
        path = os.path.join(checkpoint_dir, f"transformer_dev_e{epoch}_b{batch}.pt")
        print(f"Loading e{epoch}_b{batch}")
        return path


def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer and model
    print("Loading tokenizer...")
    tokenizer = Tokenizer()

    # Get vocab size from tokenizer
    tokenizer_vocab_size = len(tokenizer.tokens)
    print(f"Tokenizer vocab size: {tokenizer_vocab_size}")

    selected_path = pick_checkpoint()
    if selected_path is None:
        return

    # First, try to load the model to check what vocab size it was trained with
    try:
        checkpoint = torch.load(selected_path, map_location=device)
        # Check the embedding layer size to determine the trained vocab size
        trained_vocab_size = checkpoint['embedding_layer.weight'].shape[0]
        print(f"Model was trained with vocab size: {trained_vocab_size}")

        if trained_vocab_size != tokenizer_vocab_size:
            print(f"Warning: Vocab size mismatch!")
            print(f"  Tokenizer: {tokenizer_vocab_size}")
            print(f"  Model: {trained_vocab_size}")
            print(f"Using model's vocab size: {trained_vocab_size}")
            vocab_size = trained_vocab_size
        else:
            vocab_size = tokenizer_vocab_size

    except FileNotFoundError:
        print(f"Error: Model file '{selected_path}' not found.")
        print("Please make sure you have a saved model file.")
        return
    except Exception as e:
        print(f"Error checking model file: {e}")
        print(f"Using tokenizer vocab size: {tokenizer_vocab_size}")
        vocab_size = tokenizer_vocab_size

    print("Initializing model...")
    model = ShakespeareLM(vocab_size=vocab_size).to(device)

    # Load the trained model
    try:
        print(f"Loading model from {selected_path}...")
        model.load_state_dict(checkpoint)
        model.half()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    while True:
        try:
            prompt = input("\nPrompt: ").strip()

            if prompt.lower() == 'quit':
                break

            if not prompt:
                continue

            try:
                max_length = int(input("Max length (default 100): ") or "100")
                top_p = float(input("Top-p (default 0.5): ") or "0.5")
            except ValueError:
                max_length = 100
                top_p = 0.5

            print()
            generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=max_length,
                top_p=top_p,
                device=device
            )
            print()

        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()