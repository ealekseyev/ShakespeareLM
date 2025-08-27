import torch
import torch.nn.functional as F
from model_transformer import ShakespeareLM
from tokenizer import Tokenizer
from time import sleep


model_path = "stlm_dev_e0_b1000.pt"

def is_punctuation(word):
    """Check if a word is punctuation that shouldn't be adjacent"""
    return word in ['.', ',', ':', '!', ';', "'", '-', '?']


def sample_with_rules(prob_distribution, previous_word, tokenizer, top_p=0.9):
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


def generate_text(model, tokenizer, prompt, max_length=100, top_p=0.9, device='cpu'):
    """
    Generate text using the trained model with formatting rules
    """
    model.eval()

    # Split the prompt into words (as expected by your tokenizer)
    words = tokenizer.split_text(prompt)
    print(f"Split prompt into words: {words}")

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

    print(f"Starting generation with prompt: '{prompt}'")
    print("Generated text:")

    # Print initial prompt with capitalization and formatting rules
    for i, word in enumerate(valid_words):
        # Handle special cases first
        display_word = word
        if word.lower() == 'i':  # Always capitalize standalone 'i'
            display_word = 'I'
        elif i == 0:  # First word
            display_word = word.capitalize()
        elif i > 0 and (valid_words[i-1] == '.' or valid_words[i-1] == '?' or valid_words[i-1] == '!'):  # Word after period
            display_word = word.capitalize()
        
        # Special spacing rules for initial prompt
        no_space_before = False
        if i > 0 and valid_words[i-1] == "'" and (word.lower() == 's' or word.lower() == 'd'):
            no_space_before = True
            
        if i == 0:
            print(display_word, end="", flush=True)
        elif is_punctuation(word):
            print(word, end="", flush=True)
            if word == '.':
                print()  # Newline after period
        else:
            if no_space_before:
                print(display_word, end="", flush=True)  # No space for contractions
            else:
                print(" " + display_word, end="", flush=True)
        
        # Small delay for initial prompt display
        sleep(0.05)

    with torch.no_grad():
        for step in range(max_length):
            try:
                # Convert current tokens to tensor
                input_ids = torch.tensor([current_tokens], dtype=torch.long).to(device)

                # Get model predictions
                logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
                
                # Get logits for the last position (next token prediction)
                last_logits = logits[0, -1, :]  # (vocab_size,)
                prob_distribution = F.softmax(last_logits.unsqueeze(0), dim=-1)  # (1, vocab_size)

                # Get previous word for rule checking
                previous_word = generated_words[-1] if generated_words else None

                # Sample with rules
                next_token_id, next_word = sample_with_rules(
                    prob_distribution, previous_word, tokenizer, top_p
                )

                # Handle special formatting rules
                should_capitalize = False
                if generated_words:
                    # Check if previous word was a period
                    if generated_words[-1] == '.':
                        should_capitalize = True
                    # Also capitalize at the very beginning
                elif not generated_words:
                    should_capitalize = True
                
                # Always capitalize standalone 'i'
                if next_word.lower() == 'i':
                    next_word = 'I'
                elif should_capitalize and not is_punctuation(next_word):
                    next_word = next_word.capitalize()

                # Special spacing rules
                no_space_before = False
                if generated_words and generated_words[-1] == "'" and (next_word.lower() == 's' or next_word.lower() == 'd'):
                    # No space between apostrophe and 's' (contractions like "it's", "that's")
                    no_space_before = True

                # Print the word immediately with proper formatting
                if is_punctuation(next_word):
                    print(next_word, end="", flush=True)
                    if next_word == '.':
                        print()  # Newline after period
                else:
                    if no_space_before:
                        print(next_word, end="", flush=True)  # No space before 's after apostrophe
                    else:
                        print(" " + next_word, end="", flush=True)
                
                # Brief pause for real-time effect
                sleep(0.1)

                # Add to our sequences (for model context)
                current_tokens.append(next_token_id)
                generated_words.append(next_word)

                # Limit sequence length
                max_seq_len = 100
                if len(current_tokens) > max_seq_len:
                    current_tokens = current_tokens[-max_seq_len:]
                    generated_words = generated_words[-max_seq_len:]

            except RuntimeError as e:
                print(f"\nError at step {step}: {e}")
                break

    print("\n\nGeneration complete!")
    return generated_words


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

    # First, try to load the model to check what vocab size it was trained with
    try:
        checkpoint = torch.load(model_path, map_location=device)
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
        print(f"Error: Model file '{model_path}' not found.")
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
        print(f"Loading model from {model_path}...")
        model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Interactive text generation
    print("\n" + "=" * 50)
    print("Shakespeare Text Generator")
    print("=" * 50)
    print("Enter a prompt to start generating Shakespeare-like text.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            # Get user input
            prompt = input("Enter your prompt: ").strip()

            if prompt.lower() == 'quit':
                print("Goodbye!")
                break

            if not prompt:
                print("Please enter a non-empty prompt.")
                continue

            # Ask for generation parameters
            try:
                max_length = int(input("Enter max length (default 100): ") or "100")
                top_p = float(input("Enter top-p value (default 0.9): ") or "0.9")
            except ValueError:
                print("Invalid input, using default values.")
                max_length = 100
                top_p = 0.9

            print(f"\nGenerating {max_length} tokens with top-p={top_p}...")
            print("-" * 50)

            # Generate text
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=max_length,
                top_p=top_p,
                device=device
            )

            print("-" * 50)
            print("Full generated text:")
            print(" ".join(generated_text))
            print("\n")

        except KeyboardInterrupt:
            print("\n\nGeneration interrupted. Type 'quit' to exit or enter a new prompt.")
        except Exception as e:
            print(f"An error occurred during generation: {e}")
            print("Please try again with a different prompt.\n")


if __name__ == "__main__":
    main()