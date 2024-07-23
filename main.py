import torch
import clip
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image
import numpy as np
import collections

# Global parameters
k = 100  # Controls the top-K words selection
max_length = 10  # Maximum length of the generated sentence

# Custom vocabulary
# There are too many common words,
# which leads to a large amount of action space,
# so in order to facilitate the demonstration,
# the scope is narrowed, and more common words need to be put in if the test coco is tested
custom_vocab = [
    "dog", "cat", "bird", "gorilla", "elephant", "lion", "tiger", "bear", "fox", "wolf",
    "sit", "stand", "stop", "stay", "run", "jump", "walk", "eat", "drink", "sleep",
    "Black", "White", "Gray", "Red", "Green", "Blue", "Yellow", "Brown", "Purple", "Pink",
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "all",
    "smoking", "which", "it", "is", "sitting", "lying", "standing", "running", "jumping", "playing",
    "on", "in", "under", "above", "beside", "between", "behind", "in front of", "near", "next to",
    "branch", "tree", "ground", "sky", "water", "river", "lake", "mountain", "hill", "field",
    "with", "without", "and", "or", "but", "because", "so", "if", "when", "while",
    "he", "she", "they", "we", "I", "you", "it", "my", "your", "his", "her", "their", "our",
    "happy", "sad", "angry", "excited", "bored", "tired", "hungry", "thirsty", "scared", "brave",
    "big", "small", "large", "tiny", "huge", "short", "tall", "long", "wide", "narrow"
]

# Load GPT-2 model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

# Load CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Encode the image
image = Image.open("1.jpg")  # Replace with your own image path
image_inputs = preprocess(image).unsqueeze(0).to(device)
image_features = clip_model.encode_image(image_inputs).detach()  # Use detach() to avoid computation graph


# Define the state class for MCTS nodes
class State:
    def __init__(self, image_features, text="A photo of"):
        self.image_features = image_features
        self.text = text
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.simulation_cache = None  # Cache for simulation results

    # Check if the state is terminal
    def is_terminal(self):
        return "<END>" in self.text or len(self.text.split()) >= max_length

    # Expand the current node by generating all possible next words
    def expand(self):
        if not self.children:
            top_k_tokens = self.get_top_k_tokens(k)
            for token in top_k_tokens:
                new_text = self.text + " " + tokenizer.decode([token]).strip()
                if new_text.strip():  # Ensure the new text is not empty
                    self.children.append(State(self.image_features, new_text))

    # Get the top-K tokens predicted by GPT-2 model, filtered by custom vocabulary
    def get_top_k_tokens(self, k):
        inputs = tokenizer(self.text, return_tensors='pt').to(device)
        outputs = gpt_model(**inputs)
        logits = outputs.logits[:, -1, :]

        # Set logits of words not in the custom vocabulary to negative infinity
        for i, token in enumerate(tokenizer.get_vocab()):
            if tokenizer.decode([i]).strip() not in custom_vocab:
                logits[:, i] = float('-inf')

        top_k_tokens = torch.topk(logits, k).indices[0].tolist()
        return top_k_tokens

    # Simulate from the current state to generate a complete sentence and score it
    def simulate(self):
        if self.simulation_cache is not None:
            return self.simulation_cache

        final_text = self.text
        while not self.is_terminal():
            next_token = self.sample_next_token()
            next_word = tokenizer.decode(next_token.item()).strip()
            if next_word:  # Ensure the generated word is not empty
                final_text += " " + next_word
            if len(final_text.split()) >= max_length:  # Add length limit
                break
        score = clip_score(self.image_features, final_text)
        self.simulation_cache = score
        return score

    # Sample the next token
    def sample_next_token(self):
        inputs = tokenizer(self.text, return_tensors='pt').to(device)
        outputs = gpt_model(**inputs)
        logits = outputs.logits[:, -1, :]

        # Set logits of words not in the custom vocabulary to negative infinity
        for i, token in enumerate(tokenizer.get_vocab()):
            if tokenizer.decode([i]).strip() not in custom_vocab:
                logits[:, i] = float('-inf')

        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        return next_token

    # Select the best child node based on UCB1 algorithm
    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / (child.visits + 1)) + c_param * np.sqrt((2 * np.log(self.visits + 1)) / (child.visits + 1))
            for child in self.children
        ]
        best_index = np.argmax(choices_weights)
        return self.children[best_index]


# Core algorithm for MCTS
def mcts(root, n_simulations):
    statistics = collections.defaultdict(list)  # Collect statistics

    for i in range(n_simulations):
        node, path = select_and_expand(root)
        reward = node.simulate()
        backpropagate(path, reward)

        # Collect statistics
        first_word = path[0].text.split()[3]  # The first word is the fourth word in "A photo of ..."
        statistics[first_word].append(reward)

    # Calculate the mean score for each first word
    mean_scores = {word: np.mean(scores) for word, scores in statistics.items()}
    return root.best_child(c_param=0), mean_scores


# Selection and expansion steps
def select_and_expand(root):
    node = root
    path = []
    while node.children:
        node = node.best_child()
        path.append(node)
    if not node.is_terminal():
        node.expand()
        node = node.children[0]
        path.append(node)
    return node, path


# Backpropagation step
def backpropagate(path, reward):
    for node in reversed(path):
        node.visits += 1
        node.value += reward


# Score the image and text using the CLIP model
def clip_score(image_features, text):
    text_inputs = clip.tokenize([text]).to(device)
    text_features = clip_model.encode_text(text_inputs).detach()  # Use detach() to avoid computation graph
    score = torch.cosine_similarity(image_features, text_features)
    return score.item()


# Calculate the advantage function
def calculate_advantage(mean_scores, root):
    total_mean_score = np.mean(list(mean_scores.values()))
    advantages = {word: score - total_mean_score for word, score in mean_scores.items()}
    # Set advantage values of words not in top-K to 0
    for node in root.children:
        first_word = node.text.split()[3]
        if first_word not in advantages:
            advantages[first_word] = 0
    return advantages


# Update the posterior strategy based on the advantage function
def update_posterior_strategy(advantages, root, beta=1.0):
    for node in root.children:
        first_word = node.text.split()[3]  # The first word is the fourth word in "A photo of ..."
        if first_word in advantages:
            advantage = advantages[first_word]
            probability = 1 / (1 + np.exp(-beta * advantage))
            node.value = probability * node.value


# Multiple iterations to continuously optimize the strategy
simulations_per_step = 100  # Fixed number of simulations

# Initialize the root node
root = State(image_features)

while not root.is_terminal():
    # Perform MCTS to select the next word and collect statistics
    best_node, mean_scores = mcts(root, n_simulations=simulations_per_step)

    # Calculate the advantage function
    advantages = calculate_advantage(mean_scores, root)

    # Update the strategy
    update_posterior_strategy(advantages, root)

    # Update the root node to the best selected child node
    root = best_node

# Output the final generated text
print("Final Generated Text:", root.text)
