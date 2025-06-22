import random

def add_lyric(new_lyric):
    """Appends a new lyric to the lyrics.txt file."""
    with open('data/lyrics.txt', 'a') as file:
        file.write(f'"{new_lyric}"\n')

def add_flow(new_flow):
    """Appends a new flow description to the flows.txt file."""
    with open('data/flows.txt', 'a') as file:
        file.write(f'"{new_flow}"\n')

