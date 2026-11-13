import regex as re
import os
from typing import BinaryIO
from multiprocessing import get_context
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from time import perf_counter


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_chunk(input_path, start, end, special_pattern, pat):
    counts = {}
    
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    parts = re.split(special_pattern, chunk) if special_pattern else [chunk]
    for doc in parts:
        for match in re.finditer(pat, doc):
            token = match.group(0)
            counts[token] = counts.get(token, 0) + 1
    
    return counts


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_processes: int = 2,
    chunk_size_mb: int = 32,
    verbose: bool = False,
):
    if num_processes < 1 or chunk_size_mb < 1:
        raise ValueError("num_processes and chunk_size_mb must be positive")
    started = perf_counter()
    # Initialize
    vocab = {}
    cur_size = 256
    merges = []
    
    for i in range(256):
        vocab[i] = bytes([i])
    for item in special_tokens:
        vocab[cur_size] = item.encode("utf-8")
        cur_size += 1

    # Pre-Tokenization
    pre_tokens = {}
    special_pattern = "|".join(re.escape(token) for token in special_tokens)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    with open(input_path, "rb") as f:
        file_size = os.fstat(f.fileno()).st_size
        num_chunks = max(1, (file_size + chunk_size_bytes - 1) // chunk_size_bytes)
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

    task = [(input_path, start, end, special_pattern, PAT) for start, end in zip(boundaries[:-1], boundaries[1:])]

    if verbose:
        print(
            f"[BPE] Pre-tokenization: {len(task)} chunks, "
            f"target={chunk_size_mb} MiB, workers={num_processes}",
            flush=True,
        )

    def accumulate(counts, completed):
        for token, count in counts.items():
            pre_tokens[token] = pre_tokens.get(token, 0) + count
        if verbose:
            print(
                f"[BPE] Counted {completed}/{len(task)} chunks; "
                f"unique pre-tokens={len(pre_tokens):,}; "
                f"elapsed={perf_counter() - started:.1f}s",
                flush=True,
            )

    if num_processes == 1 or len(task) <= 1:
        for completed, args in enumerate(task, 1):
            accumulate(count_chunk(*args), completed)
    else:
        # Limit outstanding tasks and merge each result immediately.
        # Unlike Pool.starmap, an abruptly exited worker raises an error here.
        with ProcessPoolExecutor(
            max_workers=num_processes,
            mp_context=get_context("forkserver"),
        ) as executor:
            remaining = iter(task)
            pending = set()
            for _ in range(min(num_processes, len(task))):
                pending.add(executor.submit(count_chunk, *next(remaining)))
            completed = 0
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.remove(future)
                    completed += 1
                    accumulate(future.result(), completed)
                    args = next(remaining, None)
                    if args is not None:
                        pending.add(executor.submit(count_chunk, *args))

    if verbose:
        print("[BPE] Pre-tokenization complete; building pair counts", flush=True)

    # Merge up
    token_sequences = {
        text: [bytes([value]) for value in text.encode("utf-8")]
        for text in pre_tokens
    }
    token_pair = Counter()
    for text, sequence in token_sequences.items():
        frequence = pre_tokens[text]
        for pair in zip(sequence, sequence[1:]):
            token_pair[pair] += frequence

    if verbose:
        print(f"[BPE] Merging toward vocabulary size {vocab_size:,}", flush=True)

    while cur_size < vocab_size and token_pair:
        # find the max
        best_pair = max(token_pair, key=lambda pair: (token_pair[pair], pair))
        left, right = best_pair
        merge_token = left + right
        vocab[cur_size] = merge_token
        cur_size += 1
        merges.append(best_pair)
        # update info
        for text, sequence in token_sequences.items():
            if best_pair not in zip(sequence, sequence[1:]):
                continue
            else:
                old_counts = Counter(zip(sequence, sequence[1:]))
                new_sequence = []
                length = len(sequence)
                i = 0
                while i < length:
                    if i + 1 < length and best_pair == (sequence[i], sequence[i + 1]):
                        new_sequence.append(merge_token)
                        i += 2
                    else:
                        new_sequence.append(sequence[i])
                        i += 1
                new_counts = Counter(zip(new_sequence, new_sequence[1:]))
                frequence = pre_tokens[text]
                for pair in new_counts | old_counts:
                    token_pair[pair] += frequence * (new_counts[pair] - old_counts[pair])
                    if token_pair[pair] == 0:
                        del token_pair[pair]
            token_sequences[text] = new_sequence

        if verbose and (len(merges) == 1 or len(merges) % 100 == 0):
            print(
                f"[BPE] Vocabulary {cur_size:,}/{vocab_size:,}; "
                f"merges={len(merges):,}; elapsed={perf_counter() - started:.1f}s",
                flush=True,
            )

    if verbose:
        print(
            f"[BPE] Complete: vocabulary={len(vocab):,}, "
            f"merges={len(merges):,}, elapsed={perf_counter() - started:.1f}s",
            flush=True,
        )

    return vocab, merges
