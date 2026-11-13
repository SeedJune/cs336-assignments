import argparse
from bpe_tokenizer import Tokenizer
import numpy as np

def process(data_path:str, save_path:str, tokenizer:Tokenizer):
    buffer = []
    with open(data_path, "r", encoding="utf-8", newline="") as r:
        with open(save_path, "wb") as w:
            for token_id in tokenizer.encode_iterable(r):
                buffer.append(token_id)

                if len(buffer) > 65536:
                    np.asarray(buffer, dtype=np.uint16).tofile(w)
                    buffer.clear()
            if buffer:
                np.asarray(buffer, dtype=np.uint16).tofile(w)
    return 

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_path", type=str)
    parse.add_argument("--vocab_path", type=str)
    parse.add_argument("--save_path", type=str)

    args = parse.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab_path, special_tokens=["<|endoftext|>"])

    process(args.data_path, args.save_path, tokenizer)
    print(f"Finish preprocess {args.data_path}.")