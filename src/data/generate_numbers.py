from num2words import num2words

def gen_numbers(ulimit: int, out_filename: str) -> None:
    words = []
    numbers = []
    for i in range(1, ulimit + 1):
        numbers.append(i)
        word = num2words(i).replace("-", " ").replace(",", "")
        words.append(word)
    
    with open(out_filename, "w") as f:
        for i, w in zip(numbers, words):
            f.write(f"{i},{w}\n")

if __name__ == "__main__":
    gen_numbers(100000, "./data/num_list_en.csv")