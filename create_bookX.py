def convert_to_tsv(input_file, output_file):
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) >= 2:
                fout.write(f"{parts[0]}\t{parts[1]}\n")

dataset = "bookX"
input_path = f'./data/{dataset}/test.txt'
output_path = f'./data/{dataset}/test.tsv'

convert_to_tsv(input_path, output_path)