import csv

def convert_csv_to_tsv(input_csv, output_tsv):
    with open(input_csv, 'r', newline='') as csv_file:
        with open(output_tsv, 'w', newline='') as tsv_file:
            csv_reader = csv.reader(csv_file)
            tsv_writer = csv.writer(tsv_file, delimiter='\t')

            for row in csv_reader:
                tsv_writer.writerow(row)
    print(f"File convertito con successo: {output_tsv}")

convert_csv_to_tsv('data/gowalla_svdgcn/test_sparse.csv', 'data/gowalla_svdgcn/test_sparse.tsv')
