import pyarrow.dataset as ds
import pyarrow.csv as csv
import argparse
import sys


def list_columns(parquet_path):
    dataset = ds.dataset(parquet_path, format="parquet")
    return dataset.schema.names


def export_stream(parquet_path, columns, output_csv, batch_size=100_000):
    dataset = ds.dataset(parquet_path, format="parquet")

    scanner = dataset.scanner(
        columns=columns,
        batch_size=batch_size
    )

    first = True

    with open(output_csv, "wb") as f:
        for batch in scanner.to_batches():
            if first:
                csv.write_csv(batch, f)
                first = False
            else:
                csv.write_csv(
                    batch,
                    f,
                    write_options=csv.WriteOptions(include_header=False)
                )

            # if first:
            #     csv.write_csv(table, f)
            #     first = False
            # else:
            #     # tránh ghi header nhiều lần
            #     csv.write_csv(table, f, write_options=csv.WriteOptions(include_header=False))

    print(f"Done. Exported to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-c", "--columns", nargs="+")
    parser.add_argument("-o", "--output", default="output.csv")
    parser.add_argument("--batch", type=int, default=100_000)

    args = parser.parse_args()

    if not args.columns:
        cols = list_columns(args.input)
        print("Available columns:")
        for i, c in enumerate(cols):
            print(f"{i}: {c}")

        selected = input("Select indexes (comma): ")
        try:
            idx = [int(x.strip()) for x in selected.split(",")]
            args.columns = [cols[i] for i in idx]
        except:
            print("Invalid input")
            sys.exit(1)

    export_stream(args.input, args.columns, args.output, args.batch)


if __name__ == "__main__":
    main()