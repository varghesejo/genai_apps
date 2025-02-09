# 3. Act as a Python developer. Write code to read and print duplicate records
# from the provided CSV file

import csv

def find_and_print(csv_filepath, key_columns=None):
    """
    Finds and prints duplicate records from a CSV file.

    Args:
        csv_filepath: Path to the CSV file.
        key_columns: A list of column names (or indices) to consider for duplication.
                    If None, all columns are used.
    """

    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames:
                print("Empty or No header found in the CSV file.")
                return

            if key_columns is None:
                key_columns = reader.fieldnames
            elif all(isinstance(col, int) for col in key_columns): #check if index is provided
                if any(col >= len(reader.fieldnames) or col < 0 for col in key_columns):
                    raise IndexError("Index is out of range")
                key_columns = [reader.fieldnames[i] for i in key_columns]
            elif not all(col in reader.fieldnames for col in key_columns):
                raise ValueError("One or more key columsn not found id CSV header")

            seen = set()
            dupplicates = []
            for row in reader:
                key = tuple(row[col] for col in key_columns) #create a tuple of values from the key columns
                if key in seen:
                    dupplicates.append(row)
                else:
                    seen.add(key)
            
            if dupplicates:
                print("Duplicate records found:")
                # Print header once
                print(",".join(reader.fieldnames))
                printed_keys = set() #set to track the printed duplicates
                for duplicate in dupplicates:
                    key = tuple(duplicate[col] for col in key_columns)
                    if key not in printed_keys: # print only once
                        print(",".join(duplicate.values()))
                        printed_keys.add(key)
            else:
                print("No duplicate records found.")
    except FileNotFoundError:
        print(f"Error: File '{csv_filepath}' not found.")
    except csv.Error as e:
        print(f"CSV Error: {e}")
    except ValueError as e:
        print(e)
    except IndexError as e:
        print(e)
    except Exception as e:
        print(f"Error reading CSV file: {e}")

if __name__ == "__main__":
    csv_file = input("Enter the path to the CSV file: ")
    try:
        key_column_input = input("Enter the key columns(headers) to compare (comma-separated, or press Enter for all columns): ")
        if key_column_input:
            key_columns = [int(col.strip()) if col.strip() else col.strip() for col in key_column_input.split('.')]
        else:
            key_columns = None
        find_and_print(csv_file, key_columns)
    except Exception as e:
        print(f"Error: {e}")