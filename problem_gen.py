import random
import os

# Global Variables
# Pretty basic, num_problems needs to be figured out by hand, multipication is also available
num_problems = 19602  # Desired number of unique problems
max_digits = 2          # Maximum number of digits per number
ops = ['+','-']   # List of operators to use


def generate_problems(filename='problems.txt'):
    global ops, max_digits, num_problems  # Declare globals to use within the function

    existing_problems = set()

    # Check if the file exists and load existing problems
    if os.path.isfile(filename):
        print(f"Loading existing problems from {filename}...")
        with open(filename, 'r') as f:
            lines = f.readlines()
            # Problems are on even-indexed lines (0, 2, 4, ...)
            for i in range(0, len(lines), 2):
                problem = lines[i].strip()
                existing_problems.add(problem)
        print(f"Loaded {len(existing_problems)} existing problems.")
    else:
        print(f"{filename} does not exist. A new file will be created.")

    # Determine how many new unique problems need to be generated
    if len(existing_problems) >= num_problems:
        print(f"Already have {len(existing_problems)} problems, which meets or exceeds the requested {num_problems}.")
        return
    else:
        required = num_problems - len(existing_problems)
        print(f"Need to generate {required} new unique problems.")

    # Initialize operator counts
    op_counts = {op: 0 for op in ops}

    # Open the file in append mode to add new problems
    with open(filename, 'a') as f:
        new_count = 0
        attempts = 0
        while new_count < required:
            attempts += 1
            # Choose a random operator from the global ops list
            op = random.choice(ops)

            if op in ['+', '-', '*']:
                # Generate random number of digits for each number
                num1_digits = random.randint(1, max_digits)
                num2_digits = random.randint(1, max_digits)

                # Generate random numbers based on the number of digits
                num1 = random.randint(10**(num1_digits - 1), 10**num1_digits - 1)
                num2 = random.randint(10**(num2_digits - 1), 10**num2_digits - 1)

                # Formulate the problem string
                problem = f"{num1}{op}{num2}="

                # Calculate the answer based on the operator
                if op == '+':
                    answer_val = num1 + num2
                elif op == '-':
                    answer_val = num1 - num2
                elif op == '*':
                    answer_val = num1 * num2
            elif op == '/':
                # Special handling for division to ensure whole number results
                # Step 1: Generate num2
                num2_digits = random.randint(1, max_digits)
                num2 = random.randint(10**(num2_digits - 1), 10**num2_digits - 1)

                # Avoid division by zero just in case
                if num2 == 0:
                    continue

                # Step 2: Determine the maximum possible answer to keep num1 within max_digits
                max_answer = (10**max_digits - 1) // num2
                if max_answer < 1:
                    continue  # Skip if no valid answer can be generated

                # Step 3: Generate answer
                answer_val = random.randint(1, max_answer)

                # Step 4: Compute num1
                num1 = num2 * answer_val

                # Formulate the problem string
                problem = f"{num1}{op}{num2}="
            else:
                # In case of an unexpected operator, skip
                continue

            # Check for duplicates
            if problem in existing_problems:
                continue  # Skip duplicates without decrementing the required count

            # For division, answer_val is already computed
            # For other operations, answer_val is computed above
            answer = str(answer_val)

            # Write the problem and answer to the file
            f.write(problem + '\n')
            f.write(answer + '\n')

            # Update the set and counters
            existing_problems.add(problem)
            new_count += 1
            op_counts[op] += 1  # Increment the count for the current operator

            # Optional: Print progress every 10,000 new problems
            if new_count % 10000 == 0:
                print(f"Generated {new_count}/{required} new problems...")

        print(f"Successfully generated {new_count} new unique problems after {attempts} attempts.")

    # Print the breakdown of problems per operator
    print("\nBreakdown of generated problems per operator:")
    for op in ops:
        print(f"  {op}: {op_counts[op]}")

generate_problems()
